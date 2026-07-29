#!/usr/bin/env python3
"""
Multi-horizon bake-off (v1.6, recursive rollout of 1-step models).

Trains the same 1-step models as eval_same_features_compare.py, then
recursively forecasts H=14 days ahead with known-future calendar/holidays
and predicted demand fed back into lags/intermittent state.

Reports metrics at h=1,7,14 and mean over 1..H.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.losses import three_term_loss_config
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from eval_helpers import (
    build_deepar,
    build_tft,
    build_transformer,
    filter_aligned,
    split_components,
    train_mase_scale,
    train_volume_terciles,
)
from multihorizon_rollout import (
    build_sku_timelines,
    collect_origins,
    horizon_metrics,
    rollout_sequence,
    rollout_tabular,
)

ALL_MODELS = (
    "deepsequence",
    "lightgbm",
    "deepar_lite",
    "temporal_transformer",
    "tft_lite",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_origins_per_sku",
        type=int,
        default=8,
        help="Cap test origins per SKU for runtime (None = all).",
    )
    p.add_argument("--models", default=",".join(ALL_MODELS))
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_multihorizon_v16.json"),
    )
    return p.parse_args()


def train_seq_three_term(model, Xtr, ytr, skutr, Xva, yva, skuva, zero_rate, args, label):
    cfg = three_term_loss_config(zero_rate, alpha_bce=0.2, w_gated=1.0, w_mag=1.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0025),
        loss=cfg["losses"],
        loss_weights=cfg["weights"],
    )
    ytr_d = {
        "final_forecast": ytr.reshape(-1, 1),
        "base_forecast": ytr.reshape(-1, 1),
        "non_zero_probability": (ytr > 0).astype(np.float32).reshape(-1, 1),
    }
    yva_d = {
        "final_forecast": yva.reshape(-1, 1),
        "base_forecast": yva.reshape(-1, 1),
        "non_zero_probability": (yva > 0).astype(np.float32).reshape(-1, 1),
    }
    print(f"\n=== train {label} ===")
    t0 = time.time()
    model.fit(
        [Xtr, skutr],
        ytr_d,
        validation_data=([Xva, skuva], yva_d),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
            ),
        ],
        verbose=2,
    )
    return time.time() - t0


def build_1step_windows(train_df, val_df, X_train, X_val, lookback: int):
    """1-step windows for sequence training (history ends before target day)."""
    metas, feats, offset = [], [], 0
    for split, df, X in (("train", train_df, X_train), ("val", val_df, X_val)):
        y = df["Quantity"].to_numpy(np.float32).reshape(-1, 1)
        block = np.concatenate([y, X.astype(np.float32)], axis=1)
        feats.append(block)
        n = len(df)
        metas.append(
            pd.DataFrame(
                {
                    "id_var": df["id_var"].astype(str).to_numpy(),
                    "ds": pd.to_datetime(df["ds"]),
                    "y": df["Quantity"].to_numpy(np.float32),
                    "split": split,
                    "_pos": np.arange(offset, offset + n, dtype=np.int64),
                }
            )
        )
        offset += n
    feat_all = np.concatenate(feats, axis=0)
    meta = pd.concat(metas, ignore_index=True)
    meta = meta.sort_values(["id_var", "ds"], kind="mergesort").reset_index(drop=True)
    n_channels = feat_all.shape[1]
    xs, ys, skus, splits = [], [], [], []
    for sku, g in meta.groupby("id_var", sort=False):
        pos = g["_pos"].to_numpy()
        arr = feat_all[pos]
        y = g["y"].to_numpy(np.float32)
        sp = g["split"].to_numpy()
        n = len(g)
        if n <= lookback:
            continue
        for t in range(lookback, n):
            xs.append(arr[t - lookback : t])
            ys.append(y[t])
            skus.append(sku)
            splits.append(sp[t])
    X = np.stack(xs).astype(np.float32) if xs else np.zeros((0, lookback, n_channels), np.float32)
    return X, np.asarray(ys, np.float32), np.asarray(skus), np.asarray(splits), n_channels


def main():
    args = parse_args()
    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    unknown = selected - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"Unknown --models: {sorted(unknown)}")

    tf.keras.utils.set_random_seed(args.seed)
    data_dir_raw = args.data_dir or os.environ.get("DEEPSEQUENCE_DATA_DIR")
    if not data_dir_raw:
        raise SystemExit("Pass --data_dir or set DEEPSEQUENCE_DATA_DIR")
    data_dir = Path(data_dir_raw)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    h_tr = pd.read_csv(data_dir / "holiday_features_train.csv")
    h_va = pd.read_csv(data_dir / "holiday_features_val.csv")
    h_te = pd.read_csv(data_dir / "holiday_features_test.csv")

    rng = np.random.default_rng(args.seed)
    chosen = set(
        rng.choice(
            train_df["id_var"].unique(),
            size=min(args.max_skus, train_df["id_var"].nunique()),
            replace=False,
        )
    )
    train_df, h_tr = filter_aligned(train_df, h_tr, chosen)
    val_df, h_va = filter_aligned(val_df, h_va, chosen)
    test_df, h_te = filter_aligned(test_df, h_te, chosen)

    volume_map, volume_stats = train_volume_terciles(train_df)
    mase_scale = train_mase_scale(train_df, season=7)
    print(f"MASE scale (train seasonal-naive |y_t-y_{{t-7}}| mean): {mase_scale}")
    cats = pd.Categorical(train_df["id_var"].astype(str))
    sku_map = {str(k): i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return df["id_var"].astype(str).map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val = enc(train_df), enc(val_df)
    zero_rate = float((y_train == 0).mean())

    cfg = load_feature_config()
    print("Building causal features...")
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, _ = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
    feature_names = list(Xtr_df.columns)
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    t_idx = cfg.trend_indices[0]
    tmin = float(X_train[:, t_idx].min())
    tmax = float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    # Keep raw epoch days for rollout; normalize copy for training
    # Recompute raw from dates for train scaling only
    epoch = pd.Timestamp("1970-01-01")
    raw_tr = (pd.to_datetime(train_df["ds"]) - epoch).dt.days.to_numpy(np.float64)
    tmin_raw, tmax_raw = float(raw_tr.min()), float(raw_tr.max())
    span_raw = max(tmax_raw - tmin_raw, 1.0)

    X_train_n = X_train.copy()
    X_val_n = X_val.copy()
    X_test_n = X_test.copy()
    for X in (X_train_n, X_val_n, X_test_n):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    tr, va = split_components(X_train_n, cfg), split_components(X_val_n, cfg)
    lag_periods = cfg.lag_periods

    results = {
        "config": {
            "protocol": "recursive_rollout_after_origin",
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "seed": args.seed,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "report_horizons": [1, 7, 14],
            "max_origins_per_sku": args.max_origins_per_sku,
            "zero_rate": zero_rate,
            "feature_contract": f"feature_config v{cfg.config['metadata']['version']}",
            "feature_names": feature_names,
            "volume_stats": volume_stats,
            "models": sorted(selected),
            "note": (
                "After observing day t, forecast t+1..t+H. Known-future calendar/holidays; "
                "recursive demand into lags/intermittent. Same 1-step models as v1.6 bake-off."
            ),
        },
        "models": {},
    }

    # ------------------------------------------------------------------
    # Train DeepSequence
    # ------------------------------------------------------------------
    ds_model = None
    if "deepsequence" in selected:
        print("\n=== DeepSequence train ===")
        base = build_hierarchical_model_lightweight(
            n_temporal_features=len(cfg.trend_indices),
            n_fourier_features=len(cfg.seasonal_indices),
            n_holiday_features=len(cfg.holiday_indices),
            n_lag_features=len(cfg.regressor_indices),
            n_skus=n_skus,
            hidden_dim=48,
            sku_embedding_dim=4,
            dropout_rate=0.23,
            use_cross_layers=True,
            use_intermittent=True,
            n_changepoints=15,
        )
        _ = base(
            [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
            training=False,
        )
        pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))
        ds_model = AdaptiveWeightedModel(
            base_model=base,
            bce_loss_fn=WeightedBCELoss(weight_nonzero=pos_weight, weight_zero=1.0),
            mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            zero_rate=zero_rate,
            avg_nonzero_demand=float(y_train[y_train > 0].mean()),
            pos_weight=pos_weight,
            loss_recipe="three_term",
            alpha_bce=0.2,
            w_gated=1.0,
            w_mag=1.0,
            use_fixed_weights=True,
        )
        ds_model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
        ytr = {"final_forecast": y_train.reshape(-1, 1), "base_forecast": y_train.reshape(-1, 1)}
        yva = {"final_forecast": y_val.reshape(-1, 1), "base_forecast": y_val.reshape(-1, 1)}
        t0 = time.time()
        ds_model.fit(
            [*tr, sku_train],
            ytr,
            validation_data=([*va, sku_val], yva),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
                )
            ],
            verbose=2,
        )
        results["models"].setdefault("deepsequence", {})["train_seconds"] = time.time() - t0

    # ------------------------------------------------------------------
    # Train LightGBM
    # ------------------------------------------------------------------
    lgb_model = None
    if "lightgbm" in selected:
        print("\n=== LightGBM train ===")
        Xlgb_tr = np.concatenate([X_train_n, sku_train.astype(np.float32)], axis=1)
        Xlgb_va = np.concatenate([X_val_n, sku_val.astype(np.float32)], axis=1)
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            n_jobs=-1,
        )
        t0 = time.time()
        lgb_model.fit(
            Xlgb_tr,
            y_train,
            eval_set=[(Xlgb_va, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        results["models"].setdefault("lightgbm", {})["train_seconds"] = time.time() - t0

    # ------------------------------------------------------------------
    # Train sequence models
    # ------------------------------------------------------------------
    seq_models = {}
    need_seq = bool(selected & {"deepar_lite", "temporal_transformer", "tft_lite"})
    if need_seq:
        print("\nBuilding 1-step sequence windows for training...")
        Xseq, yseq, sku_seq_raw, split_seq, n_channels = build_1step_windows(
            train_df, val_df, X_train_n, X_val_n, args.lookback
        )
        sku_seq = np.array([sku_map[str(s)] for s in sku_seq_raw], dtype=np.int32).reshape(-1, 1)
        m_tr = split_seq == "train"
        m_va = split_seq == "val"
        builders = {
            "deepar_lite": build_deepar,
            "temporal_transformer": build_transformer,
            "tft_lite": build_tft,
        }
        for name, builder in builders.items():
            if name not in selected:
                continue
            model = builder(args.lookback, n_skus, n_channels=n_channels)
            train_s = train_seq_three_term(
                model,
                Xseq[m_tr],
                yseq[m_tr],
                sku_seq[m_tr],
                Xseq[m_va],
                yseq[m_va],
                sku_seq[m_va],
                zero_rate,
                args,
                name,
            )
            seq_models[name] = model
            results["models"].setdefault(name, {})["train_seconds"] = train_s
            results["models"][name]["n_channels"] = n_channels

    # ------------------------------------------------------------------
    # Build timelines + origins (test days as origins)
    # ------------------------------------------------------------------
    print("\nBuilding timelines for recursive rollout...")
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    panel = pd.concat([train_df, val_df, test_df], ignore_index=True)
    hol = pd.concat([h_tr, h_va, h_te], ignore_index=True)
    timelines = build_sku_timelines(panel, hol, cfg.holiday_names)

    origin_mask: dict = {}
    for sku, g in panel.groupby(panel["id_var"].astype(str), sort=False):
        g = g.sort_values("ds", kind="mergesort")
        origin_mask[str(sku)] = (g["split"].to_numpy() == "test")

    origins = collect_origins(
        timelines,
        sku_map,
        horizon=args.horizon,
        origin_split_mask=origin_mask,
        max_origins_per_sku=args.max_origins_per_sku,
        seed=args.seed,
    )
    print(f"origins={len(origins)} horizon={args.horizon}")
    results["config"]["n_origins"] = len(origins)

    def ds_predict(X, sku):
        parts = split_components(X, cfg)
        pred = ds_model.predict([*parts, sku], batch_size=4096, verbose=0)
        return (
            np.asarray(pred["final_forecast"]).reshape(-1),
            np.asarray(pred["non_zero_probability"]).reshape(-1),
        )

    def lgb_predict(X, sku):
        Xin = np.concatenate([X, sku.astype(np.float32)], axis=1)
        yh = np.maximum(lgb_model.predict(Xin), 0.0).astype(np.float32)
        p = np.clip(1.0 - np.exp(-yh), 0, 1).astype(np.float32)
        return yh, p

    def make_seq_predict(model):
        def _pred(windows, sku):
            pred = model.predict([windows, sku], batch_size=2048, verbose=0)
            return (
                np.asarray(pred["final_forecast"]).reshape(-1),
                np.asarray(pred["non_zero_probability"]).reshape(-1),
            )

        return _pred

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------
    if "deepsequence" in selected and ds_model is not None:
        print("\n=== DeepSequence multi-horizon rollout ===")
        t0 = time.time()
        roll = rollout_tabular(
            timelines,
            origins,
            sku_map,
            ds_predict,
            lag_periods,
            tmin_raw,
            span_raw,
            args.horizon,
        )
        metrics = horizon_metrics(
            roll["y_true"],
            roll["yhat"],
            roll["p"],
            roll["skus"],
            volume_map,
            mase_scale=mase_scale,
        )
        results["models"]["deepsequence"].update(
            {"rollout_seconds": time.time() - t0, **metrics}
        )

    if "lightgbm" in selected and lgb_model is not None:
        print("\n=== LightGBM multi-horizon rollout ===")
        t0 = time.time()
        roll = rollout_tabular(
            timelines,
            origins,
            sku_map,
            lgb_predict,
            lag_periods,
            tmin_raw,
            span_raw,
            args.horizon,
        )
        metrics = horizon_metrics(
            roll["y_true"],
            roll["yhat"],
            roll["p"],
            roll["skus"],
            volume_map,
            mase_scale=mase_scale,
        )
        results["models"]["lightgbm"].update(
            {"rollout_seconds": time.time() - t0, **metrics}
        )

    for name, model in seq_models.items():
        print(f"\n=== {name} multi-horizon rollout ===")
        t0 = time.time()
        roll = rollout_sequence(
            timelines,
            origins,
            sku_map,
            make_seq_predict(model),
            lag_periods,
            tmin_raw,
            span_raw,
            args.horizon,
            args.lookback,
        )
        metrics = horizon_metrics(
            roll["y_true"],
            roll["yhat"],
            roll["p"],
            roll["skus"],
            volume_map,
            mase_scale=mase_scale,
        )
        results["models"][name].update(
            {"rollout_seconds": time.time() - t0, **metrics}
        )

    # ------------------------------------------------------------------
    # Comparison tables
    # ------------------------------------------------------------------
    results["mase_scale_season7"] = mase_scale
    comparison = {}
    for key in ("1", "7", "14", "mean"):
        comparison[key] = []
        for model, payload in results["models"].items():
            if key == "mean":
                block = payload.get("mean_1_to_H", {}).get("overall", {})
            else:
                block = payload.get("by_horizon", {}).get(key, {}).get("overall", {})
            if not block:
                continue
            comparison[key].append(
                {
                    "model": model,
                    "mae_rounded": block.get("mae_all_rounded"),
                    "mae_nonzero": block.get("mae_nonzero"),
                    "iwmae_rounded": block.get("iwmae_rounded"),
                    "mase_rounded": block.get("mase_rounded"),
                    "occ_f1": block.get("occ_f1"),
                    "underforecast_rate_nonzero": block.get(
                        "underforecast_rate_nonzero"
                    ),
                    "bias": block.get("bias"),
                    "bias_nonzero": block.get("bias_nonzero"),
                    "aucroc": block.get("aucroc"),
                }
            )
        comparison[key] = sorted(
            comparison[key],
            key=lambda r: (
                r["iwmae_rounded"] is None,
                r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
            ),
        )
    results["comparison"] = comparison

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("MULTI-HORIZON COMPARISON (recursive; primary sort: iwmae_rounded)")
    print("=" * 70)
    for key in ("1", "7", "14", "mean"):
        print(f"\n[h={key}]")
        for row in comparison[key]:
            print(
                f"  {row['model']:28s} iwmae={row['iwmae_rounded']:.3f} "
                f"mae={row['mae_rounded']:.3f} nz={row['mae_nonzero']:.3f} "
                f"mase={row.get('mase_rounded')} occ_f1={row.get('occ_f1')} "
                f"under={row.get('underforecast_rate_nonzero')} "
                f"bias={row['bias']:.3f}"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
