#!/usr/bin/env python3
"""
DeepSequence improvement bake-off: 1-step recursive vs direct multi-horizon.

Trains:
  - ds_h1: classic horizon=1 (recursive eval)
  - ds_mh: direct Dense(H) gated head (direct eval)
  - ds_mh_tuned: same head with stronger BCE/mag + horizon decay

Primary ranking: IWMAE on recursive/direct H-step forecasts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.losses import weighted_bce_loss, masked_mae_loss
from train_lightweight_adaptive_loss import AdaptiveWeightedModel
from eval_helpers import (
    filter_aligned,
    split_components,
    train_mase_scale,
    train_volume_terciles,
)
from multihorizon_rollout import (
    build_sku_timelines,
    collect_origins,
    horizon_metrics,
    rollout_direct_tabular,
    rollout_tabular,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_origins_per_sku", type=int, default=8)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_ds_mh_improve_v16.json"),
    )
    return p.parse_args()


def build_mh_xy(X, y, skus, horizon: int):
    """Per-series sliding windows: X[t] -> y[t:t+H]."""
    H = int(horizon)
    xs, ys, ss = [], [], []
    skus = np.asarray(skus)
    y = np.asarray(y, np.float32)
    for sku in np.unique(skus):
        idx = np.where(skus == sku)[0]
        if len(idx) < H:
            continue
        for i in range(len(idx) - H + 1):
            sl = idx[i : i + H]
            xs.append(X[sl[0]])
            ys.append(y[sl])
            ss.append(sku)
    return (
        np.asarray(xs, np.float32),
        np.asarray(ys, np.float32),
        np.asarray(ss),
    )


def train_ds(
    *,
    horizon,
    tr,
    va,
    y_tr,
    y_va,
    sku_tr,
    sku_va,
    n_skus,
    zero_rate,
    avg_nz,
    epochs,
    batch_size,
    alpha_bce=0.2,
    w_gated=1.0,
    w_mag=1.0,
    horizon_decay=1.0,
    label="ds",
):
    pos_weight = min(20.0, zero_rate / max(1.0 - zero_rate, 1e-6))
    base = build_hierarchical_model_lightweight(
        n_temporal_features=tr[0].shape[1],
        n_fourier_features=tr[1].shape[1],
        n_holiday_features=tr[2].shape[1],
        n_lag_features=tr[3].shape[1],
        n_skus=n_skus,
        hidden_dim=48,
        sku_embedding_dim=4,
        dropout_rate=0.23,
        use_cross_layers=True,
        use_intermittent=True,
        n_changepoints=15,
        horizon=horizon,
    )
    # Build once
    _ = base(
        [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
        training=False,
    )
    model = AdaptiveWeightedModel(
        base,
        bce_loss_fn=weighted_bce_loss(pos_weight=pos_weight),
        mae_loss_fn=masked_mae_loss(),
        zero_rate=zero_rate,
        avg_nonzero_demand=avg_nz,
        pos_weight=pos_weight,
        use_fixed_weights=True,
        loss_recipe="three_term",
        alpha_bce=alpha_bce,
        w_gated=w_gated,
        w_mag=w_mag,
        horizon_decay=horizon_decay,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
    ytr = {
        "final_forecast": y_tr,
        "base_forecast": y_tr,
        "non_zero_probability": (y_tr > 0).astype(np.float32),
    }
    yva = {
        "final_forecast": y_va,
        "base_forecast": y_va,
        "non_zero_probability": (y_va > 0).astype(np.float32),
    }
    print(f"\n=== train {label} horizon={horizon} alpha_bce={alpha_bce} w_mag={w_mag} decay={horizon_decay} ===")
    t0 = time.time()
    model.fit(
        [*tr, sku_tr],
        ytr,
        validation_data=([*va, sku_va], yva),
        epochs=epochs,
        batch_size=batch_size,
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
    return model, time.time() - t0


def calibrate_bias(y_true, yhat, grid=None):
    """Simple multiplicative bias on final forecast to minimize IWMAE on val."""
    if grid is None:
        grid = np.linspace(0.7, 1.3, 25)
    y = y_true.reshape(-1)
    yh0 = np.maximum(yhat.reshape(-1), 0.0)
    nz = y > 0
    z = ~nz
    if not (0 < nz.mean() < 1):
        return 1.0
    w = np.where(nz, 1.0 / nz.mean(), 1.0 / z.mean())
    best_s, best = 1.0, 1e18
    for s in grid:
        e = np.average(np.abs(y - yh0 * s), weights=w)
        if e < best:
            best, best_s = e, float(s)
    return best_s


def main():
    args = parse_args()
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
    cats = pd.Categorical(train_df["id_var"].astype(str))
    sku_map = {str(k): i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return (
            df["id_var"]
            .astype(str)
            .map(sku_map)
            .astype(np.int32)
            .to_numpy()
            .reshape(-1, 1)
        )

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val = enc(train_df), enc(val_df)
    zero_rate = float((y_train == 0).mean())
    avg_nz = float(y_train[y_train > 0].mean()) if (y_train > 0).any() else 1.0

    cfg = load_feature_config()
    print("Building causal features...")
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    t_idx = cfg.trend_indices[0]
    tmin = float(X_train[:, t_idx].min())
    tmax = float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    tr = split_components(X_train, cfg)
    va = split_components(X_val, cfg)

    # 1-step targets
    y_tr_1 = y_train.reshape(-1, 1)
    y_va_1 = y_val.reshape(-1, 1)

    # Multi-horizon sliding targets (preserve sku order within each split)
    train_order = train_df.sort_values(["id_var", "ds"], kind="mergesort").index.to_numpy()
    val_order = val_df.sort_values(["id_var", "ds"], kind="mergesort").index.to_numpy()
    Xtr_s = X_train[train_order]
    ytr_s = y_train[train_order]
    sktr_s = train_df.loc[train_order, "id_var"].astype(str).to_numpy()
    Xva_s = X_val[val_order]
    yva_s = y_val[val_order]
    skva_s = val_df.loc[val_order, "id_var"].astype(str).to_numpy()

    Xtr_mh, ytr_mh, sktr_mh = build_mh_xy(Xtr_s, ytr_s, sktr_s, args.horizon)
    Xva_mh, yva_mh, skva_mh = build_mh_xy(Xva_s, yva_s, skva_s, args.horizon)
    sku_tr_mh = np.array([sku_map[s] for s in sktr_mh], np.int32).reshape(-1, 1)
    sku_va_mh = np.array([sku_map[s] for s in skva_mh], np.int32).reshape(-1, 1)
    tr_mh = split_components(Xtr_mh, cfg)
    va_mh = split_components(Xva_mh, cfg)
    print(
        f"MH windows train/val={len(ytr_mh)}/{len(yva_mh)} "
        f"H={args.horizon} zero_rate={zero_rate:.3f}"
    )

    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "seed": args.seed,
            "horizon": args.horizon,
            "epochs": args.epochs,
            "zero_rate": zero_rate,
            "mase_scale_season7": mase_scale,
            "volume_stats": volume_stats,
            "variants": ["ds_h1_recursive", "ds_mh_direct", "ds_mh_tuned_direct"],
        },
        "models": {},
    }

    # --- Train variants ---
    ds_h1, t_h1 = train_ds(
        horizon=1,
        tr=tr,
        va=va,
        y_tr=y_tr_1,
        y_va=y_va_1,
        sku_tr=sku_train,
        sku_va=sku_val,
        n_skus=n_skus,
        zero_rate=zero_rate,
        avg_nz=avg_nz,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label="ds_h1",
    )
    results["models"]["ds_h1_recursive"] = {"train_seconds": t_h1}

    ds_mh, t_mh = train_ds(
        horizon=args.horizon,
        tr=tr_mh,
        va=va_mh,
        y_tr=ytr_mh,
        y_va=yva_mh,
        sku_tr=sku_tr_mh,
        sku_va=sku_va_mh,
        n_skus=n_skus,
        zero_rate=zero_rate,
        avg_nz=avg_nz,
        epochs=args.epochs,
        batch_size=args.batch_size,
        horizon_decay=0.95,
        label="ds_mh",
    )
    results["models"]["ds_mh_direct"] = {"train_seconds": t_mh}

    ds_mh_t, t_mht = train_ds(
        horizon=args.horizon,
        tr=tr_mh,
        va=va_mh,
        y_tr=ytr_mh,
        y_va=yva_mh,
        sku_tr=sku_tr_mh,
        sku_va=sku_va_mh,
        n_skus=n_skus,
        zero_rate=zero_rate,
        avg_nz=avg_nz,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha_bce=0.35,
        w_mag=1.25,
        horizon_decay=0.95,
        label="ds_mh_tuned",
    )
    results["models"]["ds_mh_tuned_direct"] = {"train_seconds": t_mht}

    # --- Origins / timelines ---
    print("\nBuilding timelines...")
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    panel = pd.concat([train_df, val_df, test_df], ignore_index=True)
    hol = pd.concat([h_tr, h_va, h_te], ignore_index=True)
    timelines = build_sku_timelines(panel, hol, cfg.holiday_names)
    epoch = pd.Timestamp("1970-01-01")
    raw_tr = (pd.to_datetime(train_df["ds"]) - epoch).dt.days.to_numpy(np.float64)
    tmin_raw, tmax_raw = float(raw_tr.min()), float(raw_tr.max())
    span_raw = max(tmax_raw - tmin_raw, 1.0)

    origin_mask = {}
    for sku, g in panel.groupby(panel["id_var"].astype(str), sort=False):
        g = g.sort_values("ds", kind="mergesort")
        origin_mask[str(sku)] = g["split"].to_numpy() == "test"
    origins = collect_origins(
        timelines,
        sku_map,
        horizon=args.horizon,
        origin_split_mask=origin_mask,
        max_origins_per_sku=args.max_origins_per_sku,
        seed=args.seed,
    )
    print(f"origins={len(origins)}")
    results["config"]["n_origins"] = len(origins)

    # Val origins for bias calibration of MH models
    origin_mask_val = {}
    for sku, g in panel.groupby(panel["id_var"].astype(str), sort=False):
        g = g.sort_values("ds", kind="mergesort")
        origin_mask_val[str(sku)] = g["split"].to_numpy() == "val"
    origins_val = collect_origins(
        timelines,
        sku_map,
        horizon=args.horizon,
        origin_split_mask=origin_mask_val,
        max_origins_per_sku=min(4, args.max_origins_per_sku),
        seed=args.seed,
    )

    def pred_h1(X, sku):
        parts = split_components(X, cfg)
        out = ds_h1.predict([*parts, sku], batch_size=4096, verbose=0)
        return (
            np.asarray(out["final_forecast"]).reshape(-1),
            np.asarray(out["non_zero_probability"]).reshape(-1),
        )

    def make_mh_pred(model, scale=1.0):
        def _pred(X, sku):
            parts = split_components(X, cfg)
            out = model.predict([*parts, sku], batch_size=4096, verbose=0)
            yh = np.asarray(out["final_forecast"], np.float32)
            p = np.asarray(out["non_zero_probability"], np.float32)
            if yh.ndim == 1:
                yh = yh.reshape(-1, 1)
                p = p.reshape(-1, 1)
            return np.maximum(yh * scale, 0.0), p

        return _pred

    # Calibrate MH scales on val
    print("Calibrating MH bias on val origins...")
    scales = {}
    for name, model in (("ds_mh_direct", ds_mh), ("ds_mh_tuned_direct", ds_mh_t)):
        roll_v = rollout_direct_tabular(
            timelines,
            origins_val,
            sku_map,
            make_mh_pred(model, 1.0),
            cfg.lag_periods,
            tmin_raw,
            span_raw,
            args.horizon,
        )
        s = calibrate_bias(roll_v["y_true"], roll_v["yhat"])
        scales[name] = s
        print(f"  {name} bias_scale={s:.3f}")
        results["models"][name]["bias_scale"] = s

    # Evaluate
    print("\n=== ds_h1 recursive rollout ===")
    t0 = time.time()
    roll = rollout_tabular(
        timelines,
        origins,
        sku_map,
        pred_h1,
        cfg.lag_periods,
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
    results["models"]["ds_h1_recursive"].update(
        {"eval_seconds": time.time() - t0, "protocol": "recursive", **metrics}
    )

    for name, model in (("ds_mh_direct", ds_mh), ("ds_mh_tuned_direct", ds_mh_t)):
        print(f"\n=== {name} direct rollout ===")
        t0 = time.time()
        roll = rollout_direct_tabular(
            timelines,
            origins,
            sku_map,
            make_mh_pred(model, scales[name]),
            cfg.lag_periods,
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
        results["models"][name].update(
            {
                "eval_seconds": time.time() - t0,
                "protocol": "direct",
                **metrics,
            }
        )

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
    results["mase_scale_season7"] = mase_scale

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("DS IMPROVEMENT BAKE-OFF (primary: iwmae_rounded)")
    print("=" * 70)
    for key in ("1", "7", "14", "mean"):
        print(f"\n[h={key}]")
        for row in comparison[key]:
            print(
                f"  {row['model']:28s} iwmae={row['iwmae_rounded']:.3f} "
                f"mae={row['mae_rounded']:.3f} nz={row['mae_nonzero']:.3f} "
                f"occ_f1={row['occ_f1']:.3f} under={row['underforecast_rate_nonzero']:.3f} "
                f"bias={row['bias']:.3f}"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
