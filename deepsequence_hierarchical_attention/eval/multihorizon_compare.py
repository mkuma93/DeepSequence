#!/usr/bin/env python3
"""
Multi-horizon bake-off (v1.6, recursive rollout of 1-step models).

Trains the same 1-step models as eval_same_features_compare.py, then
recursively forecasts H days ahead with known-future calendar/holidays
and predicted demand fed back into lags/intermittent state.

Default report horizons are h=1,7,14 (and 21,28 when ``--horizon`` is large
enough). DeepSequence uses the preferred builder stack by default
(softsign + mono trend/holiday/regressor + context mixer; FiLM off) unless
CLI overrides are passed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deepsequence_hierarchical_attention.data.feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.losses import three_term_loss_config
from deepsequence_hierarchical_attention.training.adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from deepsequence_hierarchical_attention.eval.helpers import (
    add_panel_seed_args,
    build_deepar,
    build_tft,
    build_transformer,
    filter_aligned,
    fit_bce_sample_weight_dict,
    resolve_eval_seeds,
    resolve_sku_zero_rates,
    select_eval_skus,
    split_components,
    train_mase_scale,
    train_volume_terciles,
)
from deepsequence_hierarchical_attention.eval.multihorizon_rollout import (
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
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument(
        "--report_horizons",
        default=None,
        help=(
            "Comma-separated horizons to report (1-indexed). "
            "Default: 1,7,14 plus 21/28 when --horizon allows."
        ),
    )
    add_panel_seed_args(p)
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
    # DeepSequence builder overrides (None = preferred builder default:
    # softsign + mono + mixer, FiLM off).
    p.add_argument("--output_activation", default=None)
    p.add_argument("--trend_monotonic", type=int, default=None)
    p.add_argument("--holiday_monotonic", type=int, default=None)
    p.add_argument("--regressor_monotonic", type=int, default=None)
    p.add_argument("--context_aware_component_mixer", type=int, default=None)
    p.add_argument("--context_film_seasonal_holiday", type=int, default=None)
    p.add_argument(
        "--level1_selection_attention",
        type=int,
        default=None,
        help="Intra-expert selection attn (1/0). None = builder default (True).",
    )
    p.add_argument(
        "--use_intermittent",
        type=int,
        default=None,
        help="Occurrence gate (1/0). None = builder default (True).",
    )
    p.add_argument(
        "--use_cross_layers",
        type=int,
        default=None,
        help="DCN cross on component outputs (1/0). None = builder default (False).",
    )
    return p.parse_args()


def _default_report_horizons(horizon: int) -> list[int]:
    """Natural report points: weekly through H, plus free h=1."""
    candidates = [1, 7, 14, 21, 28]
    return [h for h in candidates if h <= int(horizon)]


def _parse_report_horizons(raw: str | None, horizon: int) -> list[int]:
    if raw is None or not str(raw).strip():
        return _default_report_horizons(horizon)
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        h = int(part)
        if h < 1 or h > int(horizon):
            raise SystemExit(
                f"report horizon {h} outside 1..{horizon}; raise --horizon or trim list"
            )
        out.append(h)
    if not out:
        raise SystemExit("--report_horizons produced an empty list")
    return out


def _ds_builder_kwargs(args) -> dict:
    """Resolve DeepSequence stack kwargs from CLI (explicit overrides only).

    Unset CLI flags keep the preferred builder defaults: softsign experts,
    monotone trend/holiday/regressor, context-aware mixer on, calendar FiLM off,
    cross layers off, Level-1 selection attention on, gate on.
    """
    sig = inspect.signature(build_hierarchical_model_lightweight)
    defaults = {k: sig.parameters[k].default for k in (
        "output_activation",
        "trend_monotonic",
        "holiday_monotonic",
        "regressor_monotonic",
        "context_aware_component_mixer",
        "context_film_seasonal_holiday",
        "level1_selection_attention",
        "use_intermittent",
        "use_cross_layers",
    )}
    out = dict(defaults)
    if args.output_activation is not None:
        out["output_activation"] = args.output_activation
    for flag in (
        "trend_monotonic",
        "holiday_monotonic",
        "regressor_monotonic",
        "context_aware_component_mixer",
        "context_film_seasonal_holiday",
        "level1_selection_attention",
        "use_intermittent",
        "use_cross_layers",
    ):
        val = getattr(args, flag)
        if val is not None:
            out[flag] = bool(val)
    return out


def train_seq_three_term(
    model, Xtr, ytr, skutr, Xva, yva, skuva, zero_rate, args, label, sku_zero_rates=None
):
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
    sw_tr = sw_va = None
    if sku_zero_rates is not None:
        sw_tr = fit_bce_sample_weight_dict(
            ytr, skutr, sku_zero_rates, panel_zero_rate=zero_rate
        )
        sw_va = fit_bce_sample_weight_dict(
            yva, skuva, sku_zero_rates, panel_zero_rate=zero_rate
        )
    print(f"\n=== train {label} ===")
    t0 = time.time()
    fit_kw = dict(
        x=[Xtr, skutr],
        y=ytr_d,
        validation_data=(
            ([Xva, skuva], yva_d, sw_va) if sw_va is not None else ([Xva, skuva], yva_d)
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
            ),
        ],
        verbose=2,
    )
    if sw_tr is not None:
        fit_kw["sample_weight"] = sw_tr
    model.fit(**fit_kw)
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
    report_horizons = _parse_report_horizons(args.report_horizons, args.horizon)

    data_seed, train_seed = resolve_eval_seeds(
        args.seed, args.data_seed, args.train_seed
    )
    tf.keras.utils.set_random_seed(train_seed)
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

    chosen_list = select_eval_skus(
        train_df["id_var"].unique(),
        max_skus=args.max_skus,
        data_seed=data_seed,
        sku_list_path=args.sku_list,
        save_sku_list_path=args.save_sku_list,
    )
    chosen = set(chosen_list)
    print(
        f"Panel lock: data_seed={data_seed} train_seed={train_seed} "
        f"n_skus={len(chosen)}"
        + (f" sku_list={args.sku_list}" if args.sku_list else "")
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
            "data_seed": data_seed,
            "train_seed": train_seed,
            "sku_list": args.sku_list,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "report_horizons": list(report_horizons),
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
            "ds_stack": _ds_builder_kwargs(args),
        },
        "models": {},
    }
    ds_stack = results["config"]["ds_stack"]
    print(f"DS stack: {ds_stack}")

    # ------------------------------------------------------------------
    # Train DeepSequence
    # ------------------------------------------------------------------
    ds_model = None
    if "deepsequence" in selected:
        print("\n=== DeepSequence train ===")
        _, sku_rates = resolve_sku_zero_rates(y_train, sku_train, n_skus=n_skus)
        print(
            f"  per-SKU BCE imbalance ON "
            f"(panel_zr={zero_rate:.3f}, n_skus={n_skus})"
        )
        base = build_hierarchical_model_lightweight(
            n_temporal_features=len(cfg.trend_indices),
            n_fourier_features=len(cfg.seasonal_indices),
            n_holiday_features=len(cfg.holiday_indices),
            n_lag_features=len(cfg.regressor_indices),
            n_skus=n_skus,
            hidden_dim=48,
            sku_embedding_dim=4,
            dropout_rate=0.23,
            n_changepoints=15,
            **ds_stack,
        )
        _ = base(
            [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
            training=False,
        )
        # Confirm Level-1 mono ⊕ selection attention and cross setting.
        from deepsequence_hierarchical_attention.components_lightweight import (
            HolidayComponentLightweight,
            RegressorComponentLightweight,
        )
        found_h = found_r = False
        h_sel = r_sel = None
        for obj in base._flatten_layers():
            if isinstance(obj, HolidayComponentLightweight):
                found_h = True
                h_sel = bool(getattr(obj, "level1_selection_attention", True))
            if isinstance(obj, RegressorComponentLightweight):
                found_r = True
                r_sel = bool(getattr(obj, "level1_selection_attention", True))
        layer_names = [l.name for l in base._flatten_layers()]
        has_cross = any("cross_layer" in n for n in layer_names)
        print(
            f"  Level-1 check: holiday={found_h} regressor={found_r} "
            f"level1_sel(h/r)={h_sel}/{r_sel} "
            f"use_cross_layers={ds_stack.get('use_cross_layers')} "
            f"cross_present={has_cross}"
        )
        if not (found_h and found_r):
            raise RuntimeError(
                "holiday/regressor experts missing from DeepSequence graph"
            )
        want_l1 = bool(ds_stack.get("level1_selection_attention", True))
        if want_l1 and h_sel is False:
            raise RuntimeError("expected Level-1 holiday selection attention")
        if bool(ds_stack.get("use_cross_layers")) != has_cross:
            raise RuntimeError(
                f"cross mismatch: flag={ds_stack.get('use_cross_layers')} "
                f"present={has_cross}"
            )
        pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))
        ds_model = AdaptiveWeightedModel(
            base_model=base,
            bce_loss_fn=WeightedBCELoss(weight_nonzero=pos_weight, weight_zero=1.0),
            mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            zero_rate=zero_rate,
            avg_nonzero_demand=float(y_train[y_train > 0].mean()),
            pos_weight=pos_weight,
            sku_zero_rates=sku_rates,
            loss_recipe="three_term",
            alpha_bce=0.2,
            w_gated=1.0,
            w_mag=1.0,
            use_fixed_weights=True,
        )
        ds_model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
        ytr = {"final_forecast": y_train.reshape(-1, 1), "base_forecast": y_train.reshape(-1, 1)}
        yva = {"final_forecast": y_val.reshape(-1, 1), "base_forecast": y_val.reshape(-1, 1)}
        try:
            n_params = int(base.count_params())
        except Exception:
            n_params = None
        t0 = time.time()
        ds_model.fit(
            [*tr, sku_train],
            ytr,
            validation_data=([*va, sku_val], yva),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
                )
            ],
            verbose=2,
        )
        results["models"].setdefault("deepsequence", {})["train_seconds"] = time.time() - t0
        results["models"]["deepsequence"]["n_params"] = n_params

    # ------------------------------------------------------------------
    # Train LightGBM
    # ------------------------------------------------------------------
    lgb_model = None
    if "lightgbm" in selected:
        print("\n=== LightGBM train ===")
        import lightgbm as lgb

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
        _, seq_sku_rates = resolve_sku_zero_rates(y_train, sku_train, n_skus=n_skus)
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
                sku_zero_rates=seq_sku_rates,
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
            report_horizons=report_horizons,
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
            report_horizons=report_horizons,
            mase_scale=mase_scale,
        )
        results["models"]["lightgbm"].update(
            {"rollout_seconds": time.time() - t0, **metrics}
        )

    for name, model in seq_models.items():
        print(f"\n=== {name} multi-horizon rollout ===")
        try:
            results["models"][name]["n_params"] = int(model.count_params())
        except Exception:
            results["models"][name]["n_params"] = None
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
            report_horizons=report_horizons,
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
    comparison_cum = {}
    compare_keys = [str(h) for h in report_horizons] + ["mean"]
    for key in compare_keys:
        comparison[key] = []
        comparison_cum[key] = []
        for model, payload in results["models"].items():
            if key == "mean":
                block = payload.get("mean_1_to_H", {}).get("overall", {})
                cum_block = {}
            else:
                block = payload.get("by_horizon", {}).get(key, {}).get("overall", {})
                cum_block = (
                    payload.get("by_horizon_cum", {}).get(key, {}).get("overall", {})
                )
            if not block:
                continue
            comparison[key].append(
                {
                    "model": model,
                    "mae_rounded": block.get("mae_all_rounded"),
                    "mae_nonzero": block.get("mae_nonzero"),
                    "iwmae_rounded": block.get("iwmae_rounded"),
                    "cummae": cum_block.get("cummae"),
                    "cummae_rounded": cum_block.get("cummae_rounded"),
                    "cum_iwmae_rounded": cum_block.get("cum_iwmae_rounded"),
                    "mase_rounded": block.get("mase_rounded"),
                    "occ_f1": block.get("occ_f1"),
                    "underforecast_rate_nonzero": block.get(
                        "underforecast_rate_nonzero"
                    ),
                    "bias": block.get("bias"),
                    "bias_nonzero": block.get("bias_nonzero"),
                    "aucroc": block.get("aucroc"),
                    "sales_revenue_loss_units": block.get(
                        "sales_revenue_loss_units"
                    ),
                    "inventory_holding_cost_zero": block.get(
                        "inventory_holding_cost_zero"
                    ),
                    "combined_ops_cost_h0p1": block.get("combined_ops_cost_h0p1"),
                }
            )
            if cum_block:
                comparison_cum[key].append(
                    {
                        "model": model,
                        "cummae": cum_block.get("cummae"),
                        "cummae_rounded": cum_block.get("cummae_rounded"),
                        "cum_iwmae": cum_block.get("cum_iwmae"),
                        "cum_iwmae_rounded": cum_block.get("cum_iwmae_rounded"),
                        "iwmae_rounded": block.get("iwmae_rounded"),
                    }
                )
        comparison[key] = sorted(
            comparison[key],
            key=lambda r: (
                r["iwmae_rounded"] is None,
                r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
            ),
        )
        if comparison_cum[key]:
            comparison_cum[key] = sorted(
                comparison_cum[key],
                key=lambda r: (
                    r["cummae_rounded"] is None,
                    r["cummae_rounded"] if r["cummae_rounded"] is not None else 1e9,
                ),
            )
    results["comparison"] = comparison
    results["comparison_cum"] = {
        k: v for k, v in comparison_cum.items() if v
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("MULTI-HORIZON COMPARISON (recursive; primary sort: iwmae_rounded)")
    print("=" * 70)
    for key in compare_keys:
        print(f"\n[h={key}]")
        for row in comparison[key]:
            cum_s = ""
            if row.get("cummae_rounded") is not None:
                cum_s = f" cummae={row['cummae_rounded']:.3f}"
            print(
                f"  {row['model']:28s} iwmae={row['iwmae_rounded']:.3f} "
                f"mae={row['mae_rounded']:.3f} nz={row['mae_nonzero']:.3f} "
                f"mase={row.get('mase_rounded')} occ_f1={row.get('occ_f1')} "
                f"under={row.get('underforecast_rate_nonzero')} "
                f"bias={row['bias']:.3f}{cum_s}"
            )
            rev = row.get("sales_revenue_loss_units")
            hold = row.get("inventory_holding_cost_zero")
            ops = row.get("combined_ops_cost_h0p1")
            if rev is not None and hold is not None and ops is not None:
                print(
                    f"  {'':28s} rev_loss={rev:.3f} hold0={hold:.3f} "
                    f"ops_h0.1={ops:.3f}"
                )
    if results["comparison_cum"]:
        print("\n" + "=" * 70)
        print("CUMULATIVE MAE (planning sum error; sort: cummae_rounded)")
        print("=" * 70)
        for key in [str(h) for h in report_horizons]:
            rows = results["comparison_cum"].get(key) or []
            if not rows:
                continue
            print(f"\n[CumMAE h={key}]")
            for row in rows:
                print(
                    f"  {row['model']:28s} cummae={row['cummae_rounded']:.3f} "
                    f"cum_iwmae={row.get('cum_iwmae_rounded')} "
                    f"(pointwise iwmae={row.get('iwmae_rounded')})"
                )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
