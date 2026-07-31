#!/usr/bin/env python3
"""
Hybrid DeepSequence multi-horizon (recursive) on locked daily panel.

Trains hybrid d=64 / 1-block / decoupled (best H=1 reclaim config), then
recursive rollout H=14 so lead-time economics can include hybrid at
h=1 (1 day), h=7 (1 week), h=14 (2 weeks).

Usage::

    DEEPSEQUENCE_DATA_DIR=... TF_USE_LEGACY_KERAS=1 \\
      python examples/eval_ds_hybrid_mh.py \\
        --sku_list ab_runs/recompare/sku_list_daily_data42.json \\
        --data_seed 42 --train_seed 42 \\
        --out_json ab_runs/reclaim/daily_mh14_hybrid_d64.json
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.hybrid_temporal import (
    build_hierarchical_model_hybrid,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from eval_helpers import (
    add_panel_seed_args,
    filter_aligned,
    resolve_eval_seeds,
    select_eval_skus,
    split_components,
    train_mase_scale,
    train_volume_terciles,
)
from eval_ds_hybrid_temporal import build_hybrid_aligned_windows
from multihorizon_rollout import (
    build_sku_timelines,
    collect_origins,
    horizon_metrics,
    rollout_hybrid,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--horizon", type=int, default=14)
    add_panel_seed_args(p)
    p.add_argument("--max_origins_per_sku", type=int, default=8)
    p.add_argument("--temporal_d_model", type=int, default=64)
    p.add_argument("--temporal_n_heads", type=int, default=4)
    p.add_argument("--temporal_n_blocks", type=int, default=1)
    p.add_argument("--decouple_gate", action="store_true", default=True)
    p.add_argument("--no_decouple_gate", action="store_false", dest="decouple_gate")
    p.add_argument(
        "--out_json",
        default=str(ROOT / "ab_runs" / "reclaim" / "daily_mh14_hybrid_d64.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_seed, train_seed = resolve_eval_seeds(args.seed, args.data_seed, args.train_seed)
    tf.keras.utils.set_random_seed(train_seed)
    data_dir_raw = args.data_dir or os.environ.get("DEEPSEQUENCE_DATA_DIR")
    if not data_dir_raw:
        raise SystemExit("Pass --data_dir or set DEEPSEQUENCE_DATA_DIR")
    data_dir = Path(data_dir_raw)

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
        f"n_skus={len(chosen)} sku_list={args.sku_list}"
    )
    train_df, h_tr = filter_aligned(train_df, h_tr, chosen)
    val_df, h_va = filter_aligned(val_df, h_va, chosen)
    test_df, h_te = filter_aligned(test_df, h_te, chosen)

    volume_map, volume_stats = train_volume_terciles(train_df)
    mase_scale = train_mase_scale(train_df, season=7)
    cats = pd.Categorical(train_df["id_var"])
    sku_map = {k: i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return df["id_var"].map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)

    y_train = train_df["Quantity"].to_numpy(np.float32)
    zero_rate = float((y_train == 0).mean())
    sku_train, sku_val = enc(train_df), enc(val_df)

    cfg = load_feature_config()
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, _ = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    t_idx = cfg.trend_indices[0]
    tmin_feat, tmax_feat = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span_feat = max(tmax_feat - tmin_feat, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin_feat) / span_feat

    print("\nBuilding hybrid aligned windows for training...")
    Xseq, Xtab, yseq, sku_seq_raw, split_seq, n_channels = build_hybrid_aligned_windows(
        train_df, val_df, test_df, X_train, X_val, X_test, args.lookback
    )
    sku_seq = np.array([sku_map[s] for s in sku_seq_raw], dtype=np.int32).reshape(-1, 1)
    m_tr = split_seq == "train"
    m_va = split_seq == "val"
    tr_w = split_components(Xtab[m_tr], cfg)
    va_w = split_components(Xtab[m_va], cfg)

    print(
        f"\n=== Hybrid train d={args.temporal_d_model} blocks={args.temporal_n_blocks} "
        f"decouple={args.decouple_gate} ==="
    )
    hybrid = build_hierarchical_model_hybrid(
        n_temporal_features=len(cfg.trend_indices),
        n_fourier_features=len(cfg.seasonal_indices),
        n_holiday_features=len(cfg.holiday_indices),
        n_lag_features=len(cfg.regressor_indices),
        n_skus=n_skus,
        n_sequence_channels=n_channels,
        lookback=args.lookback,
        temporal_d_model=args.temporal_d_model,
        temporal_n_heads=args.temporal_n_heads,
        temporal_n_blocks=args.temporal_n_blocks,
        decouple_gate=args.decouple_gate,
        hidden_dim=48,
        sku_embedding_dim=4,
        dropout_rate=0.23,
        use_cross_layers=True,
        use_intermittent=True,
        n_changepoints=15,
    )
    dummy_seq = np.zeros((1, args.lookback, n_channels), np.float32)
    _ = hybrid(
        [
            *(np.zeros((1, x.shape[1]), np.float32) for x in tr_w),
            np.zeros((1, 1), np.int32),
            dummy_seq,
        ],
        training=False,
    )
    pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))
    model = AdaptiveWeightedModel(
        base_model=hybrid,
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
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
    ytr = yseq[m_tr]
    yva = yseq[m_va]
    ytr_d = {"final_forecast": ytr.reshape(-1, 1), "base_forecast": ytr.reshape(-1, 1)}
    yva_d = {"final_forecast": yva.reshape(-1, 1), "base_forecast": yva.reshape(-1, 1)}
    t0 = time.time()
    model.fit(
        [*tr_w, sku_seq[m_tr], Xseq[m_tr]],
        ytr_d,
        validation_data=([*va_w, sku_seq[m_va], Xseq[m_va]], yva_d),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            )
        ],
        verbose=2,
    )
    train_s = time.time() - t0

    # Rollout timelines use raw epoch days (same as eval_multihorizon_compare)
    print("\nBuilding timelines for hybrid recursive rollout...")
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    panel = pd.concat([train_df, val_df, test_df], ignore_index=True)
    hol = pd.concat([h_tr, h_va, h_te], ignore_index=True)
    timelines = build_sku_timelines(panel, hol, cfg.holiday_names)

    all_t = np.concatenate([tl.time_index for tl in timelines.values()])
    tmin_raw, tmax_raw = float(all_t.min()), float(all_t.max())
    span_raw = max(tmax_raw - tmin_raw, 1.0)

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
        seed=data_seed,
    )
    origins = [(s, t) for s, t in origins if t + 1 >= args.lookback]
    print(f"origins={len(origins)} lookback={args.lookback} horizon={args.horizon}")

    lag_periods = cfg.lag_periods

    def predict_fn(X, sku, windows):
        parts = split_components(X, cfg)
        pred = model.predict([*parts, sku, windows], batch_size=1024, verbose=0)
        yhat = np.asarray(pred["final_forecast"]).reshape(-1)
        p = np.asarray(pred["non_zero_probability"]).reshape(-1)
        return yhat, p

    print("\n=== Hybrid multi-horizon rollout ===")
    t1 = time.time()
    roll = rollout_hybrid(
        timelines,
        origins,
        sku_map,
        predict_fn,
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
        report_horizons=(1, 7, 14),
        mase_scale=mase_scale,
    )
    rollout_s = time.time() - t1

    results = {
        "config": {
            "protocol": "recursive_rollout_after_origin_hybrid",
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "sku_list": args.sku_list,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "report_horizons": [1, 7, 14],
            "max_origins_per_sku": args.max_origins_per_sku,
            "zero_rate": zero_rate,
            "temporal_d_model": args.temporal_d_model,
            "temporal_n_heads": args.temporal_n_heads,
            "temporal_n_blocks": args.temporal_n_blocks,
            "decouple_gate": args.decouple_gate,
            "n_channels": n_channels,
            "volume_stats": volume_stats,
            "note": (
                "Hybrid d64/b1/decouple recursive MH for lead-time economics "
                "(1 day / 1 week / 2 weeks)."
            ),
        },
        "models": {
            "hybrid_d64_b1_decouple": {
                "train_seconds": train_s,
                "rollout_seconds": rollout_s,
                **metrics,
            }
        },
        "mase_scale_season7": mase_scale,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    for h in ("1", "7", "14"):
        block = metrics["by_horizon"][h]["overall"]
        print(
            f"  h={h}: iwmae={block.get('iwmae_rounded'):.3f} "
            f"mae={block.get('mae_all'):.3f} "
            f"under_nz={block.get('underforecast_rate_nonzero')}"
        )
    mean = metrics["mean_1_to_H"]["overall"]
    print(f"  mean: iwmae={mean.get('iwmae_rounded'):.3f}")


if __name__ == "__main__":
    main()
