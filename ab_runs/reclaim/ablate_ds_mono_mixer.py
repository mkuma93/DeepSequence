#!/usr/bin/env python3
"""Focused DS-only H=1 ablations on locked daily panel (no commit).

Preferred Level-1 stack defaults: softsign + mono + mixer on + gate on +
cross off + Level-1 selection attention on.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from eval_helpers import (
    filter_aligned,
    kpi_block,
    resolve_eval_seeds,
    select_eval_skus,
    split_components,
    strata_report,
    train_mase_scale,
    train_volume_terciles,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.environ.get("DEEPSEQUENCE_DATA_DIR"))
    p.add_argument("--sku_list", default="ab_runs/recompare/sku_list_daily_data42.json")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_activation", default="softsign")
    p.add_argument("--trend_monotonic", type=int, default=1)
    p.add_argument("--holiday_monotonic", type=int, default=1)
    p.add_argument("--regressor_monotonic", type=int, default=1)
    p.add_argument("--context_aware_component_mixer", type=int, default=1)
    p.add_argument(
        "--level1_selection_attention",
        type=int,
        default=1,
        help="1=learned intra-expert selection; 0=uniform 1/n (novelty ablation).",
    )
    p.add_argument(
        "--use_intermittent",
        type=int,
        default=1,
        help="1=occurrence gate on; 0=magnitude-only head (novelty ablation).",
    )
    p.add_argument(
        "--use_cross_layers",
        type=int,
        default=0,
        help="DCN cross (default 0; known hurt — include as +cross row).",
    )
    p.add_argument("--out_json", required=True)
    p.add_argument("--label", default="ablation")
    p.add_argument(
        "--feature_config",
        default=None,
        help="Optional feature_config.yaml path (lag A/B overrides). Default: repo/package config.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.data_dir:
        raise SystemExit("Need --data_dir or DEEPSEQUENCE_DATA_DIR")
    data_seed, train_seed = resolve_eval_seeds(args.seed, None, None)
    tf.keras.utils.set_random_seed(train_seed)

    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    h_tr = pd.read_csv(data_dir / "holiday_features_train.csv")
    h_va = pd.read_csv(data_dir / "holiday_features_val.csv")
    h_te = pd.read_csv(data_dir / "holiday_features_test.csv")

    universe = sorted(train_df["id_var"].unique())
    chosen = select_eval_skus(
        universe, max_skus=800, data_seed=data_seed, sku_list_path=args.sku_list
    )
    print(
        f"Panel lock: data_seed={data_seed} train_seed={train_seed} "
        f"n_skus={len(chosen)} label={args.label}"
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
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val, sku_test = enc(train_df), enc(val_df), enc(test_df)
    sku_test_raw = test_df["id_var"].to_numpy()
    zero_rate = float((y_train == 0).mean())

    cfg = load_feature_config(args.feature_config)
    print(
        f"Feature config: {cfg.config_path} "
        f"v={cfg.config['metadata'].get('version')} "
        f"lags={cfg.lag_periods} n_feat={cfg.total_features}"
    )
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, states = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    t_idx = cfg.trend_indices[0]
    tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    tr, va, te = (
        split_components(X_train, cfg),
        split_components(X_val, cfg),
        split_components(X_test, cfg),
    )

    flags = dict(
        output_activation=str(args.output_activation),
        trend_monotonic=bool(args.trend_monotonic),
        holiday_monotonic=bool(args.holiday_monotonic),
        regressor_monotonic=bool(args.regressor_monotonic),
        context_aware_component_mixer=bool(args.context_aware_component_mixer),
        level1_selection_attention=bool(args.level1_selection_attention),
        use_intermittent=bool(args.use_intermittent),
        use_cross_layers=bool(args.use_cross_layers),
        context_film_seasonal_holiday=False,
    )
    print(f"Flags: {flags}")

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
        **flags,
    )
    _ = base(
        [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
        training=False,
    )
    pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))
    t0 = time.time()
    if flags["use_intermittent"]:
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
        hist = ds_model.fit(
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
        train_s = time.time() - t0
        pred = ds_model.predict([*te, sku_test], batch_size=4096, verbose=0)
        yhat = np.asarray(pred["final_forecast"]).reshape(-1)
        p = np.asarray(pred["non_zero_probability"]).reshape(-1)
        best_ep = int(np.argmin(hist.history["val_loss"])) + 1
        min_val = float(np.min(hist.history["val_loss"]))
    else:
        # Gate-off: train magnitude-only head with plain MAE.
        base.compile(
            optimizer=tf.keras.optimizers.Adam(0.0025),
            loss={"final_forecast": "mae"},
        )
        hist = base.fit(
            [*tr, sku_train],
            {"final_forecast": y_train.reshape(-1, 1)},
            validation_data=(
                [*va, sku_val],
                {"final_forecast": y_val.reshape(-1, 1)},
            ),
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
        pred = base.predict([*te, sku_test], batch_size=4096, verbose=0)
        if isinstance(pred, dict):
            yhat = np.asarray(pred["final_forecast"]).reshape(-1)
        else:
            yhat = np.asarray(pred).reshape(-1)
        p = None
        best_ep = int(np.argmin(hist.history["val_loss"])) + 1
        min_val = float(np.min(hist.history["val_loss"]))

    overall = kpi_block(y_test, yhat, p, mase_scale=mase_scale)
    strata = strata_report(
        y_test, yhat, p, sku_test_raw, volume_map, mase_scale=mase_scale
    )
    out = {
        "config": {
            "label": args.label,
            "sku_list": args.sku_list,
            "epochs": args.epochs,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "feature_config": args.feature_config or cfg.config_path,
            "feature_version": cfg.config["metadata"].get("version"),
            "lags": list(cfg.lag_periods),
            "n_tabular_features": int(cfg.total_features),
            "feature_names": list(cfg.feature_names),
            "flags": flags,
            "best_epoch_by_val_loss": best_ep,
            "min_val_loss": min_val,
            "volume_stats": volume_stats,
        },
        "models": {
            "deepsequence": {
                "train_seconds": train_s,
                "overall": overall,
                "strata": strata,
            }
        },
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    o = overall
    print(
        f"\nRESULT {args.label}: iwmae={o['iwmae']:.4f} mae={o['mae_all']:.4f} "
        f"nz={o['mae_nonzero']:.4f} bias={o['bias']:+.4f} "
        f"best_ep={best_ep} -> {out_path}"
    )


if __name__ == "__main__":
    main()
