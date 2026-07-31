#!/usr/bin/env python3
"""
Train-time gate calibration bake-off (locked daily H=1).

Compares:
  - deepsequence              paper defaults (no gate reclaim)
  - deepsequence_train_cal    prior + raw regressors + learnable logit scale
                              + softplus p-scale (init 0.85) + light rate-match
  - deepsequence_posthoc      plain DS + val IWMAE scale/threshold (fallback)

Paper path stays default-off in the package; this script opts in explicitly.
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
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from eval_helpers import (
    add_panel_seed_args,
    apply_iwmae_gate,
    calibrate_iwmae_gate,
    filter_aligned,
    kpi_block,
    resolve_eval_seeds,
    select_eval_skus,
    split_components,
    strata_report,
    train_mase_scale,
    train_volume_terciles,
)

ALL_MODELS = (
    "deepsequence",
    "deepsequence_train_cal",
    "deepsequence_posthoc",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=1024)
    add_panel_seed_args(p)
    p.add_argument("--models", default=",".join(ALL_MODELS))
    p.add_argument("--gate_prob_scale_init", type=float, default=0.85)
    p.add_argument("--gate_rate_match_weight", type=float, default=0.01)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "ab_runs" / "reclaim" / "daily_h1_train_gate_cal.json"),
    )
    return p.parse_args()


def _build_and_train(
    tr,
    va,
    te,
    y_train,
    y_val,
    sku_train,
    sku_val,
    sku_test,
    zero_rate,
    n_skus,
    cfg,
    args,
    *,
    train_calibrate: bool,
):
    nz_target = max(1e-6, 1.0 - float(zero_rate))
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
        gate_use_raw_regressors=train_calibrate,
        intermittent_prior_zero_rate=zero_rate if train_calibrate else None,
        intermittent_learnable_logit_scale=train_calibrate,
        intermittent_logit_scale_init=1.0,
        gate_prob_scale=train_calibrate,
        gate_prob_scale_init=float(args.gate_prob_scale_init),
        gate_prob_scale_trainable=True,
        gate_rate_match_weight=(
            float(args.gate_rate_match_weight) if train_calibrate else 0.0
        ),
        gate_rate_match_target=nz_target if train_calibrate else None,
    )
    _ = base(
        [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
        training=False,
    )
    pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))
    model = AdaptiveWeightedModel(
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
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
    ytr = {"final_forecast": y_train.reshape(-1, 1), "base_forecast": y_train.reshape(-1, 1)}
    yva = {"final_forecast": y_val.reshape(-1, 1), "base_forecast": y_val.reshape(-1, 1)}
    t0 = time.time()
    model.fit(
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

    def _predict(parts, sku):
        pred = model.predict([*parts, sku], batch_size=4096, verbose=0)
        return (
            np.asarray(pred["final_forecast"]).reshape(-1),
            np.asarray(pred["non_zero_probability"]).reshape(-1),
        )

    yhat_va, p_va = _predict(va, sku_val)
    yhat_te, p_te = _predict(te, sku_test)
    return model, train_s, yhat_va, p_va, yhat_te, p_te


def main():
    args = parse_args()
    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    unknown = selected - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"Unknown models: {sorted(unknown)}")

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
        f"Panel lock: data_seed={data_seed} train_seed={train_seed} n_skus={len(chosen)}"
        + (f" sku_list={args.sku_list}" if args.sku_list else "")
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

    cfg = load_feature_config()
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, _ = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
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

    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "sku_list": args.sku_list,
            "epochs": args.epochs,
            "zero_rate": zero_rate,
            "gate_prob_scale_init": args.gate_prob_scale_init,
            "gate_rate_match_weight": args.gate_rate_match_weight,
            "volume_stats": volume_stats,
            "models_run": sorted(selected),
            "note": (
                "Train-time gate calibration: prior zero_rate bias, raw regressor "
                "gate path, learnable logit sharpening, softplus p-scale, rate-match aux."
            ),
        },
        "models": {},
    }

    plain_cache = None
    if "deepsequence" in selected or "deepsequence_posthoc" in selected:
        print("\n=== DeepSequence (paper defaults) ===")
        _, train_s, yhat_va, p_va, yhat_te, p_te = _build_and_train(
            tr, va, te, y_train, y_val, sku_train, sku_val, sku_test,
            zero_rate, n_skus, cfg, args, train_calibrate=False,
        )
        plain_cache = {
            "train_seconds": train_s,
            "yhat_va": yhat_va,
            "p_va": p_va,
            "yhat_te": yhat_te,
            "p_te": p_te,
        }
        if "deepsequence" in selected:
            results["models"]["deepsequence"] = {
                "train_seconds": train_s,
                "overall": kpi_block(y_test, yhat_te, p_te, mase_scale=mase_scale),
                "strata": strata_report(
                    y_test, yhat_te, p_te, sku_test_raw, volume_map, mase_scale=mase_scale
                ),
            }

    if "deepsequence_posthoc" in selected:
        if plain_cache is None:
            raise SystemExit("posthoc requires plain deepsequence trunk")
        print("\n=== DeepSequence + post-hoc IWMAE gate ===")
        iwmae_calib = calibrate_iwmae_gate(
            y_val, plain_cache["yhat_va"], plain_cache["p_va"]
        )
        yhat_te = apply_iwmae_gate(
            plain_cache["yhat_te"],
            plain_cache["p_te"],
            scale=iwmae_calib["scale"],
            threshold=iwmae_calib["threshold"],
        )
        results["models"]["deepsequence_posthoc"] = {
            "train_seconds": plain_cache["train_seconds"],
            "iwmae_calibration": iwmae_calib,
            "overall": kpi_block(
                y_test, yhat_te, plain_cache["p_te"], mase_scale=mase_scale
            ),
            "strata": strata_report(
                y_test,
                yhat_te,
                plain_cache["p_te"],
                sku_test_raw,
                volume_map,
                mase_scale=mase_scale,
            ),
        }

    if "deepsequence_train_cal" in selected:
        print("\n=== DeepSequence (train-time gate calibration) ===")
        model, train_s, _, _, yhat_te, p_te = _build_and_train(
            tr, va, te, y_train, y_val, sku_train, sku_val, sku_test,
            zero_rate, n_skus, cfg, args, train_calibrate=True,
        )
        scale_layer = None
        try:
            scale_layer = model.base_model.get_layer("non_zero_probability")
            learned_scale = float(tf.nn.softplus(scale_layer._scale_raw).numpy())
        except Exception:
            learned_scale = None
        results["models"]["deepsequence_train_cal"] = {
            "train_seconds": train_s,
            "learned_gate_prob_scale": learned_scale,
            "overall": kpi_block(y_test, yhat_te, p_te, mase_scale=mase_scale),
            "strata": strata_report(
                y_test, yhat_te, p_te, sku_test_raw, volume_map, mase_scale=mase_scale
            ),
        }

    comparison = []
    for name, payload in results["models"].items():
        block = payload["overall"]
        comparison.append(
            {
                "model": name,
                "iwmae_rounded": block.get("iwmae_rounded"),
                "mae_rounded": block.get("mae_all_rounded"),
                "mean_p": block.get("mean_p"),
                "bias": block.get("bias"),
                "occ_f1": block.get("occ_f1"),
            }
        )
    comparison = sorted(
        comparison,
        key=lambda r: (
            r["iwmae_rounded"] is None,
            r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
        ),
    )
    results["comparison"] = comparison

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("TRAIN-TIME GATE CALIBRATION H=1")
    print("=" * 70)
    for row in comparison:
        print(
            f"  {row['model']:28s} iwmae={row['iwmae_rounded']:.3f} "
            f"mean_p={row.get('mean_p')} bias={row['bias']:.3f} "
            f"occ_f1={row.get('occ_f1')}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
