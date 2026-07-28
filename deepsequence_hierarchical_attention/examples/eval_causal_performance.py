#!/usr/bin/env python3
"""
Performance eval for lightweight hierarchical model on Jubilant data.

Uses causal intermittent regressor features (feature_config v1.4).
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
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.losses import masked_mae_loss, weighted_bce_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
    )
    p.add_argument("--max_skus", type=int, default=800, help="0 = all SKUs")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--sku_embedding_dim", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=str(ROOT / "eval_results_causal.json"))
    return p.parse_args()


def select_skus(train_df, max_skus, seed):
    if not max_skus or max_skus <= 0:
        return None
    skus = train_df["id_var"].unique()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(skus, size=min(max_skus, len(skus)), replace=False)
    return set(chosen)


def filter_df(df, sku_set):
    if sku_set is None:
        return df
    return df[df["id_var"].isin(sku_set)].copy()


def filter_aligned(df, holidays, sku_set):
    if sku_set is None:
        return df.reset_index(drop=True), holidays.reset_index(drop=True)
    mask = df["id_var"].isin(sku_set).to_numpy()
    return (
        df.loc[mask].reset_index(drop=True),
        holidays.loc[mask].reset_index(drop=True),
    )


def split_components(X, cfg):
    return (
        X[:, cfg.trend_indices],
        X[:, cfg.seasonal_indices],
        X[:, cfg.holiday_indices],
        X[:, cfg.regressor_indices],
    )


def evaluate(model, inputs, y, zero_rate, threshold=None):
    if threshold is None:
        threshold = max(0.05, 1.0 - zero_rate)
    preds = model.predict(inputs, batch_size=4096, verbose=0)
    if isinstance(preds, dict):
        yhat = preds["final_forecast"].reshape(-1)
        p = preds["non_zero_probability"].reshape(-1)
    else:
        raise RuntimeError("Expected multi-output dict model")

    y = y.reshape(-1)
    y_bin = (y > 0).astype(np.float32)
    p_bin = (p >= threshold).astype(np.float32)

    mae_all = float(mean_absolute_error(y, yhat))
    nz = y > 0
    mae_nz = float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else float("nan")
    # gated: zeros forced when p < threshold
    yhat_gate = yhat * p_bin
    mae_gate = float(mean_absolute_error(y, yhat_gate))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_bin, p_bin, average="binary", zero_division=0
    )
    try:
        aucpr = float(average_precision_score(y_bin, p))
    except ValueError:
        aucpr = float("nan")
    try:
        aucroc = float(roc_auc_score(y_bin, p))
    except ValueError:
        aucroc = float("nan")

    return {
        "threshold": float(threshold),
        "mae_all": mae_all,
        "mae_nonzero": mae_nz,
        "mae_threshold_gated": mae_gate,
        "nonzero_precision": float(prec),
        "nonzero_recall": float(rec),
        "nonzero_f1": float(f1),
        "aucpr": aucpr,
        "aucroc": aucroc,
        "pred_nonzero_rate": float(p_bin.mean()),
        "actual_nonzero_rate": float(y_bin.mean()),
        "mean_p": float(p.mean()),
        "mean_final": float(yhat.mean()),
        "mean_actual": float(y.mean()),
    }


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("CAUSAL LIGHTWEIGHT PERFORMANCE EVAL")
    print("=" * 70)
    print(f"data_dir={data_dir}")
    print(f"max_skus={args.max_skus} epochs={args.epochs} batch={args.batch_size}")

    t0 = time.time()
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    holiday_train = pd.read_csv(data_dir / "holiday_features_train.csv")
    holiday_val = pd.read_csv(data_dir / "holiday_features_val.csv")
    holiday_test = pd.read_csv(data_dir / "holiday_features_test.csv")
    print(f"loaded CSVs in {time.time()-t0:.1f}s")

    sku_set = select_skus(train_df, args.max_skus, args.seed)
    train_df, holiday_train = filter_aligned(train_df, holiday_train, sku_set)
    val_df, holiday_val = filter_aligned(val_df, holiday_val, sku_set)
    test_df, holiday_test = filter_aligned(test_df, holiday_test, sku_set)

    # Shared SKU coding from train vocabulary
    categories = pd.Categorical(train_df["id_var"])
    sku_map = {k: i for i, k in enumerate(categories.categories)}
    n_skus = len(sku_map)

    def encode_sku(df):
        return df["id_var"].map(sku_map).fillna(-1).astype(np.int32).to_numpy().reshape(-1, 1)

    # Drop unknown SKUs on val/test (should be none with shared set)
    for name, df in [("val", val_df), ("test", test_df)]:
        unknown = ~df["id_var"].isin(sku_map)
        if unknown.any():
            print(f"WARNING dropping {unknown.sum()} unknown SKU rows from {name}")

    cfg = load_feature_config()
    print(f"feature_config v{cfg.config['metadata']['version']} n_features={cfg.total_features}")

    t1 = time.time()
    X_train_df, states = cfg.create_features(train_df, holiday_train, return_states=True)
    X_val_df, states = cfg.create_features(
        val_df, holiday_val, prior_states=states, return_states=True
    )
    X_test_df, states = cfg.create_features(
        test_df, holiday_test, prior_states=states, return_states=True
    )
    print(f"features built in {time.time()-t1:.1f}s  regressor_dim={len(cfg.regressor_indices)}")

    X_train = X_train_df.to_numpy(np.float32)
    X_val = X_val_df.to_numpy(np.float32)
    X_test = X_test_df.to_numpy(np.float32)
    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train = encode_sku(train_df)
    sku_val = encode_sku(val_df)
    sku_test = encode_sku(test_df)

    # Keep only mapped SKUs
    for split_name, X, y, sku in [
        ("train", X_train, y_train, sku_train),
        ("val", X_val, y_val, sku_val),
        ("test", X_test, y_test, sku_test),
    ]:
        pass
    m_val = (sku_val.reshape(-1) >= 0)
    m_test = (sku_test.reshape(-1) >= 0)
    X_val, y_val, sku_val = X_val[m_val], y_val[m_val], sku_val[m_val]
    X_test, y_test, sku_test = X_test[m_test], y_test[m_test], sku_test[m_test]

    tr = split_components(X_train, cfg)
    va = split_components(X_val, cfg)
    te = split_components(X_test, cfg)

    zero_rate = float((y_train == 0).mean())
    pos_weight = min(20.0, zero_rate / max(1.0 - zero_rate, 1e-3))
    print(f"rows train/val/test={len(y_train):,}/{len(y_val):,}/{len(y_test):,}")
    print(f"n_skus={n_skus} zero_rate={zero_rate*100:.2f}% pos_weight={pos_weight:.2f}")

    model = build_hierarchical_model_lightweight(
        n_temporal_features=len(cfg.trend_indices),
        n_fourier_features=len(cfg.seasonal_indices),
        n_holiday_features=len(cfg.holiday_indices),
        n_lag_features=len(cfg.regressor_indices),
        n_skus=n_skus,
        hidden_dim=args.hidden_dim,
        sku_embedding_dim=args.sku_embedding_dim,
        dropout_rate=0.2,
        use_cross_layers=True,
        use_intermittent=True,
        n_changepoints=15,
    )

    # Scale time_index into [0,1] for changepoint layer defaults
    # (time_index is days since epoch — normalize using train stats)
    t_idx = cfg.trend_indices[0]
    t_min = float(X_train[:, t_idx].min())
    t_max = float(X_train[:, t_idx].max())
    span = max(t_max - t_min, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - t_min) / span
    tr = split_components(X_train, cfg)
    va = split_components(X_val, cfg)
    te = split_components(X_test, cfg)

    loss = {
        "final_forecast": masked_mae_loss(use_mse=False),
        "non_zero_probability": weighted_bce_loss(pos_weight=pos_weight),
    }
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=loss,
        loss_weights={"final_forecast": 1.0, "non_zero_probability": 1.0},
        metrics={
            "final_forecast": ["mae"],
            "non_zero_probability": ["accuracy"],
        },
    )

    y_train_bin = (y_train > 0).astype(np.float32)
    y_val_bin = (y_val > 0).astype(np.float32)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]

    t2 = time.time()
    history = model.fit(
        [*tr, sku_train],
        {
            "final_forecast": y_train,
            "base_forecast": y_train,
            "non_zero_probability": y_train_bin,
        },
        validation_data=(
            [*va, sku_val],
            {
                "final_forecast": y_val,
                "base_forecast": y_val,
                "non_zero_probability": y_val_bin,
            },
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    train_s = time.time() - t2
    print(f"trained in {train_s:.1f}s")

    thr = max(0.05, 1.0 - zero_rate)
    val_metrics = evaluate(model, [*va, sku_val], y_val, zero_rate, thr)
    test_metrics = evaluate(model, [*te, sku_test], y_test, zero_rate, thr)

    # Naive baselines on test
    mean_nz = float(y_train[y_train > 0].mean()) if (y_train > 0).any() else 0.0
    baseline_mean = float(mean_absolute_error(y_test, np.full_like(y_test, y_train.mean())))
    baseline_zero = float(mean_absolute_error(y_test, np.zeros_like(y_test)))
    # Seasonal-naive: lag_1 already in features — use previous day demand proxy from y shifted per sku is heavy; use train mean on nonzero days * prior
    baseline_nz_rate = float(mean_absolute_error(y_test, np.full_like(y_test, (1 - zero_rate) * mean_nz)))

    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus_used": n_skus,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "feature_version": cfg.config["metadata"]["version"],
            "n_features": cfg.total_features,
            "regressor_features": cfg.regressor_indices,
            "zero_rate_train": zero_rate,
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "train_seconds": train_s,
        },
        "baselines_test": {
            "mae_predict_train_mean": baseline_mean,
            "mae_predict_zero": baseline_zero,
            "mae_predict_expected_value": baseline_nz_rate,
            "historical_lightgbm_test_mae": 1.2864,
            "historical_twostage_test_mae": 0.9869,
            "historical_simple_additive_test_mae": 0.9876,
        },
        "val": val_metrics,
        "test": test_metrics,
        "history_tail": {
            k: [float(x) for x in v[-3:]] for k, v in history.history.items()
        },
    }

    out = Path(args.out_json)
    out.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(json.dumps({"val": val_metrics, "test": test_metrics, "baselines_test": results["baselines_test"]}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
