#!/usr/bin/env python3
"""Adaptive-loss performance eval (closer to production training recipe)."""

from __future__ import annotations

import argparse
import json
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
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
    )
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden_dim", type=int, default=48)
    p.add_argument("--sku_embedding_dim", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=0.0025)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=str(ROOT / "eval_results_adaptive.json"))
    return p.parse_args()


def filter_aligned(df, holidays, sku_set):
    if sku_set is None:
        return df.reset_index(drop=True), holidays.reset_index(drop=True)
    mask = df["id_var"].isin(sku_set).to_numpy()
    return df.loc[mask].reset_index(drop=True), holidays.loc[mask].reset_index(drop=True)


def split_components(X, cfg):
    return (
        X[:, cfg.trend_indices].astype(np.float32),
        X[:, cfg.seasonal_indices].astype(np.float32),
        X[:, cfg.holiday_indices].astype(np.float32),
        X[:, cfg.regressor_indices].astype(np.float32),
    )


def evaluate(model, inputs, y, zero_rate):
    threshold = max(0.05, 1.0 - zero_rate)
    # Also report best-F1 threshold on this split for diagnostics
    preds = model.predict(inputs, batch_size=4096, verbose=0)
    yhat = np.asarray(preds["final_forecast"]).reshape(-1)
    p = np.asarray(preds["non_zero_probability"]).reshape(-1)
    y = y.reshape(-1)
    y_bin = (y > 0).astype(np.float32)

    def metrics_at(thr):
        p_bin = (p >= thr).astype(np.float32)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_bin, p_bin, average="binary", zero_division=0
        )
        yhat_gate = yhat * p_bin
        nz = y > 0
        return {
            "threshold": float(thr),
            "mae_all": float(mean_absolute_error(y, yhat)),
            "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else None,
            "mae_threshold_gated": float(mean_absolute_error(y, yhat_gate)),
            "nonzero_precision": float(prec),
            "nonzero_recall": float(rec),
            "nonzero_f1": float(f1),
            "pred_nonzero_rate": float(p_bin.mean()),
        }

    # sweep thresholds
    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        m = metrics_at(thr)
        if best is None or m["nonzero_f1"] > best["nonzero_f1"]:
            best = m

    out = metrics_at(threshold)
    out["aucpr"] = float(average_precision_score(y_bin, p))
    out["aucroc"] = float(roc_auc_score(y_bin, p))
    out["mean_p"] = float(p.mean())
    out["mean_final"] = float(yhat.mean())
    out["mean_actual"] = float(y.mean())
    out["best_f1_threshold"] = best
    return out


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    data_dir = Path(args.data_dir)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    holiday_train = pd.read_csv(data_dir / "holiday_features_train.csv")
    holiday_val = pd.read_csv(data_dir / "holiday_features_val.csv")
    holiday_test = pd.read_csv(data_dir / "holiday_features_test.csv")

    skus = train_df["id_var"].unique()
    rng = np.random.default_rng(args.seed)
    chosen = set(rng.choice(skus, size=min(args.max_skus, len(skus)), replace=False))
    train_df, holiday_train = filter_aligned(train_df, holiday_train, chosen)
    val_df, holiday_val = filter_aligned(val_df, holiday_val, chosen)
    test_df, holiday_test = filter_aligned(test_df, holiday_test, chosen)

    cats = pd.Categorical(train_df["id_var"])
    sku_map = {k: i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return df["id_var"].map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)

    cfg = load_feature_config()
    print("Building causal features...")
    Xtr_df, states = cfg.create_features(train_df, holiday_train, return_states=True)
    Xva_df, states = cfg.create_features(val_df, holiday_val, prior_states=states, return_states=True)
    Xte_df, states = cfg.create_features(test_df, holiday_test, prior_states=states, return_states=True)

    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    # normalize time_index to [0,1]
    t_idx = cfg.trend_indices[0]
    tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val, sku_test = enc(train_df), enc(val_df), enc(test_df)

    tr, va, te = split_components(X_train, cfg), split_components(X_val, cfg), split_components(X_test, cfg)
    zero_rate = float((y_train == 0).mean())
    avg_nz = float(y_train[y_train > 0].mean())
    pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))
    print(f"n_skus={n_skus} rows={len(y_train)}/{len(y_val)}/{len(y_test)} zero_rate={zero_rate:.3f}")

    base = build_hierarchical_model_lightweight(
        n_temporal_features=len(cfg.trend_indices),
        n_fourier_features=len(cfg.seasonal_indices),
        n_holiday_features=len(cfg.holiday_indices),
        n_lag_features=len(cfg.regressor_indices),
        n_skus=n_skus,
        hidden_dim=args.hidden_dim,
        sku_embedding_dim=args.sku_embedding_dim,
        dropout_rate=0.23,
        use_cross_layers=True,
        use_intermittent=True,
        n_changepoints=15,
    )
    # build once
    _ = base([*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)], training=False)

    wrapped = AdaptiveWeightedModel(
        base_model=base,
        bce_loss_fn=WeightedBCELoss(weight_nonzero=pos_weight, weight_zero=1.0),
        mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        zero_rate=zero_rate,
        avg_nonzero_demand=avg_nz,
        pos_weight=pos_weight,
        use_fixed_weights=True,
        bce_weight=0.5,
        mae_weight=0.5,
    )
    wrapped.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))

    ytr = {
        "base_forecast": y_train,
        "final_forecast": y_train,
        "non_zero_binary": (y_train > 0).astype(np.float32),
        "non_zero_probability": (y_train > 0).astype(np.float32),
    }
    yva = {
        "base_forecast": y_val,
        "final_forecast": y_val,
        "non_zero_binary": (y_val > 0).astype(np.float32),
        "non_zero_probability": (y_val > 0).astype(np.float32),
    }

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]

    t0 = time.time()
    hist = wrapped.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=2,
    )
    train_s = time.time() - t0

    val_m = evaluate(wrapped, [*va, sku_val], y_val, zero_rate)
    test_m = evaluate(wrapped, [*te, sku_test], y_test, zero_rate)

    results = {
        "config": {
            "recipe": "AdaptiveWeightedModel fixed 50/50",
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "epochs": args.epochs,
            "feature_version": "1.4",
            "zero_rate": zero_rate,
            "train_seconds": train_s,
            "train_rows": int(len(y_train)),
            "val_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
        },
        "baselines_test_mae": {
            "predict_zero": float(mean_absolute_error(y_test, np.zeros_like(y_test))),
            "historical_lightgbm": 1.2864,
            "historical_twostage": 0.9869,
            "historical_simple_additive": 0.9876,
        },
        "val": val_m,
        "test": test_m,
    }
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
