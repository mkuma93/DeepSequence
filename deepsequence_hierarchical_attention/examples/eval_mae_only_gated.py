#!/usr/bin/env python3
"""
MAE-only on gated output: L = MAE(y, softplus(base) * sigmoid(gate)).

No separate BCE. Gradients flow through the product so the gate is learned
implicitly whenever predicting near-zero is cheaper than over-predicting.
"""

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
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_mae_only_gated.json"),
    )
    return p.parse_args()


def filter_aligned(df, holidays, sku_set):
    mask = df["id_var"].isin(sku_set).to_numpy()
    return df.loc[mask].reset_index(drop=True), holidays.loc[mask].reset_index(drop=True)


def split_components(X, cfg):
    return (
        X[:, cfg.trend_indices].astype(np.float32),
        X[:, cfg.seasonal_indices].astype(np.float32),
        X[:, cfg.holiday_indices].astype(np.float32),
        X[:, cfg.regressor_indices].astype(np.float32),
    )


class GatedMAEModel(tf.keras.Model):
    """Optimize only MAE(y, final_forecast) on ALL samples (zeros included)."""

    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.base_mae = tf.keras.metrics.MeanAbsoluteError(name="base_mae")

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        if isinstance(data, (tuple, list)) and len(data) == 3:
            x, y, _ = data
        else:
            x, y = data
        y_true = y["final_forecast"]

        with tf.GradientTape() as tape:
            out = self.base_model(x, training=True)
            yhat = out["final_forecast"]
            loss = tf.reduce_mean(tf.abs(y_true - yhat))
            if self.base_model.losses:
                loss = loss + tf.add_n(self.base_model.losses)

        vars_ = self.base_model.trainable_variables
        grads = tape.gradient(loss, vars_)
        fixed = []
        for g in grads:
            if g is None:
                fixed.append(None)
            else:
                fixed.append(tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)))
        fixed, _ = tf.clip_by_global_norm(fixed, 5.0)
        self.optimizer.apply_gradients(zip(fixed, vars_))

        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(y_true, yhat)
        base = out.get("base_forecast", yhat)
        self.base_mae.update_state(y_true, base)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
            "base_mae": self.base_mae.result(),
        }

    def test_step(self, data):
        if isinstance(data, (tuple, list)) and len(data) == 3:
            x, y, _ = data
        else:
            x, y = data
        y_true = y["final_forecast"]
        out = self.base_model(x, training=False)
        yhat = out["final_forecast"]
        loss = tf.reduce_mean(tf.abs(y_true - yhat))
        if self.base_model.losses:
            loss = loss + tf.add_n(self.base_model.losses)
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(y_true, yhat)
        base = out.get("base_forecast", yhat)
        self.base_mae.update_state(y_true, base)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_tracker.result(),
            "base_mae": self.base_mae.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_tracker, self.base_mae]


def evaluate(model, inputs, y, zero_rate):
    preds = model.predict(inputs, batch_size=4096, verbose=0)
    yhat = np.asarray(preds["final_forecast"]).reshape(-1)
    p = np.asarray(preds["non_zero_probability"]).reshape(-1)
    base = np.asarray(preds.get("base_forecast", yhat)).reshape(-1)
    y = y.reshape(-1)
    y_bin = (y > 0).astype(np.float32)
    prior_thr = max(0.05, 1.0 - zero_rate)

    def at(thr):
        pred = (p >= thr).astype(np.float32)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_bin, pred, average="binary", zero_division=0
        )
        nz = y > 0
        return {
            "threshold": float(thr),
            "accuracy": float((pred == y_bin).mean()),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mae_all": float(mean_absolute_error(y, yhat)),
            "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else None,
            "mae_gated": float(mean_absolute_error(y, yhat * pred)),
            "pred_nonzero_rate": float(pred.mean()),
        }

    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        m = at(thr)
        if best is None or m["f1"] > best["f1"]:
            best = m

    out = at(prior_thr)
    out["aucpr"] = float(average_precision_score(y_bin, p))
    out["aucroc"] = float(roc_auc_score(y_bin, p))
    out["mean_p"] = float(p.mean())
    out["mean_final"] = float(yhat.mean())
    out["mean_base"] = float(base.mean())
    out["mean_actual"] = float(y.mean())
    out["best_f1"] = best
    out["accuracy_at_0_5"] = float((((p >= 0.5).astype(np.float32) == y_bin).mean()))
    return out


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    data_dir = Path(args.data_dir)

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

    cats = pd.Categorical(train_df["id_var"])
    sku_map = {k: i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return df["id_var"].map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)

    cfg = load_feature_config()
    print("Building causal features...")
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

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val, sku_test = enc(train_df), enc(val_df), enc(test_df)
    tr, va, te = (
        split_components(X_train, cfg),
        split_components(X_val, cfg),
        split_components(X_test, cfg),
    )
    zero_rate = float((y_train == 0).mean())
    print(
        f"n_skus={n_skus} rows={len(y_train)}/{len(y_val)}/{len(y_test)} "
        f"zero_rate={zero_rate:.3f}"
    )

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
    _ = base(
        [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
        training=False,
    )

    model = GatedMAEModel(base)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))

    ytr = {"final_forecast": y_train}
    yva = {"final_forecast": y_val}
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]

    print("\n=== MAE-only on sigmoid * softplus (all samples) ===")
    t0 = time.time()
    model.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=2,
    )
    train_s = time.time() - t0

    val_m = evaluate(model, [*va, sku_val], y_val, zero_rate)
    test_m = evaluate(model, [*te, sku_test], y_test, zero_rate)

    results = {
        "config": {
            "recipe": "MAE-only on final=sigmoid*softplus (all samples, no BCE)",
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "epochs": args.epochs,
            "zero_rate": zero_rate,
            "train_seconds": train_s,
        },
        "baselines_test_mae": {
            "predict_zero": float(mean_absolute_error(y_test, np.zeros_like(y_test))),
            "prior_adaptive_bce_mae": 7.40,
            "historical_twostage": 0.9869,
        },
        "val": val_m,
        "test": test_m,
    }
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
