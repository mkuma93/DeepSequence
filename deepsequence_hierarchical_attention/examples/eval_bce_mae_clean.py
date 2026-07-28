#!/usr/bin/env python3
"""
Clean BCE + MAE only (two-term), with explicit [batch,1] shapes.

L = 0.5 * weighted_BCE(y>0, p) + 0.5 * MAE(y, p*softplus)  # MAE on ALL days
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
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--mae_weight", type=float, default=0.5)
    p.add_argument(
        "--mae_nonzero_only",
        action="store_true",
        help="If set, MAE only on y>0 (old adaptive recipe). Default: all days.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_bce_mae_clean.json"),
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


class CleanBCEMAEModel(tf.keras.Model):
    def __init__(
        self,
        base_model,
        zero_rate: float,
        bce_weight: float = 0.5,
        mae_weight: float = 0.5,
        mae_nonzero_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.bce_weight = float(bce_weight)
        self.mae_weight = float(mae_weight)
        self.mae_nonzero_only = bool(mae_nonzero_only)
        nz = max(1.0 - zero_rate, 1e-6)
        self.pos_weight = min(20.0, zero_rate / nz)
        self.zero_rate = float(zero_rate)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.bce_tracker = tf.keras.metrics.Mean(name="bce")
        self.mae_tracker = tf.keras.metrics.Mean(name="mae_term")
        self.final_mae = tf.keras.metrics.MeanAbsoluteError(name="final_mae")
        thr = max(0.05, 1.0 - zero_rate)
        self.prec = tf.keras.metrics.Precision(name="nonzero_precision", thresholds=[thr])
        self.rec = tf.keras.metrics.Recall(name="nonzero_recall", thresholds=[thr])
        self.aucpr = tf.keras.metrics.AUC(curve="PR", name="nonzero_aucpr")
        self.aucroc = tf.keras.metrics.AUC(curve="ROC", name="nonzero_aucroc")

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def _compute(self, y_true, out):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        yhat = tf.reshape(out["final_forecast"], [-1, 1])
        p = tf.reshape(out["non_zero_probability"], [-1, 1])
        y_bin = tf.cast(y_true > 0, tf.float32)

        p_clip = tf.clip_by_value(p, 1e-7, 1.0 - 1e-7)
        bce = tf.reduce_mean(
            -self.pos_weight * y_bin * tf.math.log(p_clip)
            - (1.0 - y_bin) * tf.math.log(1.0 - p_clip)
        )

        abs_err = tf.abs(y_true - yhat)
        if self.mae_nonzero_only:
            mae = tf.reduce_sum(y_bin * abs_err) / (tf.reduce_sum(y_bin) + 1e-6)
        else:
            mae = tf.reduce_mean(abs_err)

        total = self.bce_weight * bce + self.mae_weight * mae
        return total, bce, mae, yhat, p, y_bin

    def train_step(self, data):
        x, y = data[0], data[1]
        y_true = y["final_forecast"]
        with tf.GradientTape() as tape:
            out = self.base_model(x, training=True)
            total, bce, mae, yhat, p, y_bin = self._compute(y_true, out)
            if self.base_model.losses:
                total = total + tf.add_n(self.base_model.losses)
        grads = tape.gradient(total, self.base_model.trainable_variables)
        fixed = [
            None if g is None else tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            for g in grads
        ]
        fixed, _ = tf.clip_by_global_norm(fixed, 5.0)
        self.optimizer.apply_gradients(zip(fixed, self.base_model.trainable_variables))
        self.loss_tracker.update_state(total)
        self.bce_tracker.update_state(bce)
        self.mae_tracker.update_state(mae)
        self.final_mae.update_state(y_true, yhat)
        self.prec.update_state(y_bin, p)
        self.rec.update_state(y_bin, p)
        self.aucpr.update_state(y_bin, p)
        self.aucroc.update_state(y_bin, p)
        return {
            "loss": self.loss_tracker.result(),
            "bce": self.bce_tracker.result(),
            "mae_term": self.mae_tracker.result(),
            "final_mae": self.final_mae.result(),
            "nonzero_aucroc": self.aucroc.result(),
            "nonzero_aucpr": self.aucpr.result(),
        }

    def test_step(self, data):
        x, y = data[0], data[1]
        y_true = y["final_forecast"]
        out = self.base_model(x, training=False)
        total, bce, mae, yhat, p, y_bin = self._compute(y_true, out)
        if self.base_model.losses:
            total = total + tf.add_n(self.base_model.losses)
        self.loss_tracker.update_state(total)
        self.bce_tracker.update_state(bce)
        self.mae_tracker.update_state(mae)
        self.final_mae.update_state(y_true, yhat)
        self.prec.update_state(y_bin, p)
        self.rec.update_state(y_bin, p)
        self.aucpr.update_state(y_bin, p)
        self.aucroc.update_state(y_bin, p)
        return {
            "loss": self.loss_tracker.result(),
            "bce": self.bce_tracker.result(),
            "mae_term": self.mae_tracker.result(),
            "final_mae": self.final_mae.result(),
            "nonzero_aucroc": self.aucroc.result(),
            "nonzero_aucpr": self.aucpr.result(),
        }

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.bce_tracker,
            self.mae_tracker,
            self.final_mae,
            self.prec,
            self.rec,
            self.aucpr,
            self.aucroc,
        ]


def evaluate(model, inputs, y, zero_rate):
    preds = model.predict(inputs, batch_size=4096, verbose=0)
    yhat = np.asarray(preds["final_forecast"]).reshape(-1)
    p = np.asarray(preds["non_zero_probability"]).reshape(-1)
    base = np.asarray(preds["base_forecast"]).reshape(-1)
    y = y.reshape(-1)
    y_bin = (y > 0).astype(np.float32)
    nz = y > 0
    prior_thr = max(0.05, 1.0 - zero_rate)

    def at(thr):
        pred = (p >= thr).astype(np.float32)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_bin, pred, average="binary", zero_division=0
        )
        return {
            "threshold": float(thr),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mae_all": float(mean_absolute_error(y, yhat)),
            "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else None,
            "mae_gated": float(mean_absolute_error(y, yhat * pred)),
        }

    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        m = at(thr)
        if best is None or m["f1"] > best["f1"]:
            best = m
    out = at(prior_thr)
    out.update(
        {
            "aucpr": float(average_precision_score(y_bin, p)),
            "aucroc": float(roc_auc_score(y_bin, p)),
            "mean_p": float(p.mean()),
            "mean_final": float(yhat.mean()),
            "mean_base": float(base.mean()),
            "mean_actual": float(y.mean()),
            "best_f1": best,
        }
    )
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
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(
        val_df, h_va, prior_states=states, return_states=True
    )
    Xte_df, states = cfg.create_features(
        test_df, h_te, prior_states=states, return_states=True
    )
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    t_idx = cfg.trend_indices[0]
    tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    # Labels as [N,1] to match model outputs (avoids silent broadcast bugs)
    y_train = train_df["Quantity"].to_numpy(np.float32).reshape(-1, 1)
    y_val = val_df["Quantity"].to_numpy(np.float32).reshape(-1, 1)
    y_test = test_df["Quantity"].to_numpy(np.float32).reshape(-1, 1)
    sku_train, sku_val, sku_test = enc(train_df), enc(val_df), enc(test_df)
    tr, va, te = (
        split_components(X_train, cfg),
        split_components(X_val, cfg),
        split_components(X_test, cfg),
    )
    zero_rate = float((y_train == 0).mean())
    print(
        f"n_skus={n_skus} zero_rate={zero_rate:.3f} "
        f"mae_nonzero_only={args.mae_nonzero_only} "
        f"bce/mae weights={args.bce_weight}/{args.mae_weight}"
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

    model = CleanBCEMAEModel(
        base_model=base,
        zero_rate=zero_rate,
        bce_weight=args.bce_weight,
        mae_weight=args.mae_weight,
        mae_nonzero_only=args.mae_nonzero_only,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]

    print("\n=== Clean BCE + MAE (shape-safe) ===")
    t0 = time.time()
    model.fit(
        [*tr, sku_train],
        {"final_forecast": y_train},
        validation_data=([*va, sku_val], {"final_forecast": y_val}),
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
            "recipe": "clean_BCE_plus_MAE",
            "bce_weight": args.bce_weight,
            "mae_weight": args.mae_weight,
            "mae_nonzero_only": args.mae_nonzero_only,
            "pos_weight": min(20.0, zero_rate / max(1 - zero_rate, 1e-6)),
            "shape_safe": True,
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "epochs": args.epochs,
            "zero_rate": zero_rate,
            "train_seconds": train_s,
        },
        "baselines_test_mae": {
            "predict_zero": float(mean_absolute_error(y_test, np.zeros_like(y_test))),
            "three_term_composite": 2.04,
            "prior_adaptive_reported": 7.40,
            "historical_twostage": 0.9869,
        },
        "val": val_m,
        "test": test_m,
    }
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
