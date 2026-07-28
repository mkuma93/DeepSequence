#!/usr/bin/env python3
"""Weighted MAE on gated product: all samples, nonzero upweighted."""

from __future__ import annotations

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
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)

SEED, MAX_SKUS, EPOCHS, BS, LR = 42, 800, 15, 1024, 0.0025
DATA = Path(
    "/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data"
)


class WeightedGatedMAE(tf.keras.Model):
    def __init__(self, base_model, nonzero_weight):
        super().__init__()
        self.base_model = base_model
        self.nonzero_weight = float(nonzero_weight)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def _loss(self, y_true, yhat):
        w = 1.0 + (self.nonzero_weight - 1.0) * tf.cast(y_true > 0, tf.float32)
        return tf.reduce_sum(w * tf.abs(y_true - yhat)) / tf.reduce_sum(w)

    def train_step(self, data):
        x, y = data[0], data[1]
        y_true = y["final_forecast"]
        with tf.GradientTape() as tape:
            out = self.base_model(x, training=True)
            yhat = out["final_forecast"]
            loss = self._loss(y_true, yhat)
            if self.base_model.losses:
                loss = loss + tf.add_n(self.base_model.losses)
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        fixed = [
            None if g is None else tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            for g in grads
        ]
        fixed, _ = tf.clip_by_global_norm(fixed, 5.0)
        self.optimizer.apply_gradients(zip(fixed, self.base_model.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.mae.update_state(y_true, yhat)
        return {"loss": self.loss_tracker.result(), "mae": self.mae.result()}

    def test_step(self, data):
        x, y = data[0], data[1]
        y_true = y["final_forecast"]
        out = self.base_model(x, training=False)
        yhat = out["final_forecast"]
        loss = self._loss(y_true, yhat)
        if self.base_model.losses:
            loss = loss + tf.add_n(self.base_model.losses)
        self.loss_tracker.update_state(loss)
        self.mae.update_state(y_true, yhat)
        return {"loss": self.loss_tracker.result(), "mae": self.mae.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae]


def main():
    tf.keras.utils.set_random_seed(SEED)
    train = pd.read_csv(DATA / "train_split.csv", parse_dates=["ds"])
    val = pd.read_csv(DATA / "val_split.csv", parse_dates=["ds"])
    test = pd.read_csv(DATA / "test_split.csv", parse_dates=["ds"])
    htr = pd.read_csv(DATA / "holiday_features_train.csv")
    hva = pd.read_csv(DATA / "holiday_features_val.csv")
    hte = pd.read_csv(DATA / "holiday_features_test.csv")
    rng = np.random.default_rng(SEED)
    chosen = set(rng.choice(train["id_var"].unique(), size=MAX_SKUS, replace=False))

    def filt(df, h):
        m = df["id_var"].isin(chosen).to_numpy()
        return df.loc[m].reset_index(drop=True), h.loc[m].reset_index(drop=True)

    train, htr = filt(train, htr)
    val, hva = filt(val, hva)
    test, hte = filt(test, hte)
    cats = pd.Categorical(train["id_var"])
    smap = {k: i for i, k in enumerate(cats.categories)}

    def enc(df):
        return df["id_var"].map(smap).astype(np.int32).to_numpy().reshape(-1, 1)

    cfg = load_feature_config()
    Xtr, s = cfg.create_features(train, htr, return_states=True)
    Xva, s = cfg.create_features(val, hva, prior_states=s, return_states=True)
    Xte, s = cfg.create_features(test, hte, prior_states=s, return_states=True)
    Xtr, Xva, Xte = [x.to_numpy(np.float32) for x in (Xtr, Xva, Xte)]
    ti = cfg.trend_indices[0]
    tmin, tmax = float(Xtr[:, ti].min()), float(Xtr[:, ti].max())
    span = max(tmax - tmin, 1.0)
    for X in (Xtr, Xva, Xte):
        X[:, ti] = (X[:, ti] - tmin) / span

    def split(X):
        return (
            X[:, cfg.trend_indices],
            X[:, cfg.seasonal_indices],
            X[:, cfg.holiday_indices],
            X[:, cfg.regressor_indices],
        )

    ytr = train["Quantity"].to_numpy(np.float32)
    yva = val["Quantity"].to_numpy(np.float32)
    yte = test["Quantity"].to_numpy(np.float32)
    sku_tr, sku_va, sku_te = enc(train), enc(val), enc(test)
    tr, va, te = split(Xtr), split(Xva), split(Xte)
    n_skus = len(smap)
    zr = float((ytr == 0).mean())
    nz_w = min(20.0, zr / max(1 - zr, 1e-3))
    print(f"n_skus={n_skus} zero_rate={zr:.3f} nonzero_weight={nz_w:.2f}")

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

    model = WeightedGatedMAE(base, nz_w)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR))
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]
    print("=== weighted MAE on sigmoid*softplus (all samples) ===")
    t0 = time.time()
    model.fit(
        [*tr, sku_tr],
        {"final_forecast": ytr},
        validation_data=([*va, sku_va], {"final_forecast": yva}),
        epochs=EPOCHS,
        batch_size=BS,
        callbacks=cb,
        verbose=2,
    )
    print("train_s", time.time() - t0)

    def eval_split(parts, y, sku):
        pred = model.predict([*parts, sku], batch_size=4096, verbose=0)
        yhat = np.asarray(pred["final_forecast"]).reshape(-1)
        p = np.asarray(pred["non_zero_probability"]).reshape(-1)
        basev = np.asarray(pred["base_forecast"]).reshape(-1)
        y = y.reshape(-1)
        yb = (y > 0).astype(np.float32)
        nz = y > 0
        return {
            "mae_all": float(mean_absolute_error(y, yhat)),
            "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])),
            "mean_final": float(yhat.mean()),
            "mean_base": float(basev.mean()),
            "mean_p": float(p.mean()),
            "mean_actual": float(y.mean()),
            "aucroc": float(roc_auc_score(yb, p)),
            "aucpr": float(average_precision_score(yb, p)),
            "acc_0_5": float((((p >= 0.5).astype(np.float32) == yb).mean())),
        }

    res = {
        "recipe": "weighted_MAE_all_on_gated_product",
        "nonzero_weight": nz_w,
        "val": eval_split(va, yva, sku_va),
        "test": eval_split(te, yte, sku_te),
        "predict_zero_test_mae": float(mean_absolute_error(yte, np.zeros_like(yte))),
    }
    out = ROOT / "eval_results_weighted_gated_mae.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
