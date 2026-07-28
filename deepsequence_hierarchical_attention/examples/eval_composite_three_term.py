#!/usr/bin/env python3
"""
Composite loss experiment:
  L = w_gated * MAE(y, p*softplus)          # all days, inverse class weights
    + w_mag   * MAE(y, softplus)            # sale days only
    + alpha   * weighted_BCE(y>0, p)        # light classification

Same 800-SKU slice / causal v1.4 features as prior evals.
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
    p.add_argument("--alpha_bce", type=float, default=0.2)
    p.add_argument("--w_gated", type=float, default=1.0)
    p.add_argument("--w_mag", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_composite_three_term.json"),
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


class ThreeTermCompositeModel(tf.keras.Model):
    """
    L = w_gated * weighted_MAE(y, final)
      + w_mag   * MAE(y, base | y>0)
      + alpha   * weighted_BCE(y>0, p)
    """

    def __init__(
        self,
        base_model,
        zero_rate: float,
        alpha_bce: float = 0.2,
        w_gated: float = 1.0,
        w_mag: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.alpha_bce = float(alpha_bce)
        self.w_gated = float(w_gated)
        self.w_mag = float(w_mag)
        self.zero_rate = float(zero_rate)
        nz = max(1.0 - zero_rate, 1e-6)
        # Inverse-frequency class weights (balanced total mass)
        self.w_zero = 1.0 / (2.0 * max(zero_rate, 1e-6))
        self.w_nonzero = 1.0 / (2.0 * nz)
        self.pos_weight = min(20.0, zero_rate / nz)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.gated_mae_tracker = tf.keras.metrics.Mean(name="gated_mae")
        self.mag_mae_tracker = tf.keras.metrics.Mean(name="mag_mae")
        self.bce_tracker = tf.keras.metrics.Mean(name="bce")
        self.final_mae = tf.keras.metrics.MeanAbsoluteError(name="final_mae")
        thr = max(0.05, 1.0 - zero_rate)
        self.prec = tf.keras.metrics.Precision(name="nonzero_precision", thresholds=[thr])
        self.rec = tf.keras.metrics.Recall(name="nonzero_recall", thresholds=[thr])
        self.aucpr = tf.keras.metrics.AUC(curve="PR", name="nonzero_aucpr")
        self.aucroc = tf.keras.metrics.AUC(curve="ROC", name="nonzero_aucroc")

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def _class_weights(self, y_true):
        is_nz = tf.cast(y_true > 0, tf.float32)
        return self.w_zero * (1.0 - is_nz) + self.w_nonzero * is_nz

    def _compute_losses(self, y_true, out):
        # Force [batch, 1] so (batch,) labels never broadcast to [batch, batch]
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        yhat = tf.reshape(out["final_forecast"], [-1, 1])
        base = tf.reshape(out["base_forecast"], [-1, 1])
        p = tf.reshape(out["non_zero_probability"], [-1, 1])
        y_bin = tf.cast(y_true > 0, tf.float32)
        w = self._class_weights(y_true)

        gated = tf.reduce_sum(w * tf.abs(y_true - yhat)) / tf.reduce_sum(w)

        # Magnitude on sale days only (unweighted within that set)
        nz_sum = tf.reduce_sum(y_bin) + 1e-6
        mag = tf.reduce_sum(y_bin * tf.abs(y_true - base)) / nz_sum

        # Weighted BCE (pos_weight on positive class)
        p_clip = tf.clip_by_value(p, 1e-7, 1.0 - 1e-7)
        bce_pos = -self.pos_weight * y_bin * tf.math.log(p_clip)
        bce_neg = -(1.0 - y_bin) * tf.math.log(1.0 - p_clip)
        bce = tf.reduce_mean(bce_pos + bce_neg)

        total = self.w_gated * gated + self.w_mag * mag + self.alpha_bce * bce
        return total, gated, mag, bce, yhat, base, p, y_bin

    def train_step(self, data):
        if isinstance(data, (tuple, list)) and len(data) == 3:
            x, y, _ = data
        else:
            x, y = data
        y_true = y["final_forecast"]

        with tf.GradientTape() as tape:
            out = self.base_model(x, training=True)
            total, gated, mag, bce, yhat, base, p, y_bin = self._compute_losses(
                y_true, out
            )
            if self.base_model.losses:
                total = total + tf.add_n(self.base_model.losses)

        vars_ = self.base_model.trainable_variables
        grads = tape.gradient(total, vars_)
        fixed = []
        for g in grads:
            if g is None:
                fixed.append(None)
            else:
                fixed.append(tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)))
        fixed, _ = tf.clip_by_global_norm(fixed, 5.0)
        self.optimizer.apply_gradients(zip(fixed, vars_))

        self.loss_tracker.update_state(total)
        self.gated_mae_tracker.update_state(gated)
        self.mag_mae_tracker.update_state(mag)
        self.bce_tracker.update_state(bce)
        self.final_mae.update_state(y_true, yhat)
        self.prec.update_state(y_bin, p)
        self.rec.update_state(y_bin, p)
        self.aucpr.update_state(y_bin, p)
        self.aucroc.update_state(y_bin, p)
        return {
            "loss": self.loss_tracker.result(),
            "gated_mae": self.gated_mae_tracker.result(),
            "mag_mae": self.mag_mae_tracker.result(),
            "bce": self.bce_tracker.result(),
            "final_mae": self.final_mae.result(),
            "nonzero_precision": self.prec.result(),
            "nonzero_recall": self.rec.result(),
            "nonzero_aucpr": self.aucpr.result(),
            "nonzero_aucroc": self.aucroc.result(),
        }

    def test_step(self, data):
        if isinstance(data, (tuple, list)) and len(data) == 3:
            x, y, _ = data
        else:
            x, y = data
        y_true = y["final_forecast"]
        out = self.base_model(x, training=False)
        total, gated, mag, bce, yhat, base, p, y_bin = self._compute_losses(
            y_true, out
        )
        if self.base_model.losses:
            total = total + tf.add_n(self.base_model.losses)

        self.loss_tracker.update_state(total)
        self.gated_mae_tracker.update_state(gated)
        self.mag_mae_tracker.update_state(mag)
        self.bce_tracker.update_state(bce)
        self.final_mae.update_state(y_true, yhat)
        self.prec.update_state(y_bin, p)
        self.rec.update_state(y_bin, p)
        self.aucpr.update_state(y_bin, p)
        self.aucroc.update_state(y_bin, p)
        return {
            "loss": self.loss_tracker.result(),
            "gated_mae": self.gated_mae_tracker.result(),
            "mag_mae": self.mag_mae_tracker.result(),
            "bce": self.bce_tracker.result(),
            "final_mae": self.final_mae.result(),
            "nonzero_precision": self.prec.result(),
            "nonzero_recall": self.rec.result(),
            "nonzero_aucpr": self.aucpr.result(),
            "nonzero_aucroc": self.aucroc.result(),
        }

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.gated_mae_tracker,
            self.mag_mae_tracker,
            self.bce_tracker,
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
    prior_thr = max(0.05, 1.0 - zero_rate)
    nz = y > 0

    def at(thr):
        pred = (p >= thr).astype(np.float32)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_bin, pred, average="binary", zero_division=0
        )
        return {
            "threshold": float(thr),
            "accuracy": float((pred == y_bin).mean()),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mae_all": float(mean_absolute_error(y, yhat)),
            "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else None,
            "mae_gated": float(mean_absolute_error(y, yhat * pred)),
            "mae_base_nonzero": float(mean_absolute_error(y[nz], base[nz]))
            if nz.any()
            else None,
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
    print(
        f"loss: gated={args.w_gated} mag={args.w_mag} alpha_bce={args.alpha_bce} "
        f"w_zero={1/(2*zero_rate):.3f} w_nz={1/(2*(1-zero_rate)):.3f}"
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

    model = ThreeTermCompositeModel(
        base_model=base,
        zero_rate=zero_rate,
        alpha_bce=args.alpha_bce,
        w_gated=args.w_gated,
        w_mag=args.w_mag,
    )
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

    print("\n=== Three-term composite (weighted gated MAE + mag MAE + light BCE) ===")
    t0 = time.time()
    hist = model.fit(
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
            "recipe": "weighted_gated_MAE + nonzero_mag_MAE + light_weighted_BCE",
            "alpha_bce": args.alpha_bce,
            "w_gated": args.w_gated,
            "w_mag": args.w_mag,
            "w_zero": 1.0 / (2.0 * zero_rate),
            "w_nonzero": 1.0 / (2.0 * (1.0 - zero_rate)),
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "epochs": args.epochs,
            "zero_rate": zero_rate,
            "train_seconds": train_s,
            "best_epoch_hint": len(hist.history.get("loss", [])),
        },
        "baselines_test_mae": {
            "predict_zero": float(mean_absolute_error(y_test, np.zeros_like(y_test))),
            "prior_adaptive_bce_mae": 7.40,
            "prior_mae_only": 1.0585,
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
