#!/usr/bin/env python3
"""Shared helpers for the v1.6 same-feature bake-off."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)

from deepsequence_hierarchical_attention.forecast_postprocess import round_forecast


def build_deepar(lookback, n_skus, n_channels=4, hidden=64):
    hist = tf.keras.Input(shape=(lookback, n_channels), name="history")
    sku = tf.keras.Input(shape=(1,), dtype=tf.int32, name="sku_id")
    emb = tf.keras.layers.Embedding(n_skus, 8)(sku)
    emb = tf.keras.layers.Flatten()(emb)
    emb_t = tf.keras.layers.RepeatVector(lookback)(emb)
    x = tf.keras.layers.Concatenate(axis=-1)([hist, emb_t])
    h = tf.keras.layers.LSTM(hidden)(x)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    base = tf.keras.layers.Dense(1, activation="softplus", name="base_forecast")(h)
    p = tf.keras.layers.Dense(1, activation="sigmoid", name="non_zero_probability")(h)
    final = tf.keras.layers.Multiply(name="final_forecast")([base, p])
    return tf.keras.Model(
        [hist, sku],
        {"final_forecast": final, "non_zero_probability": p, "base_forecast": base},
        name="deepar_lite",
    )

def build_transformer(lookback, n_skus, n_channels=4, d_model=64, n_heads=4):
    hist = tf.keras.Input(shape=(lookback, n_channels), name="history")
    sku = tf.keras.Input(shape=(1,), dtype=tf.int32, name="sku_id")
    emb = tf.keras.layers.Embedding(n_skus, 8)(sku)
    emb = tf.keras.layers.Flatten()(emb)
    emb_t = tf.keras.layers.RepeatVector(lookback)(emb)
    x = tf.keras.layers.Concatenate(axis=-1)([hist, emb_t])
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=max(1, d_model // n_heads)
    )(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)
    ff = tf.keras.layers.Dense(d_model * 2, activation="relu")(x)
    ff = tf.keras.layers.Dense(d_model)(ff)
    x = tf.keras.layers.LayerNormalization()(x + ff)
    h = tf.keras.layers.GlobalAveragePooling1D()(x)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    base = tf.keras.layers.Dense(1, activation="softplus", name="base_forecast")(h)
    p = tf.keras.layers.Dense(1, activation="sigmoid", name="non_zero_probability")(h)
    final = tf.keras.layers.Multiply(name="final_forecast")([base, p])
    return tf.keras.Model(
        [hist, sku],
        {"final_forecast": final, "non_zero_probability": p, "base_forecast": base},
        name="temporal_transformer",
    )

def predict_seq(model, X, sku):
    pred = model.predict([X, sku], batch_size=4096, verbose=0)
    return (
        np.asarray(pred["final_forecast"]).reshape(-1),
        np.asarray(pred["non_zero_probability"]).reshape(-1),
    )

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

def train_volume_terciles(train_df: pd.DataFrame) -> dict:
    """SKU → {low, mid, high} from train sum(Quantity) terciles."""
    vol = train_df.groupby("id_var")["Quantity"].sum().astype(np.float64)
    # qcut can fail on ties; rank then cut
    ranks = vol.rank(method="first")
    labels = pd.qcut(ranks, 3, labels=["low", "mid", "high"])
    mapping = labels.to_dict()
    stats = {}
    for band in ("low", "mid", "high"):
        skus = [s for s, b in mapping.items() if b == band]
        stats[band] = {
            "n_skus": len(skus),
            "train_volume_sum": float(vol.loc[skus].sum()),
            "train_volume_mean_sku": float(vol.loc[skus].mean()),
            "train_volume_min": float(vol.loc[skus].min()),
            "train_volume_max": float(vol.loc[skus].max()),
            "train_zero_rate": float(
                (train_df.loc[train_df["id_var"].isin(skus), "Quantity"] == 0).mean()
            ),
        }
    return mapping, stats

def kpi_block(y, yhat, p=None):
    y = np.asarray(y, np.float64).reshape(-1)
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    yhat_r = round_forecast(yhat)
    nz = y > 0
    out = {
        "n_rows": int(len(y)),
        "n_nonzero": int(nz.sum()),
        "zero_rate": float((~nz).mean()) if len(y) else None,
        "mae_all": float(mean_absolute_error(y, yhat)) if len(y) else None,
        "mae_all_rounded": float(mean_absolute_error(y, yhat_r)) if len(y) else None,
        "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else None,
        "mean_final": float(yhat.mean()) if len(y) else None,
        "mean_actual": float(y.mean()) if len(y) else None,
        "bias": float(yhat.mean() - y.mean()) if len(y) else None,
        "predict_zero_mae": float(mean_absolute_error(y, np.zeros_like(y))) if len(y) else None,
    }
    if p is not None and len(y) and len(np.unique((y > 0).astype(int))) == 2:
        p = np.asarray(p, np.float64).reshape(-1)
        yb = (y > 0).astype(np.float64)
        out["mean_p"] = float(p.mean())
        out["aucroc"] = float(roc_auc_score(yb, p))
        out["aucpr"] = float(average_precision_score(yb, p))
    return out

def strata_report(y, yhat, p, skus, volume_map):
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    skus = np.asarray(skus).reshape(-1)
    p = None if p is None else np.asarray(p).reshape(-1)
    bands = np.array([volume_map.get(s, "unk") for s in skus])
    out = {"overall": kpi_block(y, yhat, p)}
    # volume-weighted MAE: weight each row by that SKU's train volume share of total
    # (approx: use band mean volume as proxy per row belonging to band)
    for band in ("low", "mid", "high"):
        m = bands == band
        out[band] = kpi_block(
            y[m], yhat[m], None if p is None else p[m]
        )
        out[band]["n_skus_in_pred"] = int(len(set(skus[m].tolist())))
    # equal-SKU mean of per-SKU MAE (rounded) within overall
    per_sku = []
    for s in np.unique(skus):
        m = skus == s
        if m.sum() == 0:
            continue
        per_sku.append(
            {
                "sku": str(s),
                "band": volume_map.get(s, "unk"),
                "mae_rounded": float(
                    mean_absolute_error(y[m], round_forecast(yhat[m]))
                ),
                "mae_nonzero": float(
                    mean_absolute_error(y[m][y[m] > 0], yhat[m][y[m] > 0])
                )
                if (y[m] > 0).any()
                else None,
                "n_rows": int(m.sum()),
            }
        )
    by_band_sku = {}
    for band in ("low", "mid", "high"):
        rows = [r for r in per_sku if r["band"] == band]
        maes = [r["mae_rounded"] for r in rows]
        nzs = [r["mae_nonzero"] for r in rows if r["mae_nonzero"] is not None]
        by_band_sku[band] = {
            "equal_sku_mae_rounded_mean": float(np.mean(maes)) if maes else None,
            "equal_sku_mae_nonzero_mean": float(np.mean(nzs)) if nzs else None,
            "n_skus": len(rows),
        }
    out["equal_sku_means"] = by_band_sku
    return out
