#!/usr/bin/env python3
"""Shared helpers for the v1.6 same-feature bake-off.

Ops metrics via ``kpi_block`` / ``inventory_cost_metrics``:

  - Sale day shortfall → ``sales_revenue_loss_*`` (revenue loss).
  - No-sale day stock sitting → ``inventory_holding_cost_zero`` (carrying cost).
  - Combined: ``combined_ops_cost_h0p1``; decision economics
    (``decision_economics_report``): cost vs r=C_lost/C_hold with crossover
    selector between lower-overstock (TST) and lower-stockout (DS) profiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

from deepsequence_hierarchical_attention.forecast_postprocess import round_forecast
from deepsequence_hierarchical_attention.inventory_metrics import (
    DEFAULT_CU_CO_RATIOS,
    PRIMARY_COMBINED_OPS_METRIC,
    PRIMARY_HOLDING_METRIC,
    PRIMARY_INVENTORY_METRIC,
    PRIMARY_SALES_REVENUE_LOSS_METRIC,
    SERVICE_CRITICAL_INVENTORY_METRIC,
    inventory_cost_from_kpi_summary,
    inventory_cost_metrics,
)


def resolve_eval_seeds(
    seed: int | None = 42,
    data_seed: int | None = None,
    train_seed: int | None = None,
) -> tuple[int, int]:
    """Split SKU-panel sampling from training RNG.

    Legacy ``--seed`` sets both when ``--data_seed`` / ``--train_seed`` are omitted,
    so existing bake-off commands stay bit-compatible.
    """
    base = 42 if seed is None else int(seed)
    return (
        base if data_seed is None else int(data_seed),
        base if train_seed is None else int(train_seed),
    )


def add_panel_seed_args(parser) -> None:
    """Shared CLI flags for apples-to-apples bake-off panels."""
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Legacy convenience seed: used for both data and training when "
        "--data_seed / --train_seed are omitted.",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=None,
        help="Seed for SKU panel sampling only. Freeze this (or --sku_list) "
        "so every model sees the same train/val/test.",
    )
    parser.add_argument(
        "--train_seed",
        type=int,
        default=None,
        help="Seed for TF/numpy training noise only (init, dropout, shuffle).",
    )
    parser.add_argument(
        "--sku_list",
        default=None,
        help="Path to a frozen SKU list (JSON array or one id per line). "
        "When set, skips sampling and uses this panel for all models.",
    )
    parser.add_argument(
        "--save_sku_list",
        default=None,
        help="Write the chosen SKU ids to this path (JSON array) for reuse.",
    )


def select_eval_skus(
    universe,
    *,
    max_skus: int,
    data_seed: int,
    sku_list_path: str | None = None,
    save_sku_list_path: str | None = None,
) -> list:
    """Return the bake-off SKU panel (stable list; callers may wrap in set).

    Prefer ``sku_list_path`` for locked comparisons. Otherwise sample
    ``min(max_skus, len(universe))`` with ``data_seed``.
    """
    from pathlib import Path

    universe = list(universe)
    universe_set = set(universe)

    if sku_list_path:
        path = Path(sku_list_path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"Empty SKU list: {path}")
        if text.startswith("["):
            import json

            chosen = json.loads(text)
        else:
            chosen = [line.strip() for line in text.splitlines() if line.strip()]
        missing = [sku for sku in chosen if sku not in universe_set]
        # Allow typed mismatch (e.g. int ids in CSV vs str in list)
        if missing:
            as_str = {str(sku) for sku in universe}
            chosen = [str(sku) for sku in chosen]
            missing = [sku for sku in chosen if sku not in as_str]
            if missing:
                raise SystemExit(
                    f"{len(missing)} SKUs from {path} are not in the panel "
                    f"(e.g. {missing[:3]})"
                )
            # Remap to native universe dtype
            native = {str(sku): sku for sku in universe}
            chosen = [native[sku] for sku in chosen]
        if not chosen:
            raise SystemExit(f"No SKUs resolved from {path}")
    else:
        n = min(int(max_skus), len(universe))
        rng = np.random.default_rng(int(data_seed))
        chosen = list(rng.choice(universe, size=n, replace=False))

    if save_sku_list_path:
        import json
        from pathlib import Path

        out = Path(save_sku_list_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Persist as strings for portability across loaders.
        out.write_text(
            json.dumps([str(sku) for sku in chosen], indent=2) + "\n",
            encoding="utf-8",
        )

    return chosen


def class_balance_pos_weight(y: np.ndarray) -> float:
    """Data-driven BCE pos weight from this panel's class counts (neg/pos)."""
    y = np.asarray(y).reshape(-1)
    n_pos = float(np.sum(y > 0))
    n_neg = float(np.sum(y <= 0))
    return float(n_neg / max(n_pos, 1.0))


def calibrate_iwmae_gate(
    y_true: np.ndarray,
    yhat: np.ndarray,
    p: np.ndarray | None = None,
    *,
    scales: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Fit a post-hoc gate/scale on validation to minimize rounded IWMAE.

    Applies ``yhat_cal = scale * where(p >= threshold, yhat, 0)``.
    If ``p`` is None, only the scale is searched.
    """
    y_true = np.asarray(y_true, np.float64).reshape(-1)
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    if scales is None:
        scales = np.linspace(0.35, 1.35, 21)
    if thresholds is None:
        thresholds = (
            np.concatenate([[0.0], np.linspace(0.05, 0.85, 17)])
            if p is not None
            else np.asarray([0.0])
        )
    p_arr = None if p is None else np.asarray(p, np.float64).reshape(-1)

    best = {"scale": 1.0, "threshold": 0.0, "iwmae_rounded": float("inf")}
    for thr in thresholds:
        masked = yhat if p_arr is None else np.where(p_arr >= thr, yhat, 0.0)
        for scale in scales:
            cand = scale * masked
            score = kpi_block(y_true, cand)["iwmae_rounded"]
            if score is not None and score < best["iwmae_rounded"]:
                best = {
                    "scale": float(scale),
                    "threshold": float(thr),
                    "iwmae_rounded": float(score),
                }
    return best


def apply_iwmae_gate(
    yhat: np.ndarray,
    p: np.ndarray | None,
    *,
    scale: float = 1.0,
    threshold: float = 0.0,
) -> np.ndarray:
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    if p is not None and threshold > 0.0:
        p = np.asarray(p, np.float64).reshape(-1)
        yhat = np.where(p >= threshold, yhat, 0.0)
    return (float(scale) * yhat).astype(np.float64)


def calibrate_probability_temperature(
    y_true: np.ndarray,
    yhat: np.ndarray,
    p: np.ndarray,
    *,
    temperatures: np.ndarray | None = None,
) -> dict:
    """
    Fit a post-hoc temperature on ``p`` that minimizes rounded IWMAE.

    Treats ``yhat = base * p`` and rebuilds ``yhat' = base * sigmoid(logit(p)/T)``.
    ``T>1`` softens (lowers mean_p when p is mid-range); ``T<1`` sharpens.
    """
    y_true = np.asarray(y_true, np.float64).reshape(-1)
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    p = np.clip(np.asarray(p, np.float64).reshape(-1), 1e-6, 1.0 - 1e-6)
    base = yhat / p
    if temperatures is None:
        temperatures = np.concatenate(
            [np.linspace(0.5, 0.95, 10), [1.0], np.linspace(1.05, 2.5, 15)]
        )

    best = {"temperature": 1.0, "iwmae_rounded": float("inf")}
    logit = np.log(p) - np.log(1.0 - p)
    for t in temperatures:
        p_cal = 1.0 / (1.0 + np.exp(-logit / float(t)))
        cand = np.maximum(base * p_cal, 0.0)
        score = kpi_block(y_true, cand, p_cal)["iwmae_rounded"]
        if score is not None and score < best["iwmae_rounded"]:
            best = {
                "temperature": float(t),
                "iwmae_rounded": float(score),
                "mean_p": float(p_cal.mean()),
            }
    return best


def apply_probability_temperature(
    yhat: np.ndarray,
    p: np.ndarray,
    *,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply temperature to ``p`` and rebuild ``yhat = (yhat/p) * p_cal``."""
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    p = np.clip(np.asarray(p, np.float64).reshape(-1), 1e-6, 1.0 - 1e-6)
    if abs(float(temperature) - 1.0) < 1e-12:
        return yhat.astype(np.float64), p.astype(np.float64)
    base = yhat / p
    logit = np.log(p) - np.log(1.0 - p)
    p_cal = 1.0 / (1.0 + np.exp(-logit / float(temperature)))
    return (base * p_cal).astype(np.float64), p_cal.astype(np.float64)


def seasonal_naive_mae_scale(y: np.ndarray, season: int = 7) -> float | None:
    """In-sample mean |y_t - y_{t-season}| used as MASE / RMSSE denominator."""
    y = np.asarray(y, np.float64).reshape(-1)
    if len(y) <= season:
        return None
    scale = float(np.mean(np.abs(y[season:] - y[:-season])))
    return scale if scale > 1e-12 else None


def train_mase_scale(train_df: pd.DataFrame, season: int = 7) -> float | None:
    """Pooled seasonal-naive MAE over all series in train (daily panel)."""
    scales = []
    for _, g in train_df.sort_values("ds").groupby("id_var", sort=False):
        s = seasonal_naive_mae_scale(g["Quantity"].to_numpy(), season=season)
        if s is not None:
            scales.append(s)
    if not scales:
        return None
    return float(np.mean(scales))


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


def _grn(x, units, dropout=0.1, name="grn"):
    """Gated Residual Network (TFT building block), simplified."""
    skip = (
        x
        if int(x.shape[-1]) == units
        else tf.keras.layers.Dense(units, name=f"{name}_skip")(x)
    )
    h = tf.keras.layers.Dense(units, activation="elu", name=f"{name}_fc1")(x)
    h = tf.keras.layers.Dense(units, name=f"{name}_fc2")(h)
    h = tf.keras.layers.Dropout(dropout, name=f"{name}_drop")(h)
    gate = tf.keras.layers.Dense(units, activation="sigmoid", name=f"{name}_gate")(x)
    out = skip + gate * h
    return tf.keras.layers.LayerNormalization(name=f"{name}_ln")(out)


def build_tft(
    lookback,
    n_skus,
    n_channels=4,
    d_model=64,
    n_heads=4,
    lstm_hidden=64,
    dropout=0.1,
):
    """
    TFT-lite for the same sequence contract as DeepAR/TST.

    Uses: SKU static embedding, per-timestep variable selection (softmax over
    channels), GRN, LSTM encoder, causal multi-head attention, gated
    intermittent head (base × p) — fair bake-off with other sequence models.
    """
    hist = tf.keras.Input(shape=(lookback, n_channels), name="history")
    sku = tf.keras.Input(shape=(1,), dtype=tf.int32, name="sku_id")

    # Static context from SKU
    static = tf.keras.layers.Embedding(n_skus, d_model, name="sku_emb")(sku)
    static = tf.keras.layers.Flatten()(static)
    static = _grn(static, d_model, dropout=dropout, name="static_grn")

    # Variable selection over input channels (soft weights shared across time)
    # scores: (B, C) -> softmax over channels, broadcast to (B, L, C)
    vs_scores = tf.keras.layers.Dense(n_channels, name="var_select_logits")(static)
    vs_weights = tf.keras.layers.Activation("softmax", name="var_select_weights")(vs_scores)
    vs_weights_t = tf.keras.layers.RepeatVector(lookback)(vs_weights)
    selected = tf.keras.layers.Multiply(name="var_selected")([hist, vs_weights_t])

    # Project selected inputs + static context
    x = tf.keras.layers.Dense(d_model, name="input_proj")(selected)
    static_t = tf.keras.layers.RepeatVector(lookback)(static)
    x = tf.keras.layers.Add(name="add_static")([x, static_t])
    x = _grn(x, d_model, dropout=dropout, name="temporal_grn")

    # Local processing (LSTM encoder) + self-attention (TFT-style)
    lstm_out = tf.keras.layers.LSTM(
        lstm_hidden, return_sequences=True, name="lstm_encoder"
    )(x)
    lstm_out = tf.keras.layers.Dense(d_model, name="lstm_proj")(lstm_out)
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=max(1, d_model // n_heads),
        name="tft_mha",
    )(lstm_out, lstm_out, use_causal_mask=True)
    x = tf.keras.layers.Add(name="attn_residual")([lstm_out, attn])
    x = tf.keras.layers.LayerNormalization(name="attn_ln")(x)
    x = _grn(x, d_model, dropout=dropout, name="post_attn_grn")

    # Predict from last encoder step
    h = x[:, -1, :]
    h = tf.keras.layers.Dense(32, activation="relu", name="head_hidden")(h)
    base = tf.keras.layers.Dense(1, activation="softplus", name="base_forecast")(h)
    p = tf.keras.layers.Dense(1, activation="sigmoid", name="non_zero_probability")(h)
    final = tf.keras.layers.Multiply(name="final_forecast")([base, p])
    return tf.keras.Model(
        [hist, sku],
        {"final_forecast": final, "non_zero_probability": p, "base_forecast": base},
        name="tft_lite",
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
    if cfg.holiday_indices:
        holiday = X[:, cfg.holiday_indices].astype(np.float32)
    else:
        # Model expects a 1-d holiday dummy when holidays are disabled
        holiday = np.zeros((X.shape[0], 1), dtype=np.float32)
    return (
        X[:, cfg.trend_indices].astype(np.float32),
        X[:, cfg.seasonal_indices].astype(np.float32),
        holiday,
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


def kpi_block(y, yhat, p=None, mase_scale: float | None = None):
    """Intermittent-aware forecast KPIs.

    All-day MAE alone is weak under high zero rates (near-zero forecasts look good).
    This block reports timing (occurrence), magnitude (sale days), scale-free error
    (MASE), inverse-frequency weighted MAE, sales/revenue loss, and newsvendor
    costs (see ``inventory_cost_metrics``; primary sales-loss key
    ``sales_revenue_loss_units``).
    """
    y = np.asarray(y, np.float64).reshape(-1)
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    yhat_r = round_forecast(yhat)
    nz = y > 0
    z = ~nz
    n = len(y)
    out = {
        "n_rows": int(n),
        "n_nonzero": int(nz.sum()),
        "zero_rate": float(z.mean()) if n else None,
        # Magnitude / level (weak alone under intermittency)
        "mae_all": float(mean_absolute_error(y, yhat)) if n else None,
        "mae_all_rounded": float(mean_absolute_error(y, yhat_r)) if n else None,
        "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else None,
        "rmse_nonzero": float(np.sqrt(np.mean((y[nz] - yhat[nz]) ** 2))) if nz.any() else None,
        "mean_final": float(yhat.mean()) if n else None,
        "mean_actual": float(y.mean()) if n else None,
        "bias": float(yhat.mean() - y.mean()) if n else None,
        "bias_nonzero": float(yhat[nz].mean() - y[nz].mean()) if nz.any() else None,
        "predict_zero_mae": float(mean_absolute_error(y, np.zeros_like(y))) if n else None,
    }

    # Inverse-frequency weighted MAE: upweight rare sale days (and quiet days symmetrically)
    if n and 0 < nz.mean() < 1.0:
        w = np.where(nz, 1.0 / nz.mean(), 1.0 / z.mean())
        out["iwmae"] = float(np.average(np.abs(y - yhat), weights=w))
        out["iwmae_rounded"] = float(np.average(np.abs(y - yhat_r), weights=w))
    else:
        out["iwmae"] = out["mae_all"]
        out["iwmae_rounded"] = out["mae_all_rounded"]

    # Relative to seasonal naive (train pooled scale when provided)
    if n and mase_scale is not None and mase_scale > 0:
        out["mase"] = float(np.mean(np.abs(y - yhat)) / mase_scale)
        out["mase_rounded"] = float(np.mean(np.abs(y - yhat_r)) / mase_scale)
        if nz.any():
            out["mase_nonzero"] = float(np.mean(np.abs(y[nz] - yhat[nz])) / mase_scale)
    else:
        out["mase"] = None
        out["mase_rounded"] = None
        out["mase_nonzero"] = None

    # Timing / occurrence from rounded forecast (ŷ_r > 0)
    if n and len(np.unique(nz.astype(int))) == 2:
        pred_nz = yhat_r > 0
        pr, rc, f1, _ = precision_recall_fscore_support(
            nz.astype(int), pred_nz.astype(int), average="binary", zero_division=0
        )
        out["occ_precision"] = float(pr)
        out["occ_recall"] = float(rc)
        out["occ_f1"] = float(f1)
        # Service-oriented on sale days: under-forecast fraction / stockout proxy
        out["underforecast_rate_nonzero"] = float(np.mean(yhat[nz] < y[nz]))
        out["hit_nonzero_rate"] = float(np.mean(pred_nz[nz]))  # predicted demand when sale
    else:
        out["occ_precision"] = None
        out["occ_recall"] = None
        out["occ_f1"] = None
        out["underforecast_rate_nonzero"] = None
        out["hit_nonzero_rate"] = None

    # Inventory / planning (always attached; does not change IWMAE primary)
    out.update(inventory_cost_metrics(y, yhat))

    if p is not None and n and len(np.unique(nz.astype(int))) == 2:
        p = np.asarray(p, np.float64).reshape(-1)
        yb = nz.astype(np.float64)
        out["mean_p"] = float(p.mean())
        out["aucroc"] = float(roc_auc_score(yb, p))
        out["aucpr"] = float(average_precision_score(yb, p))
        pred_p = p >= 0.5
        pr, rc, f1, _ = precision_recall_fscore_support(
            nz.astype(int), pred_p.astype(int), average="binary", zero_division=0
        )
        out["p_precision@0.5"] = float(pr)
        out["p_recall@0.5"] = float(rc)
        out["p_f1@0.5"] = float(f1)
    return out

def strata_report(y, yhat, p, skus, volume_map, mase_scale: float | None = None):
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    skus = np.asarray(skus).reshape(-1)
    p = None if p is None else np.asarray(p).reshape(-1)
    bands = np.array([volume_map.get(s, "unk") for s in skus])
    out = {"overall": kpi_block(y, yhat, p, mase_scale=mase_scale)}
    # volume-weighted MAE: weight each row by that SKU's train volume share of total
    # (approx: use band mean volume as proxy per row belonging to band)
    for band in ("low", "mid", "high"):
        m = bands == band
        out[band] = kpi_block(
            y[m], yhat[m], None if p is None else p[m], mase_scale=mase_scale
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
