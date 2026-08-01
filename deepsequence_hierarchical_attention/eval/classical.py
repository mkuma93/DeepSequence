"""Classical intermittent baselines: Croston, SBA, TSB."""

from __future__ import annotations

import numpy as np


def croston_variants(y: np.ndarray, alpha: float = 0.1):
    """
    Fit Croston / SBA / TSB on history ``y`` (1-d, non-negative).

    Returns dict of next-step mean demand forecasts (scalars).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return {"croston": 0.0, "sba": 0.0, "tsb": 0.0}

    # Initialize
    z = float(y[0]) if y[0] > 0 else 1.0  # demand size
    p = 1.0  # inter-demand interval
    q = 1.0  # periods since last demand (for TSB)
    p_hat = 1.0
    z_hat = z
    first = True
    tsb_p = float((y > 0).mean()) if (y > 0).any() else 0.0
    tsb_z = float(y[y > 0].mean()) if (y > 0).any() else 0.0

    for i, val in enumerate(y):
        if val > 0:
            if first:
                z_hat = val
                p_hat = float(i + 1) if i > 0 else 1.0
                first = False
            else:
                z_hat = z_hat + alpha * (val - z_hat)
                p_hat = p_hat + alpha * (q - p_hat)
            q = 1.0
            # TSB updates
            tsb_p = tsb_p + alpha * (1.0 - tsb_p)
            tsb_z = tsb_z + alpha * (val - tsb_z)
        else:
            q += 1.0
            tsb_p = tsb_p + alpha * (0.0 - tsb_p)

    croston = z_hat / max(p_hat, 1e-6)
    sba = croston * (1.0 - alpha / 2.0)  # Syntetos-Boylan approximation
    tsb = tsb_p * tsb_z
    return {
        "croston": float(max(croston, 0.0)),
        "sba": float(max(sba, 0.0)),
        "tsb": float(max(tsb, 0.0)),
    }


def predict_classical_on_panel(
    train_df,
    val_df,
    test_df,
    alpha: float = 0.1,
):
    """
    For each test row, fit Croston/SBA/TSB on all history strictly before ``ds``
    (train+val+earlier test within the series). Returns dict name -> yhat array
    aligned to test_df row order.
    """
    import pandas as pd

    hist = pd.concat([train_df, val_df, test_df], ignore_index=True)
    hist = hist.sort_values(["id_var", "ds"], kind="mergesort")
    hist["id_var"] = hist["id_var"].astype(str)

    # Precompute cumulative lists per sku
    series = {}
    for sku, g in hist.groupby("id_var", sort=False):
        series[str(sku)] = {
            "ds": pd.to_datetime(g["ds"]).to_numpy(),
            "y": g["Quantity"].to_numpy(np.float64),
        }

    out = {k: np.zeros(len(test_df), np.float32) for k in ("croston", "sba", "tsb")}
    test = test_df.reset_index(drop=True)
    for i, row in test.iterrows():
        sku = str(row["id_var"])
        ds = pd.Timestamp(row["ds"])
        s = series[sku]
        # history strictly before this date
        mask = s["ds"] < np.datetime64(ds)
        y_hist = s["y"][mask]
        preds = croston_variants(y_hist, alpha=alpha)
        for k in out:
            out[k][i] = preds[k]
    return out
