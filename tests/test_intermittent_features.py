"""Leakage and parity tests for causal intermittent regressor features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepsequence_hierarchical_attention.intermittent_features import (
    CausalInferenceFeatureServer,
    build_states_from_history,
    transform_panel,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_sku_panel():
    # One SKU, known demand path
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    qty = np.array([0, 5, 0, 0, 3, 0, 0, 0, 2, 0], dtype=float)
    return pd.DataFrame(
        {
            "id_var": ["SKU_A"] * len(dates),
            "ds": dates,
            "Quantity": qty,
        }
    )


def test_no_same_day_leakage():
    df = _synthetic_sku_panel()
    feats, _ = transform_panel(df, return_states=True)

    # On day with sale=5 (index 1), features must not see that 5 yet
    assert feats.loc[1, "lag_1"] == 0.0
    assert feats.loc[1, "last_sale_quantity"] == 0.0
    assert feats.loc[1, "lifetime_cumsum"] == 0.0
    assert feats.loc[1, "days_since_last_sale"] == -1.0

    # Next day sees the sale
    assert feats.loc[2, "lag_1"] == 5.0
    assert feats.loc[2, "last_sale_quantity"] == 5.0
    assert feats.loc[2, "lifetime_cumsum"] == 5.0
    assert feats.loc[2, "days_since_last_sale"] == 1.0


def test_future_rows_do_not_affect_past_features():
    df = _synthetic_sku_panel()
    feats_full, _ = transform_panel(df, return_states=True)

    # Corrupt future quantities and recompute prefix — past features unchanged
    df2 = df.copy()
    df2.loc[5:, "Quantity"] = 999.0
    feats_prefix, _ = transform_panel(df2.iloc[:5], return_states=True)

    for col in [
        "lag_1",
        "lag_2",
        "lag_7",
        "days_since_last_sale",
        "last_sale_quantity",
        "lifetime_cumsum",
    ]:
        np.testing.assert_allclose(
            feats_full.loc[:4, col].to_numpy(),
            feats_prefix[col].to_numpy(),
        )


def test_val_warm_start_from_train_states():
    df = _synthetic_sku_panel()
    train = df.iloc[:6].copy()
    val = df.iloc[6:].copy()

    _, states = transform_panel(train, return_states=True)
    feats_val, _ = transform_panel(val, prior_states=states, return_states=True)

    # First val row is 2020-01-07 (index 6 in original); last sale was day index 4 qty=3
    # days since = 2
    assert feats_val.iloc[0]["last_sale_quantity"] == 3.0
    assert feats_val.iloc[0]["days_since_last_sale"] == 2.0
    assert feats_val.iloc[0]["lifetime_cumsum"] == 8.0  # 5+3


def test_inference_server_matches_batch_transform():
    df = _synthetic_sku_panel()
    feats_batch, states = transform_panel(df, return_states=True)

    # Rebuild via server: warm empty, observe day by day AFTER scoring
    server = CausalInferenceFeatureServer(lags=[1, 2, 7])
    rows = []
    for _, row in df.iterrows():
        rows.append(server.features_for(row["id_var"], row["ds"]))
        server.observe(row["id_var"], row["ds"], row["Quantity"])
    feats_serve = pd.DataFrame(rows)

    for col in feats_batch.columns:
        np.testing.assert_allclose(feats_batch[col].to_numpy(), feats_serve[col].to_numpy())

    # End state parity
    end = build_states_from_history(df)
    assert end["SKU_A"].lifetime_cumsum == states["SKU_A"].lifetime_cumsum
    assert end["SKU_A"].last_sale_quantity == states["SKU_A"].last_sale_quantity


def test_series_local_rate_features_causal():
    df = _synthetic_sku_panel()
    feats, _ = transform_panel(
        df,
        intermittent_names=[
            "days_since_last_sale",
            "last_sale_quantity",
            "lifetime_cumsum",
            "rolling_nonzero_rate",
            "rolling_mean_size",
            "age_normalized_cumsum",
        ],
        rate_window=4,
        return_states=True,
    )
    # Before any obs
    assert feats.loc[0, "rolling_nonzero_rate"] == 0.0
    assert feats.loc[0, "age_normalized_cumsum"] == 0.0
    # After first sale of 5 at index 1, index 2 sees rate 0.5 over [0,5]
    assert feats.loc[2, "rolling_nonzero_rate"] == 0.5
    assert feats.loc[2, "rolling_mean_size"] == 5.0
    assert feats.loc[2, "age_normalized_cumsum"] == 2.5  # 5/2
    # Same-day: index 1 must not include its own sale in rate stats
    assert feats.loc[1, "rolling_nonzero_rate"] == 0.0
    assert feats.loc[1, "age_normalized_cumsum"] == 0.0


def test_feature_config_create_features_causal():
    import importlib.util

    from deepsequence_hierarchical_attention.data.feature_config_loader import load_feature_config
    cfg = load_feature_config()

    df = _synthetic_sku_panel()
    # Dummy holiday distances (15 cols) aligned to df
    holiday_cols = cfg.holiday_names
    holidays = pd.DataFrame(
        {c: np.arange(len(df), dtype=float) for c in holiday_cols}
    )
    X, states = cfg.create_features(df, holidays, return_states=True)
    assert list(X.columns) == cfg.feature_names
    assert X.shape[1] == 28
    assert "lag_14" not in X.columns
    assert not any(c.startswith("is_") for c in X.columns)
    # Same-day sale not in lag_1
    assert X.loc[1, "lag_1"] == 0.0
    assert X.loc[2, "lag_1"] == 5.0
    assert "SKU_A" in states
