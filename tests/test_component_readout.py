"""Tests for interpretable component readout (trend/seasonal/holiday/regressor + p/b)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")


@pytest.fixture
def tiny_model_and_batch():
    from deepsequence_hierarchical_attention import build_hierarchical_model_lightweight

    n = 16
    rng = np.random.default_rng(0)
    x = [
        rng.uniform(0, 1, (n, 1)).astype(np.float32),
        rng.normal(size=(n, 6)).astype(np.float32),
        rng.uniform(0, 30, (n, 4)).astype(np.float32),
        rng.normal(size=(n, 6)).astype(np.float32),
        rng.integers(0, 4, (n, 1)).astype(np.int32),
    ]
    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        hidden_dim=8,
        sku_embedding_dim=4,
        dropout_rate=0.0,
        use_cross_layers=False,
        use_intermittent=True,
        n_changepoints=5,
        horizon=1,
    )
    return model, x


def test_component_readout_keys_and_shapes(tiny_model_and_batch):
    from deepsequence_hierarchical_attention import (
        build_component_readout_model,
        predict_with_components,
    )

    model, x = tiny_model_and_batch
    probe = build_component_readout_model(model)
    n = x[0].shape[0]
    for name in ("trend", "seasonal", "holiday", "regressor", "component_alpha"):
        assert name in probe.output_names or name in getattr(probe, "output", {})

    out = predict_with_components(model, x, batch_size=8, verbose=0)
    for name in ("trend", "seasonal", "holiday", "regressor"):
        assert out[name].shape[0] == n
        assert out[f"mixed_contribution_{name}"].shape[0] == n
    assert out["component_alpha"].shape == (n, 4)
    assert out["final_forecast"].shape[0] == n
    assert out["base_forecast"].shape[0] == n
    assert out["non_zero_probability"].shape[0] == n
    # Gate identity: yhat ≈ p * b
    yhat = out["final_forecast"].reshape(-1)
    p = out["non_zero_probability"].reshape(-1)
    b = out["base_forecast"].reshape(-1)
    np.testing.assert_allclose(yhat, p * b, rtol=1e-5, atol=1e-5)


def test_component_readout_matches_model_heads(tiny_model_and_batch):
    from deepsequence_hierarchical_attention import predict_with_components

    model, x = tiny_model_and_batch
    pred = model.predict(x, verbose=0)
    out = predict_with_components(model, x, verbose=0)
    np.testing.assert_allclose(
        out["final_forecast"].reshape(-1),
        np.asarray(pred["final_forecast"]).reshape(-1),
        rtol=1e-5,
        atol=1e-5,
    )
