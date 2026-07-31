"""Tests for intermittent gate calibration and residual transformer reclaim path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from deepsequence_hierarchical_attention.components_lightweight import (
    IntermittentHandlerLightweight,
    _logit_probability,
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.residual_transformer import (
    build_residual_transformer,
    mask_predict_step,
)


def test_logit_probability_roundtrip():
    for p in (0.1, 0.5, 0.9):
        z = _logit_probability(p)
        assert abs(1.0 / (1.0 + np.exp(-z)) - p) < 1e-6


def test_gate_prior_bias_shifts_mean_probability():
    """Prior toward high zero rate should raise initial P(zero) vs default."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 6)).astype(np.float32)

    default = IntermittentHandlerLightweight(hidden_dim=8, name="gate_default")
    prior = IntermittentHandlerLightweight(
        hidden_dim=8, prior_zero_rate=0.9, name="gate_prior"
    )
    p0 = default(x, training=False).numpy().mean()
    p1 = prior(x, training=False).numpy().mean()
    # prior_zero_rate=0.9 → higher mean zero-prob before training
    assert p1 > p0
    assert p1 > 0.7


def test_gate_temperature_softens_probability():
    handler = IntermittentHandlerLightweight(
        hidden_dim=8, temperature=2.0, name="gate_temp"
    )
    # Force extreme logits via a constant bias-ish input after build
    x = np.full((64, 4), 5.0, np.float32)
    p = handler(x, training=False).numpy()
    assert np.all((p > 0.0) & (p < 1.0))


def test_builder_gate_raw_regressors_opt_in():
    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        hidden_dim=16,
        n_changepoints=5,
        gate_use_raw_regressors=True,
        intermittent_prior_zero_rate=0.85,
    )
    names = {layer.name for layer in model.layers}
    assert "gate_raw_regressor_proj" in names
    assert "gate_raw_regressor_0" in names

    b = 8
    outs = model(
        [
            np.zeros((b, 1), np.float32),
            np.zeros((b, 6), np.float32),
            np.zeros((b, 4), np.float32),
            np.zeros((b, 6), np.float32),
            np.zeros((b, 1), np.int32),
        ],
        training=False,
    )
    assert outs["non_zero_probability"].shape[-1] == 1


def test_builder_defaults_skip_raw_regressor_gate():
    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        hidden_dim=16,
        n_changepoints=5,
    )
    names = {layer.name for layer in model.layers}
    assert "gate_raw_regressor_proj" not in names


def test_residual_preserves_ds_gate_and_zero_delta_start():
    lookback, n_channels, n_skus = 8, 4, 3
    model = build_residual_transformer(
        lookback=lookback,
        n_channels=n_channels,
        n_skus=n_skus,
        preserve_ds_gate=True,
        encoder_gate_mix=0.0,
    )
    rng = np.random.default_rng(1)
    hist = rng.normal(size=(16, lookback, n_channels)).astype(np.float32)
    hist = mask_predict_step(hist)
    # Channel 3 = p_ds
    p_ds = np.clip(rng.uniform(0.05, 0.4, size=(16, lookback)), 0.05, 0.4).astype(
        np.float32
    )
    hist[..., 3] = p_ds
    y_struct = rng.uniform(0.5, 3.0, size=(16, 1)).astype(np.float32)
    sku = rng.integers(0, n_skus, size=(16, 1)).astype(np.int32)

    out = model([hist, sku, y_struct], training=False)
    p = out["non_zero_probability"].numpy().reshape(-1)
    delta = out["delta"].numpy().reshape(-1)
    final = out["final_forecast"].numpy().reshape(-1)
    base = out["base_forecast"].numpy().reshape(-1)

    np.testing.assert_allclose(p, p_ds[:, -1], atol=1e-5)
    np.testing.assert_allclose(delta, 0.0, atol=1e-5)
    np.testing.assert_allclose(base, np.maximum(y_struct.reshape(-1), 0.0), atol=1e-5)
    np.testing.assert_allclose(final, base * p, atol=1e-5)


def test_calibrate_probability_temperature_helper():
    from eval_helpers import (
        apply_probability_temperature,
        calibrate_probability_temperature,
    )

    rng = np.random.default_rng(2)
    y = (rng.random(200) > 0.9).astype(np.float64) * rng.lognormal(0.5, 0.3, 200)
    p = np.clip(rng.uniform(0.1, 0.5, 200), 1e-3, 1 - 1e-3)
    base = np.where(y > 0, y * 1.2, 2.0)
    yhat = base * p
    best = calibrate_probability_temperature(y, yhat, p)
    assert "temperature" in best
    y2, p2 = apply_probability_temperature(yhat, p, temperature=best["temperature"])
    assert y2.shape == yhat.shape
    assert p2.shape == p.shape
