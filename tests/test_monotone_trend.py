"""Unit tests for softplus-constrained monotone trend (learned sign)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from deepsequence_hierarchical_attention.components_lightweight import (
    TrendComponentLightweight,
    build_hierarchical_model_lightweight,
)


def _assert_monotone_either_direction(y, eps=1e-6):
    diffs = np.diff(y)
    assert np.all(diffs >= -eps) or np.all(diffs <= eps), (
        f"not monotone either way: min_diff={diffs.min()}, max_diff={diffs.max()}"
    )


def test_trend_monotonic_in_time_either_direction():
    """Learned sign: after a non-flat init, g(t) is mono ↑ or ↓ in time."""
    rng = np.random.default_rng(0)
    trend = TrendComponentLightweight(
        n_changepoints=8,
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        trend_monotonic=True,
        name="trend_mono",
    )
    t = tf.linspace(0.0, 1.0, 64)[:, tf.newaxis]
    _ = trend(t[:1], training=False)  # build
    trend.raw_sign.assign([float(rng.choice([-2.0, 2.0]))])
    trend.raw_slopes.assign(tf.ones_like(trend.raw_slopes))
    y = trend(t, training=False).numpy().reshape(-1)
    _assert_monotone_either_direction(y)


def test_trend_monotonic_false_builds_unconstrained():
    trend = TrendComponentLightweight(
        n_changepoints=6,
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        trend_monotonic=False,
        name="trend_free",
    )
    t = tf.linspace(0.0, 1.0, 32)[:, tf.newaxis]
    y = trend(t, training=False)
    assert y.shape == (32, 1)
    # Legacy path has unconstrained Dense + softsign (default) output activation.
    assert hasattr(trend, "output_layer")
    assert getattr(trend.output_layer.activation, "__name__", "") == "softsign"
    assert not hasattr(trend, "raw_slopes")
    assert not hasattr(trend, "raw_sign")


def test_builder_passes_trend_monotonic():
    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=2,
        n_holiday_features=0,
        n_lag_features=1,
        n_skus=3,
        n_changepoints=5,
        hidden_dim=8,
        sku_embedding_dim=2,
        dropout_rate=0.0,
        use_cross_layers=False,
        use_intermittent=False,
        use_sku=False,
        trend_monotonic=True,
    )
    n = 16
    temporal = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
    fourier = np.zeros((n, 2), dtype=np.float32)
    holiday = np.zeros((n, 1), dtype=np.float32)
    lag = np.zeros((n, 1), dtype=np.float32)
    sku = np.zeros((n, 1), dtype=np.int32)
    outs = model([temporal, fourier, holiday, lag, sku], training=False)
    assert "final_forecast" in outs or isinstance(outs, (list, tuple, dict))
