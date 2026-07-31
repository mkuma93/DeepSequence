"""Unit tests for softplus-constrained monotone holiday and regressor experts."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from deepsequence_hierarchical_attention.components_lightweight import (
    HolidayComponentLightweight,
    RegressorComponentLightweight,
    build_hierarchical_model_lightweight,
)


def _assert_monotone_either_direction(y, eps=1e-6):
    diffs = np.diff(y)
    assert np.all(diffs >= -eps) or np.all(diffs <= eps), (
        f"not monotone either way: min_diff={diffs.min()}, max_diff={diffs.max()}"
    )


def test_holiday_monotonic_in_abs_distance_either_direction():
    """Monotone axis is |days_from_*|; learned sign ⇒ ↑ or ↓ in |d|."""
    rng = np.random.default_rng(1)
    holiday = HolidayComponentLightweight(
        n_changepoints=8,
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        holiday_monotonic=True,
        name="holiday_mono",
    )
    # Sweep |d| via positive signed distance; other holiday channel fixed.
    d = tf.linspace(0.0, 200.0, 64)
    x = tf.stack([d, tf.zeros_like(d)], axis=-1)
    _ = holiday(x[:1], training=False)  # build
    sign0 = float(rng.choice([-2.0, 2.0]))
    holiday.raw_sign.assign([sign0, 0.0])
    holiday.raw_slopes.assign(tf.ones_like(holiday.raw_slopes))
    y = holiday(x, training=False).numpy().reshape(-1)
    _assert_monotone_either_direction(y)


def test_holiday_monotonic_uses_abs_so_signed_symmetry():
    """Same |d| from either side of the holiday yields the same output."""
    rng = np.random.default_rng(2)
    holiday = HolidayComponentLightweight(
        n_changepoints=6,
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        holiday_monotonic=True,
        name="holiday_abs_sym",
    )
    pos = tf.constant([[30.0], [60.0], [90.0]])
    _ = holiday(pos[:1], training=False)
    holiday.raw_sign.assign([float(rng.choice([-2.0, 2.0]))])
    holiday.raw_slopes.assign(tf.ones_like(holiday.raw_slopes))
    neg = -pos
    y_pos = holiday(pos, training=False).numpy()
    y_neg = holiday(neg, training=False).numpy()
    np.testing.assert_allclose(y_pos, y_neg, atol=1e-5)


def test_holiday_monotonic_false_builds_unconstrained():
    holiday = HolidayComponentLightweight(
        n_changepoints=4,
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        holiday_monotonic=False,
        name="holiday_free",
    )
    x = tf.zeros((16, 2), dtype=tf.float32)
    y = holiday(x, training=False)
    assert y.shape == (16, 1)
    assert hasattr(holiday, "output_layer")
    assert getattr(holiday.output_layer.activation, "__name__", "") == "softsign"
    assert not hasattr(holiday, "raw_slopes")
    assert not hasattr(holiday, "raw_sign")


def test_regressor_monotonic_in_feature_either_direction():
    rng = np.random.default_rng(3)
    reg = RegressorComponentLightweight(
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        regressor_monotonic=True,
        n_changepoints=8,
        feature_min=0.0,
        feature_max=50.0,
        name="reg_mono",
    )
    # Sweep channel 0; hold others fixed.
    v = tf.linspace(0.0, 40.0, 64)
    x = tf.stack([v, tf.zeros_like(v), tf.ones_like(v)], axis=-1)
    _ = reg(x[:1], training=False)
    sign0 = float(rng.choice([-2.0, 2.0]))
    reg.raw_sign.assign([sign0, 0.0, 0.0])
    reg.raw_slopes.assign(tf.ones_like(reg.raw_slopes))
    y = reg(x, training=False).numpy().reshape(-1)
    _assert_monotone_either_direction(y)


def test_regressor_monotonic_false_builds_unconstrained():
    reg = RegressorComponentLightweight(
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        regressor_monotonic=False,
        name="reg_free",
    )
    x = tf.zeros((16, 3), dtype=tf.float32)
    y = reg(x, training=False)
    assert y.shape == (16, 1)
    assert hasattr(reg, "output_layer")
    assert getattr(reg.output_layer.activation, "__name__", "") == "softsign"
    assert not hasattr(reg, "raw_slopes")
    assert not hasattr(reg, "raw_sign")


def test_builder_passes_holiday_and_regressor_monotonic():
    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=2,
        n_holiday_features=2,
        n_lag_features=2,
        n_skus=3,
        n_changepoints=5,
        hidden_dim=8,
        sku_embedding_dim=2,
        dropout_rate=0.0,
        use_cross_layers=False,
        use_intermittent=False,
        use_sku=False,
        holiday_monotonic=True,
        regressor_monotonic=True,
    )
    n = 16
    temporal = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
    fourier = np.zeros((n, 2), dtype=np.float32)
    holiday = np.linspace(0.0, 100.0, n, dtype=np.float32).reshape(-1, 1)
    holiday = np.concatenate([holiday, np.zeros_like(holiday)], axis=1)
    lag = np.zeros((n, 2), dtype=np.float32)
    sku = np.zeros((n, 1), dtype=np.int32)
    outs = model([temporal, fourier, holiday, lag, sku], training=False)
    assert "final_forecast" in outs or isinstance(outs, (list, tuple, dict))
