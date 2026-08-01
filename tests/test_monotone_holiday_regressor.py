"""Unit tests for softplus-constrained monotone holiday and regressor experts.

Mono ⊕ selection attention (not XOR): each channel map is monotone; attention
selects among channels. Full expert output is not claimed mono in one channel
when attention weights move with that channel.
"""

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


def test_holiday_monotonic_single_holiday_in_abs_distance():
    """Single holiday ⇒ trivial attention; expert stays mono in |d|."""
    rng = np.random.default_rng(1)
    holiday = HolidayComponentLightweight(
        n_changepoints=8,
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        holiday_monotonic=True,
        name="holiday_mono_single",
    )
    d = tf.linspace(0.0, 200.0, 64)
    x = tf.reshape(d, (-1, 1))
    _ = holiday(x[:1], training=False)
    assert hasattr(holiday, "selection_logits")
    assert hasattr(holiday, "selection_softmax")
    holiday.raw_sign.assign([float(rng.choice([-2.0, 2.0]))])
    holiday.raw_slopes.assign(tf.ones_like(holiday.raw_slopes))
    y = holiday(x, training=False).numpy().reshape(-1)
    _assert_monotone_either_direction(y)


def test_holiday_mono_channel_maps_are_monotone_in_abs_distance():
    """Per-holiday softplus-PWL maps (pre-attention) are mono in |d|."""
    rng = np.random.default_rng(11)
    holiday = HolidayComponentLightweight(
        n_changepoints=8,
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        holiday_monotonic=True,
        name="holiday_mono_channels",
    )
    d = tf.linspace(0.0, 200.0, 64)
    x = tf.stack([d, tf.zeros_like(d)], axis=-1)
    _ = holiday(x[:1], training=False)
    sign0 = float(rng.choice([-2.0, 2.0]))
    holiday.raw_sign.assign([sign0, 0.0])
    holiday.raw_slopes.assign(tf.ones_like(holiday.raw_slopes))
    channels = holiday._mono_channel_scalars(x).numpy()
    _assert_monotone_either_direction(channels[:, 0])


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


def test_holiday_selection_attention_mass_moves_with_distance():
    """Different holiday distances change selection attention mass."""
    holiday = HolidayComponentLightweight(
        n_changepoints=6,
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        holiday_monotonic=True,
        attention_temperature=1.0,
        name="holiday_attn_mass",
    )
    # Two holidays; identity logits so softmax tracks mono channel values.
    _ = holiday(tf.zeros((1, 2), dtype=tf.float32), training=False)
    holiday.raw_sign.assign([1.5, 1.5])
    holiday.raw_slopes.assign(0.25 * tf.ones_like(holiday.raw_slopes))
    holiday.selection_logits.kernel.assign(tf.eye(2, dtype=tf.float32))

    near = tf.constant([[5.0, 80.0]], dtype=tf.float32)
    far = tf.constant([[120.0, 80.0]], dtype=tf.float32)
    w_near = holiday.selection_softmax(
        holiday.selection_logits(holiday._mono_channel_scalars(near))
    ).numpy()
    w_far = holiday.selection_softmax(
        holiday.selection_logits(holiday._mono_channel_scalars(far))
    ).numpy()
    assert w_near.shape == (1, 2)
    assert np.allclose(w_near.sum(axis=-1), 1.0, atol=1e-5)
    assert not np.allclose(w_near, w_far, atol=1e-4)


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
    assert not hasattr(holiday, "selection_softmax")


def test_regressor_monotonic_builds_with_lag_attention():
    """Mono path keeps MaskedEntropyAttention over softplus-PWL lag scalars."""
    reg = RegressorComponentLightweight(
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        regressor_monotonic=True,
        n_changepoints=8,
        feature_min=0.0,
        feature_max=50.0,
        name="reg_mono_attn",
    )
    x = tf.zeros((16, 3), dtype=tf.float32)
    y = reg(x, training=False)
    assert y.shape == (16, 1)
    assert hasattr(reg, "attention")
    assert hasattr(reg, "raw_slopes")
    assert hasattr(reg, "output_layer")
    assert "attention" in reg.attention.name
    # Dense/mish after attention is not claimed end-to-end mono in one lag.


def test_regressor_mono_channel_maps_are_monotone():
    """Per-lag softplus-PWL maps (pre-attention) are mono in that lag."""
    rng = np.random.default_rng(13)
    reg = RegressorComponentLightweight(
        hidden_dim=8,
        dropout_rate=0.0,
        use_sku_shift_scale=False,
        regressor_monotonic=True,
        n_changepoints=8,
        feature_min=0.0,
        feature_max=50.0,
        name="reg_mono_channels",
    )
    v = tf.linspace(0.0, 40.0, 64)
    x = tf.stack([v, tf.zeros_like(v), tf.ones_like(v)], axis=-1)
    _ = reg(x[:1], training=False)
    sign0 = float(rng.choice([-2.0, 2.0]))
    reg.raw_sign.assign([sign0, 0.0, 0.0])
    reg.raw_slopes.assign(tf.ones_like(reg.raw_slopes))
    channels = reg._mono_channel_scalars(x).numpy()
    _assert_monotone_either_direction(channels[:, 0])


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
    assert hasattr(reg, "attention")
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
    found_holiday_sel = False
    found_reg_attn = False
    for obj in model._flatten_layers():
        if isinstance(obj, HolidayComponentLightweight):
            found_holiday_sel = hasattr(obj, "selection_softmax")
        if isinstance(obj, RegressorComponentLightweight):
            found_reg_attn = hasattr(obj, "attention")
    assert found_holiday_sel, "holiday mono path missing selection attention"
    assert found_reg_attn, "regressor mono path missing lag attention"
