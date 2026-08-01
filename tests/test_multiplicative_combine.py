"""Multiplicative Level-2 combine: shape, positivity, and formula checks."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

pytest.importorskip("tensorflow")

from deepsequence_hierarchical_attention.components_lightweight import (
    MultiplicativeComponentCombine,
    build_hierarchical_model_lightweight,
)


def _tiny_kwargs(**overrides):
    base = dict(
        n_temporal_features=1,
        n_fourier_features=4,
        n_holiday_features=2,
        n_lag_features=3,
        n_skus=3,
        hidden_dim=8,
        sku_embedding_dim=2,
        dropout_rate=0.0,
        use_intermittent=True,
        use_sku=True,
        n_changepoints=4,
        orthogonality_weight=0.0,
        use_cross_layers=False,
    )
    base.update(overrides)
    return base


def test_multiplicative_layer_matches_closed_form():
    eps = 1e-3
    layer = MultiplicativeComponentCombine(
        component_flags=[True, True, True, True],
        eps=eps,
    )
    stacked = tf.constant(
        [
            [0.5, 0.2, -0.4, 0.1],
            [-1.0, 0.9, 0.0, -0.8],
        ],
        dtype=tf.float32,
    )
    weights = tf.constant(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=tf.float32,
    )
    out = layer([stacked, weights]).numpy()
    assert out.shape == (2, 1)

    for i in range(2):
        t, s, h, r = stacked[i].numpy()
        _, a_s, a_h, a_r = weights[i].numpy()
        expected = float(np.log1p(np.exp(t)))  # softplus
        for a, e in ((a_s, s), (a_h, h), (a_r, r)):
            expected *= max(eps, 1.0 + a * e)
        np.testing.assert_allclose(out[i, 0], expected, rtol=1e-5, atol=1e-5)

    assert np.all(out > 0.0)
    assert np.all(np.isfinite(out))


def test_multiplicative_skips_inactive_experts():
    layer = MultiplicativeComponentCombine(
        component_flags=[True, False, True, False],
        eps=1e-3,
    )
    stacked = tf.constant([[0.0, 0.99, -0.5, 0.99]], dtype=tf.float32)
    weights = tf.constant([[0.25, 0.25, 0.25, 0.25]], dtype=tf.float32)
    out = float(layer([stacked, weights]).numpy()[0, 0])
    # softplus(0) * max(eps, 1 + 0.25*(-0.5)); seasonal/regressor skipped
    expected = float(np.log1p(np.exp(0.0))) * max(1e-3, 1.0 + 0.25 * (-0.5))
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_multiplicative_eps_floor_avoids_zero_collapse():
    layer = MultiplicativeComponentCombine(
        component_flags=[False, True, True, True],
        eps=1e-3,
    )
    # Extreme negative softsign-like experts with α≈1 would otherwise ≈0.
    stacked = tf.constant([[0.0, -0.999, -0.999, -0.999]], dtype=tf.float32)
    weights = tf.constant([[0.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
    out = float(layer([stacked, weights]).numpy()[0, 0])
    assert out >= 1e-3 - 1e-9
    assert np.isfinite(out)


def test_builder_additive_default_vs_multiplicative_layers():
    add = build_hierarchical_model_lightweight(**_tiny_kwargs())
    mul = build_hierarchical_model_lightweight(
        **_tiny_kwargs(component_combine="multiplicative")
    )
    add_names = {layer.name for layer in add.layers}
    mul_names = {layer.name for layer in mul.layers}
    assert "sum_weighted_components" in add_names
    assert "multiplicative_component_combine" not in add_names
    assert "multiplicative_component_combine" in mul_names
    assert "sum_weighted_components" not in mul_names
    # Shared Level-2 attention scaffolding remains.
    assert "component_attention_softmax" in mul_names
    assert "print_attention_weights" in mul_names
    assert "component_mixer_context" in mul_names


def test_multiplicative_forward_shape_and_finite():
    tf.keras.utils.set_random_seed(0)
    model = build_hierarchical_model_lightweight(
        **_tiny_kwargs(component_combine="multiplicative")
    )
    n = 8
    rng = np.random.default_rng(0)
    inputs = [
        rng.uniform(0, 1, size=(n, 1)).astype(np.float32),
        rng.normal(size=(n, 4)).astype(np.float32),
        rng.normal(size=(n, 2)).astype(np.float32),
        rng.normal(size=(n, 3)).astype(np.float32),
        np.zeros((n, 1), dtype=np.int32),
    ]
    out = model(inputs, training=False)
    yhat = np.asarray(out["final_forecast"])
    base = np.asarray(out["base_forecast"])
    assert yhat.shape == (n, 1)
    assert base.shape == (n, 1)
    assert np.all(np.isfinite(yhat))
    assert np.all(np.isfinite(base))
    assert np.all(base >= 0.0)


def test_invalid_component_combine_raises():
    with pytest.raises(ValueError, match="component_combine"):
        build_hierarchical_model_lightweight(
            **_tiny_kwargs(component_combine="xor")
        )
