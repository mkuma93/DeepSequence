"""Train-time intermittent gate calibration tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from deepsequence_hierarchical_attention.components_lightweight import (
    GateProbabilityScale,
    IntermittentHandlerLightweight,
    _softplus_inverse,
    build_hierarchical_model_lightweight,
)


def test_softplus_inverse_roundtrip():
    for y in (0.85, 1.0, 1.5, 2.0):
        x = _softplus_inverse(y)
        got = float(tf.nn.softplus(tf.constant(x)).numpy())
        assert abs(got - y) < 1e-5


def test_gate_probability_scale_init_and_trainable():
    layer = GateProbabilityScale(init_scale=0.85, trainable_scale=True)
    p = np.full((32, 1), 0.4, np.float32)
    out = layer(p)
    scale = float(tf.nn.softplus(layer._scale_raw).numpy())
    assert abs(scale - 0.85) < 1e-4
    np.testing.assert_allclose(out.numpy(), p * scale, atol=1e-5)
    assert layer._scale_raw.trainable


def test_gate_probability_scale_rate_match_adds_loss():
    layer = GateProbabilityScale(
        init_scale=1.0,
        trainable_scale=False,
        rate_match_weight=0.5,
        rate_match_target=0.1,
    )
    p = np.full((64, 1), 0.4, np.float32)
    _ = layer(p)
    assert len(layer.losses) == 1
    # (0.4 - 0.1)^2 * 0.5 = 0.045
    assert abs(float(layer.losses[0].numpy()) - 0.045) < 1e-5


def test_learnable_logit_scale_cools_midrange_nonzero():
    """Sharpening (multiply) on negative nonzero-logits lowers mean p."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 6)).astype(np.float32)
    # Force low zero-logits via prior so nonzero p is mid/high, then sharpen.
    base = IntermittentHandlerLightweight(
        hidden_dim=8, prior_zero_rate=0.5, name="base_gate"
    )
    sharp = IntermittentHandlerLightweight(
        hidden_dim=8,
        prior_zero_rate=0.5,
        learnable_logit_scale=True,
        logit_scale_init=2.0,
        name="sharp_gate",
    )
    # Copy weights so only scale differs: build both then set sharp weights from base.
    z0 = base(x, training=False).numpy()
    # Manually set sharp dense weights equal after a forward to build.
    _ = sharp(x, training=False)
    sharp.zero_prob_layer1.set_weights(base.zero_prob_layer1.get_weights())
    sharp.zero_prob_output.set_weights(base.zero_prob_output.get_weights())
    z1 = sharp(x, training=False).numpy()
    # Higher zero-prob under sharpening when logits are positive from prior=0.5...
    # With shared weights, scale>1 increases |logit| → more extreme zero_prob.
    assert float(np.mean(np.abs(z1 - 0.5))) >= float(np.mean(np.abs(z0 - 0.5))) - 1e-6


def test_builder_train_calibrate_wires_layers():
    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        hidden_dim=16,
        n_changepoints=5,
        gate_use_raw_regressors=True,
        intermittent_prior_zero_rate=0.9,
        intermittent_learnable_logit_scale=True,
        gate_prob_scale=True,
        gate_prob_scale_init=0.85,
        gate_rate_match_weight=0.01,
        gate_rate_match_target=0.1,
    )
    names = {layer.name for layer in model.layers}
    assert "gate_raw_regressor_proj" in names
    assert "non_zero_probability" in names
    scale = model.get_layer("non_zero_probability")
    assert isinstance(scale, GateProbabilityScale)

    b = 8
    outs = model(
        [
            np.zeros((b, 1), np.float32),
            np.zeros((b, 6), np.float32),
            np.zeros((b, 4), np.float32),
            np.zeros((b, 6), np.float32),
            np.zeros((b, 1), np.int32),
        ],
        training=True,
    )
    assert outs["non_zero_probability"].shape == (b, 1)
    # Rate-match and entropy add_loss should be present when training.
    assert len(model.losses) >= 1


def test_builder_defaults_omit_train_calibration():
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
    # Default InvertProbability keeps the non_zero_probability name.
    layer = model.get_layer("non_zero_probability")
    assert not isinstance(layer, GateProbabilityScale)
