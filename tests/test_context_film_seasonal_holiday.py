"""Context FiLM on seasonal + holiday: lag modulates calendar experts only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

pytest.importorskip("tensorflow")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _build_model(**kwargs):
    from deepsequence_hierarchical_attention import (
        build_hierarchical_model_lightweight,
    )

    defaults = dict(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=3,
        n_skus=4,
        hidden_dim=16,
        sku_embedding_dim=4,
        dropout_rate=0.0,
        use_cross_layers=False,
        use_intermittent=True,
        n_changepoints=8,
        # Isolate lag → FiLM: no lag path through regressor or mixer.
        enable_regressor=False,
        context_aware_component_mixer=False,
    )
    defaults.update(kwargs)
    return build_hierarchical_model_lightweight(**defaults)


def _fixed_batch(n=16, seed=0):
    rng = np.random.default_rng(seed)
    temporal = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    fourier = rng.normal(size=(n, 6)).astype(np.float32)
    holiday = rng.normal(size=(n, 4)).astype(np.float32)
    sku_id = np.zeros((n, 1), dtype=np.int32)
    return temporal, fourier, holiday, sku_id


def _probe(model, layer_name):
    return tf.keras.Model(model.inputs, model.get_layer(layer_name).output)


def _set_nonzero_film_weights(model, scale=0.5, shift=0.25):
    """Break zeros-init identity so lag context can move post-FiLM scalars."""
    for name in (
        "context_film_seasonal_scale",
        "context_film_holiday_scale",
        "context_film_seasonal_shift",
        "context_film_holiday_shift",
    ):
        layer = model.get_layer(name)
        w = layer.get_weights()
        assert len(w) == 1
        filled = np.full_like(w[0], scale if "scale" in name else shift)
        layer.set_weights([filled])
    # Non-zero proj so different lag → different context embedding.
    proj = model.get_layer("context_film_proj")
    pw = proj.get_weights()
    rng = np.random.default_rng(123)
    proj.set_weights([rng.normal(scale=0.3, size=pw[0].shape).astype(np.float32)])


def test_lag_context_modulates_seasonal_and_holiday():
    """Same calendar experts, different lag → different post-FiLM seasonal/holiday."""
    tf.keras.utils.set_random_seed(21)
    model = _build_model(context_film_seasonal_holiday=True)
    names = {layer.name for layer in model.layers}
    assert "context_film_proj" in names
    assert "context_film_seasonal_add" in names
    assert "context_film_holiday_add" in names

    _set_nonzero_film_weights(model)

    temporal, fourier, holiday, sku_id = _fixed_batch()
    lag_a = np.zeros((16, 3), dtype=np.float32)
    lag_b = np.full((16, 3), 4.0, dtype=np.float32)

    pre_s = _probe(model, "seasonal")
    pre_h = _probe(model, "holiday")
    post_s = _probe(model, "context_film_seasonal_add")
    post_h = _probe(model, "context_film_holiday_add")
    trend_probe = _probe(model, "trend")

    inputs_a = [temporal, fourier, holiday, lag_a, sku_id]
    inputs_b = [temporal, fourier, holiday, lag_b, sku_id]

    # Pre-FiLM experts ignore lag (regressor off).
    np.testing.assert_allclose(
        pre_s(inputs_a, training=False).numpy(),
        pre_s(inputs_b, training=False).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pre_h(inputs_a, training=False).numpy(),
        pre_h(inputs_b, training=False).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    # Trend unchanged by lag.
    np.testing.assert_allclose(
        trend_probe(inputs_a, training=False).numpy(),
        trend_probe(inputs_b, training=False).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    s_a = post_s(inputs_a, training=False).numpy()
    s_b = post_s(inputs_b, training=False).numpy()
    h_a = post_h(inputs_a, training=False).numpy()
    h_b = post_h(inputs_b, training=False).numpy()
    assert not np.allclose(s_a, s_b, rtol=1e-5, atol=1e-5)
    assert not np.allclose(h_a, h_b, rtol=1e-5, atol=1e-5)


def test_flag_false_is_identity_no_film_layers():
    """Legacy flag: no FiLM layers; lag does not change seasonal/holiday."""
    tf.keras.utils.set_random_seed(33)
    model = _build_model(context_film_seasonal_holiday=False)
    names = {layer.name for layer in model.layers}
    assert "context_film_proj" not in names
    assert "context_film_seasonal_add" not in names
    assert "context_film_holiday_add" not in names

    temporal, fourier, holiday, sku_id = _fixed_batch(seed=4)
    lag_a = np.zeros((16, 3), dtype=np.float32)
    lag_b = np.full((16, 3), -2.5, dtype=np.float32)

    s_probe = _probe(model, "seasonal")
    h_probe = _probe(model, "holiday")
    inputs_a = [temporal, fourier, holiday, lag_a, sku_id]
    inputs_b = [temporal, fourier, holiday, lag_b, sku_id]

    np.testing.assert_allclose(
        s_probe(inputs_a, training=False).numpy(),
        s_probe(inputs_b, training=False).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        h_probe(inputs_a, training=False).numpy(),
        h_probe(inputs_b, training=False).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_zeros_init_film_is_near_identity():
    """Untrained FiLM (zeros kernels) leaves seasonal/holiday ≈ pre-FiLM."""
    tf.keras.utils.set_random_seed(41)
    model = _build_model(context_film_seasonal_holiday=True)
    temporal, fourier, holiday, sku_id = _fixed_batch(seed=8)
    lag = np.ones((16, 3), dtype=np.float32) * 3.0
    inputs = [temporal, fourier, holiday, lag, sku_id]

    pre_s = _probe(model, "seasonal")(inputs, training=False).numpy()
    post_s = _probe(model, "context_film_seasonal_add")(
        inputs, training=False
    ).numpy()
    pre_h = _probe(model, "holiday")(inputs, training=False).numpy()
    post_h = _probe(model, "context_film_holiday_add")(
        inputs, training=False
    ).numpy()
    np.testing.assert_allclose(pre_s, post_s, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pre_h, post_h, rtol=1e-5, atol=1e-5)
