"""Context-aware component mixer: regime signals reweight experts per SKU."""

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
        # Keep experts independent of lag so mixer context is the only lag path.
        enable_regressor=False,
    )
    defaults.update(kwargs)
    return build_hierarchical_model_lightweight(**defaults)


def _fixed_batch(n=16, seed=0):
    rng = np.random.default_rng(seed)
    temporal = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    fourier = rng.normal(size=(n, 6)).astype(np.float32)
    holiday = rng.normal(size=(n, 4)).astype(np.float32)
    sku_id = np.zeros((n, 1), dtype=np.int32)  # same SKU for all rows
    return temporal, fourier, holiday, sku_id


def _attention_probe(model):
    return tf.keras.Model(
        model.inputs, model.get_layer("component_attention_softmax").output
    )


def test_context_changes_component_attention_weights():
    """Same SKU + same experts, different lag/intermittent context → different mix."""
    tf.keras.utils.set_random_seed(7)
    model = _build_model(context_aware_component_mixer=True)
    assert "component_mixer_context" in {layer.name for layer in model.layers}
    assert "component_mixer_source" in {layer.name for layer in model.layers}

    temporal, fourier, holiday, sku_id = _fixed_batch()
    lag_a = np.zeros((16, 3), dtype=np.float32)
    lag_b = np.ones((16, 3), dtype=np.float32) * 5.0

    probe = _attention_probe(model)
    w_a = probe([temporal, fourier, holiday, lag_a, sku_id], training=False).numpy()
    w_b = probe([temporal, fourier, holiday, lag_b, sku_id], training=False).numpy()

    # Expert outputs are lag-independent (regressor off); only mixer context differs.
    assert w_a.shape == (16, 4)
    assert w_b.shape == (16, 4)
    np.testing.assert_allclose(w_a.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5)
    assert not np.allclose(w_a, w_b, rtol=1e-5, atol=1e-5)


def test_flag_false_restores_sku_only_mixer():
    """Legacy flag: attention ignores lag; same SKU ⇒ identical weights."""
    tf.keras.utils.set_random_seed(11)
    model = _build_model(context_aware_component_mixer=False)
    layer_names = {layer.name for layer in model.layers}
    assert "component_mixer_context" not in layer_names
    assert "component_mixer_source" not in layer_names

    temporal, fourier, holiday, sku_id = _fixed_batch(seed=3)
    lag_a = np.zeros((16, 3), dtype=np.float32)
    lag_b = np.full((16, 3), -3.0, dtype=np.float32)

    probe = _attention_probe(model)
    w_a = probe([temporal, fourier, holiday, lag_a, sku_id], training=False).numpy()
    w_b = probe([temporal, fourier, holiday, lag_b, sku_id], training=False).numpy()

    np.testing.assert_allclose(w_a, w_b, rtol=1e-5, atol=1e-5)


def test_context_only_when_sku_off():
    """SKU off + context mixer: lag still drives attention (no SKU concat)."""
    tf.keras.utils.set_random_seed(19)
    model = _build_model(
        context_aware_component_mixer=True,
        use_sku=False,
    )
    layer_names = {layer.name for layer in model.layers}
    assert "component_mixer_context" in layer_names
    assert "component_mixer_source" not in layer_names

    temporal, fourier, holiday, sku_id = _fixed_batch(seed=5)
    lag_a = np.zeros((16, 3), dtype=np.float32)
    lag_b = np.full((16, 3), 2.5, dtype=np.float32)

    probe = _attention_probe(model)
    w_a = probe([temporal, fourier, holiday, lag_a, sku_id], training=False).numpy()
    w_b = probe([temporal, fourier, holiday, lag_b, sku_id], training=False).numpy()
    assert not np.allclose(w_a, w_b, rtol=1e-5, atol=1e-5)
