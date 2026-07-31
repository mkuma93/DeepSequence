"""Tests for hybrid temporal DeepSequence + decoupled gate."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.hybrid_temporal import (
    build_hierarchical_model_hybrid,
)


def _dummy_tabular(b=8, n_t=1, n_f=6, n_h=4, n_l=6, n_skus=4):
    return [
        np.zeros((b, n_t), np.float32),
        np.zeros((b, n_f), np.float32),
        np.zeros((b, n_h), np.float32),
        np.zeros((b, n_l), np.float32),
        np.zeros((b, 1), np.int32),
    ]


def test_paper_defaults_have_no_sequence_input():
    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        hidden_dim=16,
        n_changepoints=5,
    )
    assert len(model.inputs) == 5
    names = {layer.name for layer in model.layers}
    assert "temporal_encoder_mha_0" not in names
    assert "gate_temporal_proj" not in names


def test_hybrid_build_call_shapes():
    lookback, n_ch = 8, 10
    model = build_hierarchical_model_hybrid(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        n_sequence_channels=n_ch,
        lookback=lookback,
        temporal_d_model=16,
        temporal_n_heads=2,
        temporal_n_blocks=1,
        decouple_gate=True,
        hidden_dim=16,
        n_changepoints=5,
    )
    assert len(model.inputs) == 6
    assert model.inputs[-1].shape.as_list() == [None, lookback, n_ch]
    b = 4
    outs = model(
        _dummy_tabular(b) + [np.zeros((b, lookback, n_ch), np.float32)],
        training=False,
    )
    assert set(outs.keys()) >= {
        "final_forecast",
        "non_zero_probability",
        "base_forecast",
    }
    assert outs["final_forecast"].shape == (b, 1)
    assert outs["non_zero_probability"].shape == (b, 1)
    names = {layer.name for layer in model.layers}
    assert "temporal_encoder_mha_0" in names
    assert "gate_raw_regressor_proj" in names
    assert "gate_temporal_proj" in names
    assert "magnitude_temporal_concat" in names


def test_decoupled_gate_omits_base_level_from_handler_inputs():
    """Gate path must not concatenate softplus base / component experts."""
    lookback, n_ch = 6, 8
    model = build_hierarchical_model_hybrid(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        n_sequence_channels=n_ch,
        lookback=lookback,
        decouple_gate=True,
        hidden_dim=16,
        n_changepoints=5,
        use_cross_layers=True,
    )
    names = {layer.name for layer in model.layers}
    assert "gate_raw_regressor_proj" in names
    assert "gate_temporal_proj" in names
    assert "gate_raw_regressor_0" in names
    assert "gate_temporal_0" in names
    # Coupled path feeds experts+base into cross_layer_intermittent;
    # decoupled uses raw/temporal projections into that cross layer instead.
    cross = model.get_layer("cross_layer_intermittent")
    inbound = cross.input
    n_gate_inputs = len(inbound) if isinstance(inbound, (list, tuple)) else None
    # sku + 3 lag proj + 2 temporal proj = 6 (no 4 experts + base)
    assert n_gate_inputs == 6
    outs = model(
        _dummy_tabular(4) + [np.zeros((4, lookback, n_ch), np.float32)],
        training=False,
    )
    assert outs["final_forecast"].shape == (4, 1)


def test_decouple_gate_without_temporal_still_drops_base():
    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        hidden_dim=16,
        n_changepoints=5,
        use_temporal_context=False,
        decouple_gate=True,
        use_cross_layers=True,
    )
    names = {layer.name for layer in model.layers}
    assert "gate_raw_regressor_proj" in names
    assert "gate_temporal_proj" not in names
    cross = model.get_layer("cross_layer_intermittent")
    inbound = cross.input
    n_gate_inputs = len(inbound) if isinstance(inbound, (list, tuple)) else None
    # sku + 3 lag proj
    assert n_gate_inputs == 4
    outs = model(_dummy_tabular(4), training=False)
    assert outs["final_forecast"].shape == (4, 1)


def _assert_dense_bias(model, name, expected):
    layer = model.get_layer(name)
    assert bool(layer.use_bias) is expected, f"{name}.use_bias={layer.use_bias}"


def test_final_heads_keep_bias_intermediates_do_not():
    """Only softplus magnitude / occurrence logit / MH gate offsets use bias."""
    lookback, n_ch = 8, 10
    hybrid = build_hierarchical_model_hybrid(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        n_sequence_channels=n_ch,
        lookback=lookback,
        temporal_d_model=16,
        temporal_n_heads=2,
        temporal_n_blocks=1,
        decouple_gate=True,
        hidden_dim=16,
        n_changepoints=5,
    )
    for name in (
        "magnitude_temporal_hidden",
        "gate_raw_regressor_proj",
        "gate_temporal_proj",
        "component_attention_logits",
        "temporal_encoder_in_proj",
        "temporal_encoder_ff_up_0",
        "temporal_encoder_ff_down_0",
    ):
        _assert_dense_bias(hybrid, name, False)
    _assert_dense_bias(hybrid, "base_forecast", True)
    handler = hybrid.get_layer("intermittent")
    assert handler.zero_prob_layer1.use_bias is False
    assert handler.zero_prob_output.use_bias is True

    mh = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=6,
        n_skus=4,
        hidden_dim=16,
        n_changepoints=5,
        horizon=4,
    )
    _assert_dense_bias(mh, "mh_head_hidden", False)
    _assert_dense_bias(mh, "base_level", True)
    _assert_dense_bias(mh, "base_forecast", True)
    _assert_dense_bias(mh, "mh_gate_offsets", True)
    mh_handler = mh.get_layer("intermittent")
    assert mh_handler.zero_prob_output.use_bias is True
