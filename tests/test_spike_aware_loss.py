"""Unit tests for spike-aware loss masking / positive BCE weights."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

pytest.importorskip("tensorflow")

from deepsequence_hierarchical_attention.losses import (
    resolve_positive_bce_weight,
    spike_aware_loss_config,
    spike_aware_magnitude_loss,
    focal_weighted_bce_loss,
    three_term_loss_config,
)


def test_resolve_positive_bce_weight_boost_and_override():
    # zr=0.9 → base 9; boost 2 → 18 (under cap 20)
    assert resolve_positive_bce_weight(0.9, None, boost=2.0, cap=20.0) == pytest.approx(18.0)
    assert resolve_positive_bce_weight(0.9, None, boost=1.0, cap=20.0) == pytest.approx(9.0)
    assert resolve_positive_bce_weight(0.99, None, boost=2.0, cap=20.0) == pytest.approx(20.0)
    assert resolve_positive_bce_weight(0.5, 7.5, boost=2.0, cap=20.0) == pytest.approx(7.5)


def test_spike_aware_config_knobs_and_bakeoff_unchanged():
    cfg = spike_aware_loss_config(
        0.9,
        alpha_bce=1.0,
        w_mag=1.0,
        zero_mag_weight=0.05,
        w_gated=0.0,
        positive_bce_boost=2.0,
        focal_gamma=0.0,
    )
    assert cfg["recipe"] == "spike_aware"
    assert cfg["meta"]["positive_bce_weight"] == pytest.approx(18.0)
    assert cfg["meta"]["zero_mag_weight"] == pytest.approx(0.05)
    assert cfg["weights"]["non_zero_probability"] == pytest.approx(1.0)
    assert cfg["weights"]["base_forecast"] == pytest.approx(1.0)
    # Tiny final weight when timing off (keeps head tracked)
    assert cfg["weights"]["final_forecast"] < 1e-3

    # Default bake-off recipe still three_term
    tt = three_term_loss_config(0.9, alpha_bce=0.2, w_gated=1.0, w_mag=1.0)
    assert tt["recipe"] == "three_term"
    assert tt["weights"]["final_forecast"] == pytest.approx(1.0)


def test_magnitude_masks_zeros_when_zero_weight_off():
    loss_fn = spike_aware_magnitude_loss(zero_mag_weight=0.0, use_mse=False)
    y = tf.constant([[0.0], [0.0], [5.0], [10.0]], dtype=tf.float32)
    # Predictions: huge error on zeros, perfect on positives
    b = tf.constant([[100.0], [100.0], [5.0], [10.0]], dtype=tf.float32)
    val = float(loss_fn(y, b).numpy())
    assert val == pytest.approx(0.0, abs=1e-5)


def test_magnitude_zero_weight_pulls_b_on_quiet_days():
    loss_fn = spike_aware_magnitude_loss(zero_mag_weight=1.0, use_mse=False)
    y = tf.constant([[0.0], [0.0], [4.0], [4.0]], dtype=tf.float32)
    b_bad_zero = tf.constant([[8.0], [8.0], [4.0], [4.0]], dtype=tf.float32)
    b_good_zero = tf.constant([[0.0], [0.0], [4.0], [4.0]], dtype=tf.float32)
    assert float(loss_fn(y, b_bad_zero).numpy()) > float(loss_fn(y, b_good_zero).numpy())


def test_focal_bce_positive_weight_higher_on_sale_days():
    loss_fn = focal_weighted_bce_loss(pos_weight=9.0, gamma=0.0)
    y_pos = tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32)
    y_neg = tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
    p = tf.constant([0.5, 0.5, 0.5, 0.5], dtype=tf.float32)
    # Per-sample BCE: pos gets 9x the -log(p) term
    pos_mean = float(tf.reduce_mean(loss_fn(y_pos, p)).numpy())
    neg_mean = float(tf.reduce_mean(loss_fn(y_neg, p)).numpy())
    assert pos_mean == pytest.approx(9.0 * neg_mean, rel=1e-4)


def test_adaptive_spike_aware_masks_and_weights_positives():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples_dir))
    spec = importlib.util.spec_from_file_location(
        "train_lightweight_adaptive_loss",
        examples_dir / "train_lightweight_adaptive_loss.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from deepsequence_hierarchical_attention import build_hierarchical_model_lightweight

    base = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=3,
        n_skus=2,
        hidden_dim=8,
        sku_embedding_dim=2,
        n_changepoints=5,
        use_sku=True,
    )
    wrapped = mod.AdaptiveWeightedModel(
        base_model=base,
        bce_loss_fn=mod.WeightedBCELoss(weight_nonzero=1.0, weight_zero=1.0),
        mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        zero_rate=0.9,
        avg_nonzero_demand=5.0,
        pos_weight=9.0,
        loss_recipe="spike_aware",
        alpha_bce=1.0,
        w_gated=0.0,
        w_mag=1.0,
        zero_mag_weight=0.0,
        positive_bce_weight=12.0,
        focal_gamma=0.0,
        use_fixed_weights=True,
    )
    assert wrapped.pos_weight == pytest.approx(12.0)

    y = np.array([[0.0], [0.0], [5.0], [5.0]], dtype=np.float32)
    sku = np.zeros((4, 1), dtype=np.int32)
    common = [
        np.zeros((4, 1), np.float32),
        np.zeros((4, 6), np.float32),
        np.zeros((4, 4), np.float32),
        np.zeros((4, 3), np.float32),
    ]
    _ = wrapped(common + [sku], training=False)
    p = tf.constant(0.5, shape=(4, 1), dtype=tf.float32)
    # b perfect on sales, wildly wrong on zeros — with zero_mag_weight=0 mag ignores zeros
    b = tf.constant([[99.0], [99.0], [5.0], [5.0]], dtype=tf.float32)
    yhat = p * b
    total, bce, _, mag, y_nz = wrapped._compute_task_losses(
        y, yhat, p, b, sku_ids=sku
    )
    assert float(mag.numpy()) == pytest.approx(0.0, abs=1e-5)
    assert float(tf.reduce_sum(y_nz).numpy()) == pytest.approx(2.0)
    # Positive BCE weight 12 vs unit negative → mean > unweighted
    assert float(bce.numpy()) > 0.5  # -0.5*log(0.5)*avg with heavy pos
    assert float(total.numpy()) >= float(bce.numpy()) - 1e-5
