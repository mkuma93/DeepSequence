"""Per-SKU zero-rate estimation, fail-fast loss wiring, and SKU gate priors."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from deepsequence_hierarchical_attention.components_lightweight import (
    IntermittentHandlerLightweight,
    _logit_probability,
    bce_sample_weights_from_sku_zero_rates,
    build_hierarchical_model_lightweight,
    create_model_from_features,
    estimate_zero_rate_by_sku,
    multioutput_bce_sample_weight_dict,
    pos_weight_from_zero_rate,
)


def test_create_model_from_features_requires_zero_rate_or_y_train():
    X = np.zeros((8, 10), dtype=np.float32)
    sku = np.zeros((8, 1), dtype=np.int32)
    feature_indices = {
        "trend": [0],
        "seasonal": [1, 2, 3],
        "holiday": [4, 5, 6],
        "regressor": [7, 8, 9],
    }
    with pytest.raises(ValueError, match="zero_rate is required"):
        create_model_from_features(
            X_train=X,
            sku_train=sku,
            feature_indices=feature_indices,
            n_skus=2,
            hidden_dim=8,
            sku_embedding_dim=2,
            zero_rate=None,
            y_train=None,
        )


def test_estimate_zero_rate_by_sku_dense_vs_sparse():
    # SKU 0: denser sales (20% zeros); SKU 1: sparse (90% zeros)
    y_dense = np.array([0.0] * 2 + [1.0] * 8, dtype=np.float32)  # 0.2
    y_sparse = np.array([0.0] * 9 + [1.0] * 1, dtype=np.float32)  # 0.9
    y = np.concatenate([y_dense, y_sparse])
    sku = np.concatenate(
        [np.zeros(len(y_dense), dtype=np.int32), np.ones(len(y_sparse), dtype=np.int32)]
    )
    info = estimate_zero_rate_by_sku(y, sku, n_skus=3, min_obs=1)
    assert info["rates"][0] == pytest.approx(0.2, abs=1e-6)
    assert info["rates"][1] == pytest.approx(0.9, abs=1e-6)
    # Unseen SKU 2 → panel mean fallback
    assert info["rates"][2] == pytest.approx(info["panel_mean"], abs=1e-6)
    assert info["rates"][0] < info["rates"][1]
    assert info["panel_mean"] == pytest.approx(float(np.mean(y == 0)), abs=1e-6)


def test_bce_sample_weights_higher_for_high_zero_rate_sku():
    y = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # all nonzero → pos weights
    sku = np.array([0, 0, 1, 1], dtype=np.int32)
    rates = np.array([0.2, 0.9], dtype=np.float32)
    w = bce_sample_weights_from_sku_zero_rates(y, sku, rates, cap=20.0)
    assert w[0] == pytest.approx(pos_weight_from_zero_rate(0.2), abs=1e-5)
    assert w[2] == pytest.approx(pos_weight_from_zero_rate(0.9), abs=1e-5)
    assert w[2] > w[0]
    # Zero rows keep weight_zero
    yz = np.array([0.0, 1.0], dtype=np.float32)
    skz = np.array([1, 1], dtype=np.int32)
    wz = bce_sample_weights_from_sku_zero_rates(yz, skz, rates, weight_zero=1.0)
    assert wz[0] == pytest.approx(1.0)
    assert wz[1] == pytest.approx(pos_weight_from_zero_rate(0.9), abs=1e-5)


def test_bce_sample_weights_relative_to_panel_avoids_double_count():
    y = np.array([1.0, 1.0], dtype=np.float32)
    sku = np.array([0, 1], dtype=np.int32)
    rates = np.array([0.5, 0.9], dtype=np.float32)
    panel = 0.5
    w = bce_sample_weights_from_sku_zero_rates(
        y, sku, rates, reference_zero_rate=panel, cap=20.0
    )
    # SKU0 matches panel → relative 1.0; SKU1 higher → relative > 1
    assert w[0] == pytest.approx(1.0, abs=1e-5)
    assert w[1] == pytest.approx(
        pos_weight_from_zero_rate(0.9) / pos_weight_from_zero_rate(0.5), abs=1e-5
    )


def test_create_model_from_features_attaches_fit_sample_weights():
    rng = np.random.default_rng(0)
    n = 64
    X = rng.normal(size=(n, 10)).astype(np.float32)
    sku = rng.integers(0, 3, size=(n, 1), dtype=np.int32)
    y = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.7, 0.2, 0.1]).astype(np.float32)
    feature_indices = {
        "trend": [0],
        "seasonal": [1, 2, 3],
        "holiday": [4, 5, 6],
        "regressor": [7, 8, 9],
    }
    model, split_fn = create_model_from_features(
        X_train=X,
        sku_train=sku,
        feature_indices=feature_indices,
        n_skus=3,
        hidden_dim=8,
        sku_embedding_dim=2,
        y_train=y,
    )
    assert model.sku_zero_rates is not None
    assert hasattr(model, "make_fit_sample_weights")
    sw = model.make_fit_sample_weights(y, sku)
    assert "non_zero_probability" in sw
    assert sw["non_zero_probability"].shape[0] == n
    # High-zero SKU nonzero rows should get higher relative weight than low-zero
    rates = model.sku_zero_rates
    dense_sku = int(np.argmin(rates))
    sparse_sku = int(np.argmax(rates))
    nz = y > 0
    dense_w = sw["non_zero_probability"][(sku.reshape(-1) == dense_sku) & nz]
    sparse_w = sw["non_zero_probability"][(sku.reshape(-1) == sparse_sku) & nz]
    if len(dense_w) and len(sparse_w):
        assert float(sparse_w.mean()) >= float(dense_w.mean()) - 1e-5

    xb = split_fn(X[:16], sku[:16])
    model.train_on_batch(
        xb,
        {
            "final_forecast": y[:16].reshape(-1, 1),
            "base_forecast": y[:16].reshape(-1, 1),
            "non_zero_probability": (y[:16] > 0).astype(np.float32).reshape(-1, 1),
        },
        sample_weight=multioutput_bce_sample_weight_dict(
            y[:16],
            sku[:16],
            model.sku_zero_rates,
            reference_zero_rate=model.panel_zero_rate,
            other_keys=("final_forecast",),
        ),
    )


def test_adaptive_model_uses_sku_pos_weights():
    """In-graph SKU lookup: sparse SKU nonzero rows get higher BCE weight."""
    import sys

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples_dir))
    spec = importlib.util.spec_from_file_location(
        "train_lightweight_adaptive_loss",
        examples_dir / "train_lightweight_adaptive_loss.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rates = np.array([0.2, 0.9], dtype=np.float32)
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
        zero_rate=0.5,
        avg_nonzero_demand=1.0,
        pos_weight=1.0,
        sku_zero_rates=rates,
        loss_recipe="three_term",
        use_fixed_weights=True,
    )
    # Nonzero targets; SKU 0 vs SKU 1
    y = np.ones((4, 1), dtype=np.float32)
    sku0 = np.zeros((4, 1), dtype=np.int32)
    sku1 = np.ones((4, 1), dtype=np.int32)
    common = [
        np.zeros((4, 1), np.float32),
        np.zeros((4, 6), np.float32),
        np.zeros((4, 4), np.float32),
        np.zeros((4, 3), np.float32),
    ]
    # Build once so layers exist
    _ = wrapped(common + [sku0], training=False)
    outs = wrapped.base_model(common + [sku0], training=False)
    # Force identical p so only pos_weight differs
    p = tf.constant(0.5, shape=(4, 1), dtype=tf.float32)
    base_f = outs["base_forecast"]
    final_f = outs["final_forecast"]
    _, bce0, _, _, _ = wrapped._compute_task_losses(
        y, final_f, p, base_f, sku_ids=sku0
    )
    _, bce1, _, _, _ = wrapped._compute_task_losses(
        y, final_f, p, base_f, sku_ids=sku1
    )
    assert float(bce1.numpy()) > float(bce0.numpy())


def test_gate_prior_builds_with_sku_zero_rates():
    rates = np.array([0.2, 0.9, 0.5, 0.95], dtype=np.float32)
    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=6,
        n_holiday_features=4,
        n_lag_features=3,
        n_skus=4,
        hidden_dim=8,
        sku_embedding_dim=2,
        n_changepoints=5,
        use_sku=True,
        intermittent_prior_zero_rate=float(rates.mean()),
        intermittent_prior_zero_rates=rates,
    )
    handler = model.get_layer("intermittent")
    assert handler.prior_zero_rates is not None
    assert handler.sku_prior_embedding is not None
    assert not handler.sku_prior_embedding.trainable

    b = 16
    sku_a = np.zeros((b, 1), dtype=np.int32)
    sku_b = np.ones((b, 1), dtype=np.int32)
    common = [
        np.zeros((b, 1), np.float32),
        np.zeros((b, 6), np.float32),
        np.zeros((b, 4), np.float32),
        np.zeros((b, 3), np.float32),
    ]
    out_a = model(common + [sku_a], training=False)
    out_b = model(common + [sku_b], training=False)
    # Higher zero-rate prior → lower initial non-zero probability
    assert out_a["non_zero_probability"].numpy().mean() > out_b[
        "non_zero_probability"
    ].numpy().mean()


def test_handler_sku_prior_logit_matches_embedding():
    rates = np.array([0.1, 0.9], dtype=np.float32)
    handler = IntermittentHandlerLightweight(
        hidden_dim=4,
        prior_zero_rates=rates,
        n_skus=2,
        name="sku_prior_unit",
    )
    feats = np.zeros((4, 3), dtype=np.float32)
    sku = np.array([[0], [0], [1], [1]], dtype=np.int32)
    _ = handler([feats, sku], training=False)
    emb = handler.sku_prior_embedding.get_weights()[0].reshape(-1)
    np.testing.assert_allclose(
        emb,
        [_logit_probability(0.1), _logit_probability(0.9)],
        rtol=1e-5,
        atol=1e-5,
    )


def test_pos_weight_from_zero_rate_caps():
    assert pos_weight_from_zero_rate(0.5) == pytest.approx(1.0)
    assert pos_weight_from_zero_rate(0.95, cap=20.0) == pytest.approx(19.0)
    assert pos_weight_from_zero_rate(0.99, cap=20.0) == pytest.approx(20.0)
