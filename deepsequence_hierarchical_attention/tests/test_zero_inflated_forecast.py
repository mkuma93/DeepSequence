"""
Smoke tests for zero-inflated personalized forecasting.

Covers the production lightweight path: SKU-conditioned intermittent demand
with non_zero_probability gating and final_forecast = base × P(nonzero).

Feature layout matches feature_config.yaml:
  1 trend (time_index) + 6 cyclical + 3 lags + holiday block.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

pytest.importorskip("tensorflow")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_zero_inflated():
    """Synthetic intermittent demand: ~90% zeros, SKU-specific magnitude."""
    rng = np.random.default_rng(42)
    n_samples, n_skus = 256, 8
    # Match production feature_config: single time index for changepoint ReLU
    n_temporal, n_fourier, n_holiday, n_lag = 1, 6, 10, 3

    sku_id = rng.integers(0, n_skus, size=(n_samples, 1)).astype(np.int32)
    # time_index scaled to [0, 1] to match TrendComponentLightweight defaults
    temporal = rng.uniform(0.0, 1.0, size=(n_samples, n_temporal)).astype(np.float32)
    fourier = rng.normal(size=(n_samples, n_fourier)).astype(np.float32)
    holiday = rng.normal(size=(n_samples, n_holiday)).astype(np.float32)
    lag = rng.normal(size=(n_samples, n_lag)).astype(np.float32)

    # Personalized base rate + zero inflation
    sku_scale = (1.0 + sku_id.astype(np.float32) * 0.5).reshape(-1)
    nonzero = rng.random(n_samples) > 0.90
    y = np.where(
        nonzero,
        rng.lognormal(mean=0.5, sigma=0.4, size=n_samples) * sku_scale,
        0.0,
    )
    y = y.astype(np.float32).reshape(-1, 1)
    y_binary = (y > 0).astype(np.float32)

    return {
        "n_skus": n_skus,
        "n_temporal": n_temporal,
        "n_fourier": n_fourier,
        "n_holiday": n_holiday,
        "n_lag": n_lag,
        "x": [temporal, fourier, holiday, lag, sku_id],
        "y": y,
        "y_binary": y_binary,
        "zero_rate": float(np.mean(y == 0)),
        "time_min": float(temporal.min()),
        "time_max": float(temporal.max()),
    }


def _build_lightweight(d, **kwargs):
    from deepsequence_hierarchical_attention import build_hierarchical_model_lightweight

    defaults = dict(
        n_temporal_features=d["n_temporal"],
        n_fourier_features=d["n_fourier"],
        n_holiday_features=d["n_holiday"],
        n_lag_features=d["n_lag"],
        n_skus=d["n_skus"],
        hidden_dim=16,
        sku_embedding_dim=4,
        dropout_rate=0.1,
        use_cross_layers=True,
        use_intermittent=True,
        n_changepoints=10,
    )
    defaults.update(kwargs)
    return build_hierarchical_model_lightweight(**defaults)


def test_lightweight_outputs_zero_inflated_heads(synthetic_zero_inflated):
    d = synthetic_zero_inflated
    model = _build_lightweight(d)

    assert set(model.output_names) >= {
        "final_forecast",
        "non_zero_probability",
        "base_forecast",
    }

    outs = model(d["x"], training=False)
    p = outs["non_zero_probability"].numpy()
    base = outs["base_forecast"].numpy()
    final = outs["final_forecast"].numpy()

    assert p.shape == d["y"].shape
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert np.all(base >= 0.0)
    assert np.all(final >= 0.0)
    # Soft gate: final ≈ base × P(nonzero)
    np.testing.assert_allclose(final, base * p, rtol=1e-5, atol=1e-5)


def test_compile_helper_matches_nonzero_probability_head(synthetic_zero_inflated):
    from deepsequence_hierarchical_attention import compile_hierarchical_model

    d = synthetic_zero_inflated
    model = _build_lightweight(d)
    compile_hierarchical_model(model, learning_rate=1e-3)
    model.train_on_batch(
        d["x"],
        {
            "final_forecast": d["y"],
            "base_forecast": d["y"],
            "non_zero_probability": d["y_binary"],
        },
    )


def test_create_model_from_features_compiles_for_intermittent(synthetic_zero_inflated):
    from deepsequence_hierarchical_attention.components_lightweight import (
        create_model_from_features,
    )

    d = synthetic_zero_inflated
    X = np.concatenate(d["x"][:4], axis=1)
    sku = d["x"][4]
    feature_indices = {
        "trend": list(range(0, d["n_temporal"])),
        "seasonal": list(range(d["n_temporal"], d["n_temporal"] + d["n_fourier"])),
        "holiday": list(
            range(
                d["n_temporal"] + d["n_fourier"],
                d["n_temporal"] + d["n_fourier"] + d["n_holiday"],
            )
        ),
        "regressor": list(
            range(
                d["n_temporal"] + d["n_fourier"] + d["n_holiday"],
                d["n_temporal"] + d["n_fourier"] + d["n_holiday"] + d["n_lag"],
            )
        ),
    }

    model, split_fn = create_model_from_features(
        X_train=X,
        sku_train=sku,
        feature_indices=feature_indices,
        n_skus=d["n_skus"],
        hidden_dim=16,
        sku_embedding_dim=4,
        y_train=d["y"].reshape(-1),
        zero_rate=d["zero_rate"],
    )
    xb = split_fn(X[:32], sku[:32])
    model.train_on_batch(
        xb,
        {
            "final_forecast": d["y"][:32],
            "base_forecast": d["y"][:32],
            "non_zero_probability": d["y_binary"][:32],
        },
    )


def test_adaptive_train_step_clips_gradients_not_loss(synthetic_zero_inflated):
    """Spiky batches must still produce finite non-zero gradient updates."""
    examples_dir = PACKAGE_ROOT / "examples"
    sys.path.insert(0, str(examples_dir))
    spec = importlib.util.spec_from_file_location(
        "train_lightweight_adaptive_loss",
        examples_dir / "train_lightweight_adaptive_loss.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    d = synthetic_zero_inflated
    base = _build_lightweight(d)
    bce_fn = mod.WeightedBCELoss(weight_nonzero=9.0, weight_zero=1.0)
    mae_fn = tf.keras.losses.MeanAbsoluteError()
    wrapped = mod.AdaptiveWeightedModel(
        base_model=base,
        bce_loss_fn=bce_fn,
        mae_loss_fn=mae_fn,
        use_fixed_weights=True,
        bce_weight=1.0,
        mae_weight=1.0,
        zero_rate=d["zero_rate"],
        avg_nonzero_demand=float(np.mean(d["y"][d["y"] > 0])),
        pos_weight=9.0,
    )
    wrapped.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    # Inflate a few targets to create a large MAE batch (old code would zero grads)
    y_spike = d["y"].copy()
    y_spike[:8] = 1e4
    y_bin = (y_spike > 0).astype(np.float32)

    before = [w.numpy().copy() for w in base.trainable_weights[:3]]
    wrapped.train_on_batch(
        d["x"],
        {
            "base_forecast": y_spike,
            "non_zero_binary": y_bin,
            "non_zero_probability": y_bin,
            "final_forecast": y_spike,
        },
    )
    after = [w.numpy() for w in base.trainable_weights[:3]]
    changed = any(not np.allclose(a, b) for a, b in zip(before, after))
    assert changed, "expected parameter update on spiked intermittent batch"


def test_create_hierarchical_model_builds_without_typeerror():
    from deepsequence_hierarchical_attention import create_hierarchical_model

    main, *_ = create_hierarchical_model(
        num_skus=4,
        n_features=10,
        trend_feature_indices=[0, 1],
        seasonal_feature_indices=[2, 3, 4],
        holiday_feature_indices=[5, 6],
        regressor_feature_indices=[7, 8, 9],
        time_min=0.0,
        time_max=100.0,
    )
    assert "zero_probability" in main.output_names or "final_forecast" in main.output_names


def test_sku_personalization_changes_forecast(synthetic_zero_inflated):
    """Same features, different SKU ids should change personalized outputs."""
    d = synthetic_zero_inflated
    model = _build_lightweight(d)

    temporal, fourier, holiday, lag, _ = d["x"]
    sku_a = np.zeros((32, 1), dtype=np.int32)
    sku_b = np.full((32, 1), 7, dtype=np.int32)
    x_common = [temporal[:32], fourier[:32], holiday[:32], lag[:32]]

    out_a = model(x_common + [sku_a], training=False)
    out_b = model(x_common + [sku_b], training=False)

    assert not np.allclose(
        out_a["base_forecast"].numpy(),
        out_b["base_forecast"].numpy(),
    ) or not np.allclose(
        out_a["non_zero_probability"].numpy(),
        out_b["non_zero_probability"].numpy(),
    )
