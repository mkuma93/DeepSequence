"""Behavioral checks for the component-attention / gate / activation fixes.

Run against either code variant; prints the observable behavior so the two
arms can be compared directly.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepsequence_hierarchical_attention.components_lightweight import (  # noqa: E402
    build_hierarchical_model_lightweight,
)

N_TREND, N_FOURIER, N_HOLIDAY, N_LAG, N_SKUS, BATCH = 1, 4, 0, 6, 20, 64


def make_inputs():
    rng = np.random.default_rng(0)
    return [
        rng.random((BATCH, N_TREND), np.float32),
        rng.random((BATCH, max(N_FOURIER, 1)), np.float32),
        np.zeros((BATCH, 1), np.float32),
        rng.random((BATCH, N_LAG), np.float32),
        rng.integers(0, N_SKUS, (BATCH, 1)).astype(np.int32),
    ]


def build(horizon, output_activation="linear"):
    return build_hierarchical_model_lightweight(
        n_temporal_features=N_TREND,
        n_fourier_features=N_FOURIER,
        n_holiday_features=N_HOLIDAY,
        n_lag_features=N_LAG,
        n_skus=N_SKUS,
        hidden_dim=16,
        sku_embedding_dim=4,
        dropout_rate=0.1,
        horizon=horizon,
        output_activation=output_activation,
    )


print("=" * 72)
model = build(horizon=1)
inputs = make_inputs()

# 1. Component attention mass on the disabled holiday slot.
softmax_layer = model.get_layer("component_attention_softmax")
probe = tf.keras.Model(model.inputs, softmax_layer.output)
weights = np.asarray(probe(inputs, training=False))
names = ["trend", "seasonal", "holiday(disabled)", "regressor"]
print("1) mean component attention weight (holiday is disabled here):")
for name, value in zip(names, weights.mean(axis=0)):
    print(f"     {name:20s} {value:.6f}")
print(f"   mass on disabled holiday slot = {weights.mean(axis=0)[2]:.6f}")

# 2. Multi-horizon gate: does the intermittent handler reach the output?
mh = build(horizon=6)
mh_inputs = make_inputs()
handler_vars = [
    v for v in mh.trainable_variables if "intermittent" in v.name.lower()
]
with tf.GradientTape() as tape:
    out = mh(mh_inputs, training=True)
    target = tf.reduce_sum(out["non_zero_probability"])
grads = tape.gradient(target, handler_vars)
connected = sum(1 for g in grads if g is not None)
print("\n2) multi-horizon occurrence gate:")
print(f"     intermittent handler trainable vars      = {len(handler_vars)}")
print(f"     vars with gradient from non_zero_prob    = {connected}")
print(f"     gate output shape                        = {tuple(out['non_zero_probability'].shape)}")

# 3. Seasonal component honors output_activation.
seasonal = build(horizon=1, output_activation="sparse_amplify").get_layer("seasonal")
activation = getattr(seasonal.output_layer.activation, "__name__", str(seasonal.output_layer.activation))
print("\n3) seasonal output activation when builder is given 'sparse_amplify':")
print(f"     seasonal.output_layer.activation = {activation}")

# 4. Self-cancelling trainable entropy scale.
scales = [v.name for v in model.trainable_variables if "entropy_scale" in v.name]
print("\n4) trainable entropy-scale weights (each can zero out its own penalty):")
print(f"     count = {len(scales)}")
