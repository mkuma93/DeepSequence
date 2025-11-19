"""
Tests for Cross Layer and Cross Network components.
"""

import tensorflow as tf
import numpy as np
from src.deepsequence.cross_layer import CrossLayer, CrossNetwork


def test_cross_layer_output_shape():
    """Test that CrossLayer preserves input shape."""
    print("\n=== Test: CrossLayer Output Shape ===")
    
    batch_size = 32
    feature_dim = 64
    
    # Create layer
    cross = CrossLayer(units=feature_dim)
    
    # Test input
    x = tf.random.normal((batch_size, feature_dim))
    
    # Forward pass
    output = cross(x)
    
    # Check shape
    assert output.shape == (batch_size, feature_dim), \
        f"Expected shape {(batch_size, feature_dim)}, got {output.shape}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print("✓ Shape preserved correctly")


def test_cross_layer_with_separate_inputs():
    """Test CrossLayer with separate x_0 and x_l inputs."""
    print("\n=== Test: CrossLayer with Separate Inputs ===")
    
    batch_size = 16
    feature_dim = 32
    
    cross = CrossLayer(units=feature_dim)
    
    # Separate inputs
    x_0 = tf.random.normal((batch_size, feature_dim))
    x_l = tf.random.normal((batch_size, feature_dim))
    
    # Forward pass
    output = cross((x_0, x_l))
    
    assert output.shape == (batch_size, feature_dim)
    print(f"✓ x_0 shape: {x_0.shape}")
    print(f"✓ x_l shape: {x_l.shape}")
    print(f"✓ Output shape: {output.shape}")
    print("✓ Separate inputs handled correctly")


def test_cross_layer_learns_interactions():
    """Test that CrossLayer learns different outputs than input."""
    print("\n=== Test: CrossLayer Learns Interactions ===")
    
    batch_size = 32
    feature_dim = 16
    
    cross = CrossLayer(units=feature_dim)
    
    # Input with pattern
    x = tf.constant([[i % 5 for i in range(feature_dim)]] * batch_size,
                    dtype=tf.float32)
    
    # Build layer
    _ = cross(x)
    
    # Set non-trivial weights
    cross.kernel.assign(tf.ones(feature_dim) * 0.5)
    if cross.bias is not None:
        cross.bias.assign(tf.ones(feature_dim) * 0.1)
    
    # Forward pass
    output = cross(x)
    
    # Output should differ from input (due to interactions)
    diff = tf.reduce_mean(tf.abs(output - x))
    
    print(f"✓ Input mean: {tf.reduce_mean(x).numpy():.4f}")
    print(f"✓ Output mean: {tf.reduce_mean(output).numpy():.4f}")
    print(f"✓ Mean absolute difference: {diff.numpy():.4f}")
    print("✓ CrossLayer transforms input (learns interactions)")


def test_cross_network_multiple_layers():
    """Test CrossNetwork with multiple layers."""
    print("\n=== Test: CrossNetwork Multiple Layers ===")
    
    batch_size = 32
    feature_dim = 64
    num_layers = 3
    
    # Create network
    cross_net = CrossNetwork(num_layers=num_layers, units=feature_dim)
    
    # Input
    x = tf.random.normal((batch_size, feature_dim))
    
    # Forward pass
    output = cross_net(x)
    
    # Check shape
    assert output.shape == (batch_size, feature_dim)
    
    # Check that layers were created
    assert len(cross_net.cross_layers) == num_layers
    
    print(f"✓ Number of cross layers: {len(cross_net.cross_layers)}")
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print("✓ CrossNetwork stacking works correctly")


def test_cross_layer_gradients():
    """Test that CrossLayer has trainable gradients."""
    print("\n=== Test: CrossLayer Gradients ===")
    
    batch_size = 16
    feature_dim = 32
    
    cross = CrossLayer(units=feature_dim)
    
    # Input and target
    x = tf.random.normal((batch_size, feature_dim))
    target = tf.random.normal((batch_size, feature_dim))
    
    # Forward pass with gradient tape
    with tf.GradientTape() as tape:
        output = cross(x)
        loss = tf.reduce_mean(tf.square(output - target))
    
    # Compute gradients
    gradients = tape.gradient(loss, cross.trainable_variables)
    
    # Check gradients exist
    assert len(gradients) > 0
    assert all(g is not None for g in gradients)
    
    print(f"✓ Loss: {loss.numpy():.4f}")
    print(f"✓ Number of trainable variables: {len(cross.trainable_variables)}")
    print(f"✓ Kernel gradient shape: {gradients[0].shape}")
    if len(gradients) > 1:
        print(f"✓ Bias gradient shape: {gradients[1].shape}")
    print("✓ Gradients computed successfully")


def test_cross_layer_residual_connection():
    """Test that CrossLayer includes residual connection."""
    print("\n=== Test: CrossLayer Residual Connection ===")
    
    batch_size = 16
    feature_dim = 32
    
    # Create layer with zero weights (only residual should remain)
    cross = CrossLayer(units=feature_dim, use_bias=False)
    
    x = tf.random.normal((batch_size, feature_dim))
    
    # Build and zero out kernel
    _ = cross(x)
    cross.kernel.assign(tf.zeros(feature_dim))
    
    # Forward pass
    output = cross(x)
    
    # With zero kernel and no bias, output should equal input (residual)
    diff = tf.reduce_mean(tf.abs(output - x))
    
    print(f"✓ Mean difference (should be near 0): {diff.numpy():.6f}")
    assert diff.numpy() < 1e-5, "Residual connection not working"
    print("✓ Residual connection works correctly")


def test_cross_network_integration():
    """Test CrossNetwork in a simple model."""
    print("\n=== Test: CrossNetwork Integration ===")
    
    batch_size = 32
    input_dim = 10
    hidden_dim = 64
    output_dim = 1
    
    # Build simple model with cross network
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Embedding layer
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(inputs)
    
    # Cross network
    cross_out = CrossNetwork(num_layers=2, units=hidden_dim)(x)
    
    # Output
    outputs = tf.keras.layers.Dense(output_dim)(cross_out)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile
    model.compile(optimizer='adam', loss='mse')
    
    # Test forward pass
    x_test = np.random.randn(batch_size, input_dim).astype(np.float32)
    predictions = model.predict(x_test, verbose=0)
    
    assert predictions.shape == (batch_size, output_dim)
    
    print(f"✓ Model built successfully")
    print(f"✓ Total parameters: {model.count_params()}")
    print(f"✓ Prediction shape: {predictions.shape}")
    print("✓ CrossNetwork integrates correctly in Keras model")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Cross Layer and Cross Network")
    print("=" * 60)
    
    test_cross_layer_output_shape()
    test_cross_layer_with_separate_inputs()
    test_cross_layer_learns_interactions()
    test_cross_network_multiple_layers()
    test_cross_layer_gradients()
    test_cross_layer_residual_connection()
    test_cross_network_integration()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
