"""
Unit Normalization Layer for DeepSequence.

Unit normalization (L2 normalization) constrains the output of each layer
to lie on a unit hypersphere. This provides several benefits:
- Stabilizes training by preventing activation explosion
- Improves gradient flow
- Acts as implicit regularization
- Particularly effective for embeddings and attention mechanisms

Reference:
Salimans, T., & Kingma, D. P. (2016). Weight normalization: A simple 
reparameterization to accelerate training of deep neural networks.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class UnitNorm(layers.Layer):
    """
    Unit normalization layer (L2 normalization).
    
    Normalizes the input to have unit L2 norm along the specified axis.
    Output satisfies: ||output||_2 = 1
    
    This is particularly useful for:
    - Embedding layers (normalized embeddings)
    - Attention mechanisms (stabilized attention weights)
    - Dense layer outputs (bounded activations)
    - Final layer outputs (controlled magnitude)
    """
    
    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-7,
        name: str = "unit_norm",
        **kwargs
    ):
        """
        Initialize unit normalization layer.
        
        Args:
            axis: Axis along which to normalize (default: -1, feature axis)
            epsilon: Small constant for numerical stability (1e-7 is robust)
            name: Layer name
        """
        super(UnitNorm, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.epsilon = epsilon
    
    def call(self, inputs, training=None):
        """
        Apply unit normalization.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode (unused, for consistency)
            
        Returns:
            Unit-normalized tensor with L2 norm = 1
        """
        # Compute L2 norm along specified axis
        norm = K.sqrt(K.sum(K.square(inputs), axis=self.axis, keepdims=True))
        
        # Normalize (add epsilon to avoid division by zero)
        normalized = inputs / (norm + self.epsilon)
        
        return normalized
    
    def compute_output_shape(self, input_shape):
        """Output shape is same as input shape."""
        return input_shape
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(UnitNorm, self).get_config()
        config.update({
            'axis': self.axis,
            'epsilon': self.epsilon,
        })
        return config


class UnitNormDense(layers.Layer):
    """
    Dense layer with unit normalization applied to output.
    
    Combines Dense layer with UnitNorm for convenience.
    Output = UnitNorm(Dense(input))
    """
    
    def __init__(
        self,
        units: int,
        activation=None,
        use_bias: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        name: str = "unit_norm_dense",
        **kwargs
    ):
        """
        Initialize unit-normalized dense layer.
        
        Args:
            units: Number of output units
            activation: Activation function (applied before normalization)
            use_bias: Whether to use bias
            kernel_regularizer: Kernel regularizer
            bias_regularizer: Bias regularizer
            name: Layer name
        """
        super(UnitNormDense, self).__init__(name=name, **kwargs)
        
        self.units = units
        self.dense = layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_dense"
        )
        self.unit_norm = UnitNorm(name=f"{name}_norm")
    
    def call(self, inputs, training=None):
        """Apply dense transformation and unit normalization."""
        x = self.dense(inputs, training=training)
        x = self.unit_norm(x, training=training)
        return x
    
    def get_config(self):
        """Get layer configuration."""
        config = super(UnitNormDense, self).get_config()
        config.update({
            'units': self.units,
        })
        return config


def apply_unit_norm(x, axis=-1, epsilon=1e-7):
    """
    Functional API for unit normalization.
    
    Args:
        x: Input tensor
        axis: Axis along which to normalize
        epsilon: Small constant for numerical stability (1e-7 is robust)
        
    Returns:
        Unit-normalized tensor
    """
    norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
    return x / (norm + epsilon)
