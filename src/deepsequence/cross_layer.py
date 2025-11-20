"""
Cross Network Layer for Deep & Cross Network (DCN)

Learns explicit feature interactions through cross product operations.
Captures high-order feature interactions efficiently.

Reference: Wang et al., "Deep & Cross Network for Ad Click Predictions", 2017
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CrossLayer(layers.Layer):
    """
    Cross Network layer that learns explicit feature interactions.
    
    Computes: x_{l+1} = x_0 * (w_l^T * x_l) + b_l + x_l
    
    This creates polynomial feature interactions while keeping computational
    complexity linear in feature dimension.
    
    Args:
        units: Dimensionality of the layer (should match input dimension)
        use_bias: Whether to use bias term
        kernel_initializer: Initializer for weight matrix
        bias_initializer: Initializer for bias vector
        kernel_regularizer: Regularizer for weight matrix
        **kwargs: Additional layer arguments
        
    Example:
        >>> cross = CrossLayer(units=64)
        >>> x = tf.random.normal((32, 64))
        >>> output = cross(x)  # Shape: (32, 64)
    """
    
    def __init__(
        self,
        units,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        **kwargs
    ):
        super(CrossLayer, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        
    def build(self, input_shape):
        """Build layer weights."""
        # Handle tuple of shapes (x_0, x_l) or single shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            # When input is (x_0, x_l), both should have same shape
            # Check if first element is also a shape tuple
            if isinstance(input_shape[0], (list, tuple)):
                feature_dim = input_shape[0][-1]
            else:
                # Single shape case
                feature_dim = input_shape[-1]
        else:
            # Single input case
            feature_dim = input_shape[-1]
        
        # Convert TensorShape/Dimension to int (Keras 2.x compatibility)
        # Handle TensorShape objects
        from tensorflow.python.framework import tensor_shape
        if isinstance(feature_dim, tensor_shape.TensorShape):
            # TensorShape object - get its dimensions list
            feature_dim = feature_dim.as_list()[-1]
        elif hasattr(feature_dim, 'value'):
            # Dimension object with .value attribute
            feature_dim = feature_dim.value
        elif not isinstance(feature_dim, int):
            # Try direct conversion for other types
            try:
                feature_dim = int(feature_dim)
            except (TypeError, ValueError):
                raise TypeError(f"Cannot convert feature_dim type {type(feature_dim)} to int")
            
        self.kernel = self.add_weight(
            name='kernel',
            shape=(feature_dim,),  # Weight vector (not matrix)
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                trainable=True
            )
        else:
            self.bias = None
            
        super(CrossLayer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        """
        Forward pass of cross layer.
        
        Args:
            inputs: Tuple of (x_0, x_l) where:
                x_0: Initial input (stored from first call)
                x_l: Current layer input
                
            If single tensor provided, uses it as both x_0 and x_l
            
        Returns:
            x_{l+1}: Cross layer output with same shape as input
        """
        if isinstance(inputs, (list, tuple)):
            x_0, x_l = inputs
        else:
            x_0 = x_l = inputs
            
        # Compute: w^T * x_l (scalar for each sample)
        # Shape: (batch_size, features) * (features,) = (batch_size,)
        prod = tf.reduce_sum(x_l * self.kernel, axis=-1, keepdims=True)
        
        # Compute: x_0 * prod (element-wise multiplication broadcasted)
        # Shape: (batch_size, features) * (batch_size, 1) = (batch_size, features)
        interaction = x_0 * prod
        
        # Add bias and residual connection
        # x_{l+1} = x_0 * (w^T * x_l) + b + x_l
        output = interaction + x_l
        
        if self.use_bias:
            output = output + self.bias
            
        return output
    
    def get_config(self):
        """Returns layer configuration for serialization."""
        config = super(CrossLayer, self).get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


class CrossNetwork(layers.Layer):
    """
    Stacked Cross Network layers.
    
    Multiple cross layers stacked sequentially to learn higher-order
    feature interactions.
    
    Args:
        num_layers: Number of cross layers to stack
        units: Dimensionality of each layer
        use_bias: Whether to use bias in cross layers
        **kwargs: Additional layer arguments
        
    Example:
        >>> cross_net = CrossNetwork(num_layers=3, units=64)
        >>> x = tf.random.normal((32, 64))
        >>> output = cross_net(x)  # Shape: (32, 64)
    """
    
    def __init__(
        self,
        num_layers=2,
        units=None,
        use_bias=True,
        **kwargs
    ):
        super(CrossNetwork, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.units = units
        self.use_bias = use_bias
        self.cross_layers = []
        
    def build(self, input_shape):
        """Build stacked cross layers."""
        units = self.units or input_shape[-1]
        
        for i in range(self.num_layers):
            self.cross_layers.append(
                CrossLayer(
                    units=units,
                    use_bias=self.use_bias,
                    name=f'cross_layer_{i}'
                )
            )
        
        super(CrossNetwork, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        """
        Forward pass through stacked cross layers.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output after num_layers cross operations
        """
        x_0 = inputs  # Store initial input
        x_l = inputs  # Current layer input
        
        for cross_layer in self.cross_layers:
            x_l = cross_layer((x_0, x_l))
            
        return x_l
    
    def get_config(self):
        """Returns network configuration for serialization."""
        config = super(CrossNetwork, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'units': self.units,
            'use_bias': self.use_bias,
        })
        return config


def apply_cross_network(x, num_layers=2, use_bias=True, name='cross_network'):
    """
    Convenience function to apply cross network to tensor.
    
    Args:
        x: Input tensor
        num_layers: Number of cross layers
        use_bias: Whether to use bias
        name: Name prefix for layers
        
    Returns:
        Output tensor after cross network
        
    Example:
        >>> x = tf.keras.Input(shape=(64,))
        >>> cross_out = apply_cross_network(x, num_layers=2)
    """
    cross_net = CrossNetwork(
        num_layers=num_layers,
        use_bias=use_bias,
        name=name
    )
    return cross_net(x)
