"""
Flexible Combination Layer for DeepSequence components.

Supports various mathematical combinations of seasonal, trend, and impact components:
- Additive: seasonal + trend + impact
- Multiplicative: seasonal × trend × impact
- Hybrid: seasonal × impact + trend
- Custom: any user-defined combination
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Optional, Callable


class CombinationLayer(layers.Layer):
    """
    Flexible layer to combine multiple forecast components.
    
    Supports various combination modes:
    - 'additive': sum all components
    - 'multiplicative': multiply all components
    - 'seasonal_multiplicative': seasonal × (trend + impact)
    - 'hybrid_add': seasonal × impact + trend
    - 'hybrid_mult': seasonal × trend + impact
    - 'three_way_mult': seasonal × trend × impact
    - 'custom': use custom combination function
    
    Examples:
        # Two-component combinations
        output = CombinationLayer(mode='additive')([seasonal, trend])
        output = CombinationLayer(mode='multiplicative')([seasonal, trend])
        
        # Three-component combinations
        output = CombinationLayer(mode='hybrid_add')([seasonal, trend, impact])
        # Result: seasonal × impact + trend
        
        output = CombinationLayer(mode='three_way_mult')([seasonal, trend, impact])
        # Result: seasonal × trend × impact
        
        # Custom combination
        def custom_fn(components):
            seasonal, trend, impact = components
            return seasonal * impact + trend * 0.5
        output = CombinationLayer(mode='custom', 
                                  custom_fn=custom_fn)([seasonal, trend, impact])
    """
    
    def __init__(
        self,
        mode: str = 'additive',
        custom_fn: Optional[Callable] = None,
        epsilon: float = 1e-7,
        name: str = 'combination',
        **kwargs
    ):
        """
        Initialize combination layer.
        
        Args:
            mode: Combination mode (see class docstring for options)
            custom_fn: Custom function for 'custom' mode
            epsilon: Small constant to prevent numerical issues
            name: Layer name
        """
        super(CombinationLayer, self).__init__(name=name, **kwargs)
        self.mode = mode
        self.custom_fn = custom_fn
        self.epsilon = epsilon
        
        # Validate mode
        valid_modes = [
            'additive', 'multiplicative', 'seasonal_multiplicative',
            'hybrid_add', 'hybrid_mult', 'three_way_mult', 'custom'
        ]
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {valid_modes}"
            )
        
        if mode == 'custom' and custom_fn is None:
            raise ValueError("custom_fn must be provided for mode='custom'")
    
    def call(self, inputs: List[tf.Tensor], training=None):
        """
        Combine input components according to specified mode.
        
        Args:
            inputs: List of component tensors [seasonal, trend, impact, ...]
            training: Whether in training mode
            
        Returns:
            Combined forecast tensor
        """
        if not isinstance(inputs, list):
            raise ValueError("inputs must be a list of tensors")
        
        num_components = len(inputs)
        
        # Two-component modes
        if num_components == 2:
            seasonal, trend = inputs
            
            if self.mode == 'additive':
                return seasonal + trend
            
            elif self.mode == 'multiplicative':
                return seasonal * trend
            
            elif self.mode == 'seasonal_multiplicative':
                # seasonal × trend (same as multiplicative for 2 components)
                return seasonal * trend
            
            elif self.mode in ['hybrid_add', 'hybrid_mult', 'three_way_mult']:
                # These modes require 3+ components
                raise ValueError(
                    f"Mode '{self.mode}' requires at least 3 components, "
                    f"but got {num_components}"
                )
            
            elif self.mode == 'custom':
                return self.custom_fn(inputs)
        
        # Three-component modes
        elif num_components == 3:
            seasonal, trend, impact = inputs
            
            if self.mode == 'additive':
                return seasonal + trend + impact
            
            elif self.mode == 'multiplicative' or self.mode == 'three_way_mult':
                return seasonal * trend * impact
            
            elif self.mode == 'seasonal_multiplicative':
                # seasonal × (trend + impact)
                return seasonal * (trend + impact)
            
            elif self.mode == 'hybrid_add':
                # seasonal × impact + trend
                return seasonal * impact + trend
            
            elif self.mode == 'hybrid_mult':
                # seasonal × trend + impact
                return seasonal * trend + impact
            
            elif self.mode == 'custom':
                return self.custom_fn(inputs)
        
        # Four+ component modes (general case)
        else:
            if self.mode == 'additive':
                # Sum all components
                result = inputs[0]
                for component in inputs[1:]:
                    result = result + component
                return result
            
            elif self.mode == 'multiplicative' or self.mode == 'three_way_mult':
                # Multiply all components
                result = inputs[0]
                for component in inputs[1:]:
                    result = result * component
                return result
            
            elif self.mode == 'seasonal_multiplicative':
                # seasonal × (sum of all others)
                seasonal = inputs[0]
                other_sum = inputs[1]
                for component in inputs[2:]:
                    other_sum = other_sum + component
                return seasonal * other_sum
            
            elif self.mode == 'custom':
                return self.custom_fn(inputs)
            
            else:
                raise ValueError(
                    f"Mode '{self.mode}' not supported for {num_components} components. "
                    f"Use 'additive', 'multiplicative', 'seasonal_multiplicative', or 'custom'."
                )
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(CombinationLayer, self).get_config()
        config.update({
            'mode': self.mode,
            'epsilon': self.epsilon,
        })
        # Note: custom_fn cannot be serialized
        return config


class ScalableCombination(layers.Layer):
    """
    Learnable combination of components with trainable weights.
    
    Instead of fixed combinations, learns optimal weights:
    output = w1 × seasonal + w2 × trend + w3 × impact
    or
    output = seasonal × (w1 × trend + w2 × impact)
    
    Useful when you want the model to learn the best combination.
    """
    
    def __init__(
        self,
        mode: str = 'weighted_additive',
        use_bias: bool = True,
        constraint: Optional[str] = None,
        name: str = 'scalable_combination',
        **kwargs
    ):
        """
        Initialize scalable combination layer.
        
        Args:
            mode: 'weighted_additive' or 'weighted_multiplicative'
            use_bias: Whether to add a bias term
            constraint: Weight constraint ('non_negative', 'unit_sum', None)
            name: Layer name
        """
        super(ScalableCombination, self).__init__(name=name, **kwargs)
        self.mode = mode
        self.use_bias = use_bias
        self.constraint = constraint
        self.weights_list = []
        self.bias = None
    
    def build(self, input_shape):
        """Build layer weights."""
        if not isinstance(input_shape, list):
            raise ValueError("input_shape must be a list")
        
        num_components = len(input_shape)
        
        # Create weight for each component
        if self.constraint == 'non_negative':
            constraint = tf.keras.constraints.NonNeg()
        elif self.constraint == 'unit_sum':
            constraint = tf.keras.constraints.UnitNorm(axis=0)
        else:
            constraint = None
        
        self.weights_list = [
            self.add_weight(
                name=f'weight_{i}',
                shape=(1,),
                initializer='ones',
                trainable=True,
                constraint=constraint
            )
            for i in range(num_components)
        ]
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(1,),
                initializer='zeros',
                trainable=True
            )
        
        super(ScalableCombination, self).build(input_shape)
    
    def call(self, inputs: List[tf.Tensor], training=None):
        """Combine inputs with learned weights."""
        if self.mode == 'weighted_additive':
            # output = Σ(w_i × component_i) + bias
            result = self.weights_list[0] * inputs[0]
            for i in range(1, len(inputs)):
                result = result + self.weights_list[i] * inputs[i]
            
            if self.use_bias:
                result = result + self.bias
            
            return result
        
        elif self.mode == 'weighted_multiplicative':
            # output = seasonal × Σ(w_i × other_i) + bias
            seasonal = inputs[0]
            
            weighted_sum = self.weights_list[1] * inputs[1]
            for i in range(2, len(inputs)):
                weighted_sum = weighted_sum + self.weights_list[i] * inputs[i]
            
            result = seasonal * weighted_sum
            
            if self.use_bias:
                result = result + self.bias
            
            return result
    
    def get_config(self):
        """Get layer configuration."""
        config = super(ScalableCombination, self).get_config()
        config.update({
            'mode': self.mode,
            'use_bias': self.use_bias,
            'constraint': self.constraint,
        })
        return config
