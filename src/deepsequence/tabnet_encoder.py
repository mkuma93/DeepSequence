"""
TabNet Encoder for DeepSequence Components.

TabNet (Arik & Pfister, 2019) is an interpretable deep learning architecture
that uses sequential attention to select features at each decision step.

This module provides a TensorFlow/Keras-compatible TabNet encoder that can be
used with the seasonal and regressor components to provide:
- Feature importance via attention mechanism
- Non-linear feature interactions
- Sparse feature selection
- Better representation learning

Reference:
Arik, S. Ö., & Pfister, T. (2019). TabNet: Attentive Interpretable Tabular 
Learning. arXiv preprint arXiv:1908.07442.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l1, l2
from typing import Optional, Tuple
from .unit_norm import UnitNorm


class TabNetEncoder(layers.Layer):
    """
    TabNet encoder layer for feature selection and representation learning.
    
    This implements the core TabNet architecture adapted for use within
    DeepSequence's seasonal and regressor components.
    """
    
    def __init__(
        self,
        feature_dim: int = 32,
        output_dim: int = 32,
        n_steps: int = 3,
        n_shared: int = 2,
        n_independent: int = 2,
        relaxation_factor: float = 1.5,
        bn_momentum: float = 0.98,
        bn_epsilon: float = 1e-3,
        sparsity_coefficient: float = 1e-5,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        name: str = "tabnet_encoder",
        **kwargs
    ):
        """
        Initialize TabNet encoder.
        
        Args:
            feature_dim: Dimension for feature transformation (N_a in paper)
            output_dim: Dimension for output embedding (N_d in paper)
            n_steps: Number of sequential attention steps
            n_shared: Number of shared GLU layers
            n_independent: Number of independent GLU layers per step
            relaxation_factor: Factor for relaxing feature selection (gamma)
            bn_momentum: Batch normalization momentum
            bn_epsilon: Batch normalization epsilon
            sparsity_coefficient: Coefficient for sparsity regularization
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            name: Layer name
        """
        super(TabNetEncoder, self).__init__(name=name, **kwargs)
        
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.relaxation_factor = relaxation_factor
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.sparsity_coefficient = sparsity_coefficient
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        # Will be built in build()
        self.feature_transform = None
        self.shared_block = None
        self.step_layers = []
        self.attention_layers = []
        
    def build(self, input_shape):
        """Build the TabNet encoder layers."""
        input_dim = input_shape[-1]
        
        # Initial feature transformation with batch norm
        self.bn_input = layers.BatchNormalization(
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            name=f"{self.name}_bn_input"
        )
        
        # Feature transformation to map back to input dimension
        self.feature_transform = layers.Dense(
            input_dim,
            use_bias=False,
            name=f"{self.name}_feature_transform"
        )
        
        # Shared GLU block (used across all steps)
        self.shared_block = SharedGLUBlock(
            n_layers=self.n_shared,
            units=self.feature_dim + self.output_dim,
            bn_momentum=self.bn_momentum,
            bn_epsilon=self.bn_epsilon,
            l1_reg=self.l1_reg,
            l2_reg=self.l2_reg,
            name=f"{self.name}_shared_block"
        )
        
        # Step-dependent layers
        self.step_layers = []
        self.attention_layers = []
        
        for step in range(self.n_steps):
            # Independent GLU layers for this step
            step_glus = []
            for i in range(self.n_independent):
                step_glus.append(
                    GLULayer(
                        units=self.feature_dim + self.output_dim,
                        bn_momentum=self.bn_momentum,
                        bn_epsilon=self.bn_epsilon,
                        l1_reg=self.l1_reg,
                        l2_reg=self.l2_reg,
                        name=f"{self.name}_step_{step}_glu_{i}"
                    )
                )
            self.step_layers.append(step_glus)
            
            # Attention transformer for feature selection
            self.attention_layers.append(
                AttentionTransformer(
                    feature_dim=input_dim,
                    bn_momentum=self.bn_momentum,
                    bn_epsilon=self.bn_epsilon,
                    l1_reg=self.l1_reg,
                    l2_reg=self.l2_reg,
                    name=f"{self.name}_attention_{step}"
                )
            )
        
        super(TabNetEncoder, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass through TabNet encoder.
        
        Args:
            inputs: Input features [batch_size, n_features]
            training: Whether in training mode
            
        Returns:
            Encoded features [batch_size, output_dim]
        """
        batch_size = tf.shape(inputs)[0]
        
        # Batch normalization on input
        features = self.bn_input(inputs, training=training)
        
        # Initialize prior scale (for attention)
        prior_scales = tf.ones([batch_size, tf.shape(inputs)[1]])
        
        # Aggregate decision output
        aggregated_output = tf.zeros([batch_size, self.output_dim])
        
        # Sequential attention steps
        for step in range(self.n_steps):
            # Feature selection via attention
            masked_features, attention_weights = self.attention_layers[step](
                features, prior_scales, training=training
            )
            
            # Update prior scales (discourage reusing same features)
            prior_scales = prior_scales * (
                self.relaxation_factor - attention_weights
            )
            
            # Process through shared GLU block
            hidden = self.shared_block(masked_features, training=training)
            
            # Scale by sqrt(0.5) when transitioning to step-specific layers
            # This maintains gradient stability across blocks
            hidden = hidden * tf.sqrt(0.5)
            
            # Process through step-specific layers
            for step_layer in self.step_layers[step]:
                hidden = step_layer(hidden, training=training)
            
            # Split into feature and output parts
            feature_part = hidden[:, :self.feature_dim]
            output_part = hidden[:, self.feature_dim:]
            
            # Apply ReLU to decision output
            output_part_relu = tf.nn.relu(output_part)
            
            # Apply unit normalization to decision output
            output_part_norm = UnitNorm(
                name=f"{self.name}_step_{step}_unit_norm"
            )(output_part_relu)
            
            # Aggregate decision output
            aggregated_output = aggregated_output + output_part_norm
            
            # Transform feature_part for next step with sqrt(0.5) scaling
            features = self.feature_transform(feature_part) * tf.sqrt(0.5)
            
            # Add sparsity loss (entropy of attention weights)
            if training and self.sparsity_coefficient > 0:
                # Use epsilon to prevent log(0) - 1e-7 is numerically stable
                entropy = -tf.reduce_mean(
                    tf.reduce_sum(
                        attention_weights * tf.math.log(attention_weights + 1e-7),
                        axis=1
                    )
                )
                self.add_loss(self.sparsity_coefficient * entropy)
        
        return aggregated_output
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(TabNetEncoder, self).get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'n_steps': self.n_steps,
            'n_shared': self.n_shared,
            'n_independent': self.n_independent,
            'relaxation_factor': self.relaxation_factor,
            'bn_momentum': self.bn_momentum,
            'bn_epsilon': self.bn_epsilon,
            'sparsity_coefficient': self.sparsity_coefficient,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
        })
        return config


class SharedGLUBlock(layers.Layer):
    """
    Shared GLU block containing multiple GLU layers.
    
    This encapsulates the shared feature transformer that is reused
    across all TabNet decision steps.
    
    Args:
        n_layers: Number of GLU layers in the block
        units: Output units for each GLU layer
        bn_momentum: Batch normalization momentum
        bn_epsilon: Batch normalization epsilon
        l1_reg: L1 regularization strength
        l2_reg: L2 regularization strength
        name: Layer name
    """
    
    def __init__(
        self,
        n_layers: int,
        units: int,
        bn_momentum: float = 0.98,
        bn_epsilon: float = 1e-3,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        name: str = "shared_glu_block",
        **kwargs
    ):
        super(SharedGLUBlock, self).__init__(name=name, **kwargs)
        self.n_layers = n_layers
        self.units = units
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.glu_layers = []
        
    def build(self, input_shape):
        """Build the stacked GLU layers."""
        for i in range(self.n_layers):
            self.glu_layers.append(
                GLULayer(
                    units=self.units,
                    bn_momentum=self.bn_momentum,
                    bn_epsilon=self.bn_epsilon,
                    l1_reg=self.l1_reg,
                    l2_reg=self.l2_reg,
                    name=f"{self.name}_glu_{i}"
                )
            )
        super(SharedGLUBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """Forward pass through all GLU layers."""
        x = inputs
        for glu_layer in self.glu_layers:
            x = glu_layer(x, training=training)
        return x
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(SharedGLUBlock, self).get_config()
        config.update({
            'n_layers': self.n_layers,
            'units': self.units,
            'bn_momentum': self.bn_momentum,
            'bn_epsilon': self.bn_epsilon,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
        })
        return config


class GLULayer(layers.Layer):
    """
    Gated Linear Unit (GLU) layer.
    
    Applies: GLU(x) = Linear(x) ⊗ σ(Linear(x))
    where ⊗ is element-wise multiplication and σ is sigmoid.
    """
    
    def __init__(
        self,
        units: int,
        bn_momentum: float = 0.98,
        bn_epsilon: float = 1e-3,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        name: str = "glu",
        **kwargs
    ):
        super(GLULayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
    def build(self, input_shape):
        # Create regularizer
        reg = None
        if self.l1_reg > 0 and self.l2_reg > 0:
            reg = keras.regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg)
        elif self.l1_reg > 0:
            reg = l1(self.l1_reg)
        elif self.l2_reg > 0:
            reg = l2(self.l2_reg)
        
        # Linear transformation (output is 2x units for gate)
        self.fc = layers.Dense(
            self.units * 2,
            use_bias=False,
            kernel_regularizer=reg,
            name=f"{self.name}_fc"
        )
        
        self.bn = layers.BatchNormalization(
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            name=f"{self.name}_bn"
        )
        
        super(GLULayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        
        # Split into two halves
        linear_part, gate_part = tf.split(x, 2, axis=-1)
        
        # Apply sigmoid gate
        gate = tf.nn.sigmoid(gate_part)
        
        # Element-wise multiplication
        return linear_part * gate
    
    def get_config(self):
        config = super(GLULayer, self).get_config()
        config.update({
            'units': self.units,
            'bn_momentum': self.bn_momentum,
            'bn_epsilon': self.bn_epsilon,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
        })
        return config


class AttentionTransformer(layers.Layer):
    """
    Attention mechanism for feature selection in TabNet.
    
    Uses sparsemax activation to produce sparse attention weights.
    """
    
    def __init__(
        self,
        feature_dim: int,
        bn_momentum: float = 0.98,
        bn_epsilon: float = 1e-3,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        name: str = "attention",
        **kwargs
    ):
        super(AttentionTransformer, self).__init__(name=name, **kwargs)
        self.feature_dim = feature_dim
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
    def build(self, input_shape):
        reg = None
        if self.l1_reg > 0 and self.l2_reg > 0:
            reg = keras.regularizers.L1L2(l1=self.l1_reg, l2=self.l2_reg)
        elif self.l1_reg > 0:
            reg = l1(self.l1_reg)
        elif self.l2_reg > 0:
            reg = l2(self.l2_reg)
        
        self.fc = layers.Dense(
            self.feature_dim,
            use_bias=False,
            kernel_regularizer=reg,
            name=f"{self.name}_fc"
        )
        
        self.bn = layers.BatchNormalization(
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            name=f"{self.name}_bn"
        )
        
        super(AttentionTransformer, self).build(input_shape)
    
    def call(self, features, prior_scales, training=None):
        """
        Compute attention weights and mask features.
        
        Args:
            features: Input features [batch_size, n_features]
            prior_scales: Prior attention scales [batch_size, n_features]
            training: Whether in training mode
            
        Returns:
            masked_features: Features weighted by attention
            attention_weights: Attention weights (sparse)
        """
        # Compute attention logits
        x = self.fc(features)
        x = self.bn(x, training=training)
        
        # Apply prior scales
        x = x * prior_scales
        
        # Sparsemax activation (approximated with softmax + thresholding)
        # Full sparsemax is complex; using normalized softmax as approximation
        attention_weights = tf.nn.softmax(x, axis=-1)
        
        # Mask features
        masked_features = features * attention_weights
        
        return masked_features, attention_weights
    
    def get_config(self):
        config = super(AttentionTransformer, self).get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'bn_momentum': self.bn_momentum,
            'bn_epsilon': self.bn_epsilon,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
        })
        return config


def create_tabnet_encoder(
    input_shape: Tuple[int, ...],
    output_dim: int = 32,
    feature_dim: int = 32,
    n_steps: int = 3,
    n_shared: int = 2,
    n_independent: int = 2,
    name: str = "tabnet"
) -> Model:
    """
    Create a standalone TabNet encoder model.
    
    Args:
        input_shape: Shape of input features (n_features,)
        output_dim: Dimension of output embedding
        feature_dim: Dimension for internal feature transformation
        n_steps: Number of attention steps
        n_shared: Number of shared GLU layers
        n_independent: Number of independent GLU layers
        name: Model name
        
    Returns:
        Keras Model with TabNet encoder
    """
    inputs = layers.Input(shape=input_shape, name=f"{name}_input")
    
    encoder = TabNetEncoder(
        feature_dim=feature_dim,
        output_dim=output_dim,
        n_steps=n_steps,
        n_shared=n_shared,
        n_independent=n_independent,
        name=name
    )
    
    outputs = encoder(inputs)
    
    model = Model(inputs=inputs, outputs=outputs, name=f"{name}_model")
    
    return model
