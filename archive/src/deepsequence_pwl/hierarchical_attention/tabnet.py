"""
Custom TabNet Implementation for TensorFlow 2.x

Lightweight TabNet encoder for feature selection and non-linear transformations.
Replaces PWL calibration + Dense + Attention in seasonal/holiday/regressor components.

Key Components:
- GhostBatchNormalization: Virtual batch normalization for stability
- GLU (Gated Linear Unit): σ(Wx) ⊙ (Vx) for non-linear feature transformation
- AttentiveTransformer: Sparse feature selection via attention masks
- FeatureTransformer: Shared + decision-specific feature processing
- TabNetEncoder: Sequential multi-step processing

Based on: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, 2019)
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Lambda, Multiply, Add
from tensorflow.keras import regularizers
import numpy as np


class GhostBatchNormalization(Layer):
    """
    Ghost Batch Normalization (Virtual Batch Normalization).
    
    Splits batch into virtual batches and normalizes each separately.
    Helps with small batch sizes and improves generalization.
    
    Args:
        virtual_batch_size: Size of virtual batches (default: 128)
        momentum: Momentum for moving average (default: 0.98)
    """
    
    def __init__(self, virtual_batch_size=128, momentum=0.98, **kwargs):
        super().__init__(**kwargs)
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
    def build(self, input_shape):
        self.bn = BatchNormalization(momentum=self.momentum)
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Simplified: Just use regular batch norm to avoid tf.function issues."""
        return self.bn(inputs, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'virtual_batch_size': self.virtual_batch_size,
            'momentum': self.momentum
        })
        return config


class GLUBlock(Layer):
    """
    Gated Linear Unit: GLU(x) = σ(Wx + b) ⊙ (Vx + c)
    
    Provides non-linear transformation with gating mechanism.
    The sigmoid gate controls information flow.
    
    Args:
        units: Output dimension
        virtual_batch_size: Virtual batch size for Ghost BN
        momentum: BN momentum
        fc: Optional shared fully connected layer
    """
    
    def __init__(self, units, virtual_batch_size=128, momentum=0.98, fc=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.fc = fc
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        if self.fc is None:
            # Create new FC layer (not shared)
            self.fc = Dense(
                self.units * 2,
                use_bias=False,
                kernel_regularizer=regularizers.l2(1e-4)
            )
        
        self.bn = GhostBatchNormalization(
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum
        )
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        # Linear transformation
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        
        # Split into two halves for gating
        out = self.units
        gate = tf.nn.sigmoid(x[..., :out])
        hidden = x[..., out:]
        
        # Apply gate: element-wise multiplication
        return tf.multiply(gate, hidden)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'virtual_batch_size': self.virtual_batch_size,
            'momentum': self.momentum
        })
        return config


class AttentiveTransformer(Layer):
    """
    Attentive Transformer for feature selection.
    
    Computes sparse attention masks to select relevant features.
    Uses sparsemax (approximated via softmax with temperature) for exact zeros.
    
    Args:
        input_dim: Input feature dimension
        virtual_batch_size: Virtual batch size for Ghost BN
        momentum: BN momentum
    """
    
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.98, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
    def build(self, input_shape):
        # Attention weights
        self.attention_fc = Dense(
            self.input_dim,
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.bn = GhostBatchNormalization(
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum
        )
        super().build(input_shape)
        
    def call(self, inputs, prior_scales, training=None):
        """
        Args:
            inputs: Processed features [batch, feature_dim]
            prior_scales: Prior scale from previous step [batch, input_dim]
            training: Training mode
            
        Returns:
            attention_mask: Sparse attention weights [batch, input_dim]
        """
        # Compute attention scores
        x = self.attention_fc(inputs)
        x = self.bn(x, training=training)
        
        # Multiply by prior scales (from previous steps)
        x = tf.multiply(x, prior_scales)
        
        # Sparsemax approximation: softmax with temperature
        # Lower temperature → more sparse (closer to sparsemax)
        temperature = 1.0
        attention_mask = tf.nn.softmax(x / temperature, axis=-1)
        
        return attention_mask
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'virtual_batch_size': self.virtual_batch_size,
            'momentum': self.momentum
        })
        return config


class FeatureTransformer(Layer):
    """
    Feature Transformer: Shared + Decision-specific processing.
    
    Applies a sequence of GLU blocks to transform features.
    Can be shared across decision steps or decision-specific.
    
    Args:
        feature_dim: Output feature dimension
        num_layers: Number of GLU layers (default: 2)
        virtual_batch_size: Virtual batch size for Ghost BN
        momentum: BN momentum
        shared_fc: Optional shared FC layer for first GLU
    """
    
    def __init__(self, feature_dim, num_layers=2, virtual_batch_size=128, 
                 momentum=0.98, shared_fc=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.shared_fc = shared_fc
        
    def build(self, input_shape):
        self.glu_layers = []
        
        for i in range(self.num_layers):
            # First layer can use shared FC
            fc = self.shared_fc if (i == 0 and self.shared_fc is not None) else None
            
            glu = GLUBlock(
                units=self.feature_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                fc=fc,
                name=f'glu_{i}'
            )
            self.glu_layers.append(glu)
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        x = inputs
        for i, glu in enumerate(self.glu_layers):
            glu_out = glu(x, training=training)
            
            # Residual connection: only add if shapes match
            if i > 0:  # Skip residual for first layer (shapes may not match)
                x = tf.multiply(glu_out, 0.5) + tf.multiply(x, 0.5)
            else:
                x = glu_out
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'num_layers': self.num_layers,
            'virtual_batch_size': self.virtual_batch_size,
            'momentum': self.momentum
        })
        return config


class TabNetEncoder(Layer):
    """
    TabNet Encoder: Sequential multi-step feature selection and processing.
    
    Replaces PWL + Dense + Attention in our architecture:
    - Old: Input → PWL → Dense → SparseAttention → Dropout → Output
    - New: Input → TabNet(n_steps) → Output
    
    Each step:
    1. AttentiveTransformer: Select features via attention mask
    2. FeatureTransformer: Process masked features via GLU blocks
    3. Aggregate: Accumulate decision outputs
    
    Args:
        feature_dim: Feature dimension for processing (default: 32)
        output_dim: Output dimension (default: 32)
        num_steps: Number of sequential decision steps (default: 3)
        num_shared_layers: Number of shared GLU layers (default: 2)
        num_decision_layers: Number of decision-specific GLU layers (default: 1)
        relaxation_factor: Relaxation factor for prior scaling (default: 1.5)
        virtual_batch_size: Virtual batch size for Ghost BN (default: 128)
        momentum: BN momentum (default: 0.98)
    """
    
    def __init__(self, feature_dim=32, output_dim=32, num_steps=3,
                 num_shared_layers=2, num_decision_layers=1,
                 relaxation_factor=1.5, virtual_batch_size=128, 
                 momentum=0.98, sparsity_coefficient=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.num_shared_layers = num_shared_layers
        self.num_decision_layers = num_decision_layers
        self.relaxation_factor = relaxation_factor
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.sparsity_coefficient = sparsity_coefficient
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Input batch normalization
        self.input_bn = BatchNormalization(momentum=self.momentum)
        
        # Shared feature transformer (used across all steps)
        shared_fc = Dense(
            self.feature_dim * 2,
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4)
        )
        
        self.shared_transformer = FeatureTransformer(
            feature_dim=self.feature_dim,
            num_layers=self.num_shared_layers,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            shared_fc=shared_fc
        )
        
        # Create decision steps
        self.attention_transformers = []
        self.feature_transformers = []
        self.output_layers = []
        
        for step in range(self.num_steps):
            # Attention for feature selection
            attn = AttentiveTransformer(
                input_dim=input_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                name=f'attention_step_{step}'
            )
            self.attention_transformers.append(attn)
            
            # Decision-specific feature transformer
            ft = FeatureTransformer(
                feature_dim=self.feature_dim,
                num_layers=self.num_decision_layers,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                name=f'feature_transformer_step_{step}'
            )
            self.feature_transformers.append(ft)
            
            # Output layer for this step
            output_layer = Dense(
                self.output_dim,
                activation=None,
                use_bias=False,
                kernel_regularizer=regularizers.l2(1e-4),
                name=f'output_step_{step}'
            )
            self.output_layers.append(output_layer)
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        input_dim = tf.shape(inputs)[-1]
        
        # Normalize input
        x = self.input_bn(inputs, training=training)
        
        # Initialize prior scale (all features equally important initially)
        prior_scales = tf.ones([batch_size, input_dim])
        
        # Accumulate outputs from all steps
        aggregated_output = tf.zeros([batch_size, self.output_dim])
        
        # Store attention masks for interpretability (optional)
        attention_masks = []
        
        # Sparsity loss accumulator
        total_sparsity_loss = 0.0
        
        # Sequential decision steps
        for step in range(self.num_steps):
            # 1. Attentive Transformer: Select features
            attention_mask = self.attention_transformers[step](
                x, prior_scales, training=training
            )
            attention_masks.append(attention_mask)
            
            # 2. Add entropy-based sparsity loss
            if training and self.sparsity_coefficient > 0:
                # Average attention across batch
                avg_mask = tf.reduce_mean(attention_mask, axis=0)
                # Entropy: -sum(p * log(p))
                # Lower entropy = sparser selection
                entropy = -tf.reduce_sum(
                    avg_mask * tf.math.log(avg_mask + 1e-10)
                )
                total_sparsity_loss += entropy
            
            # 3. Mask input features
            masked_input = tf.multiply(x, attention_mask)
            
            # 4. Shared feature processing
            shared_features = self.shared_transformer(
                masked_input, training=training
            )
            
            # 5. Decision-specific processing
            decision_features = self.feature_transformers[step](
                shared_features, training=training
            )
            
            # 6. Generate output for this step
            step_output = self.output_layers[step](decision_features)
            
            # 7. Aggregate output
            aggregated_output = aggregated_output + step_output
            
            # 8. Update prior scales for next step
            # Reduce importance of already-used features
            prior_scales = tf.multiply(
                prior_scales,
                (self.relaxation_factor - attention_mask)
            )
        
        # Add sparsity regularization loss
        if training and self.sparsity_coefficient > 0:
            self.add_loss(self.sparsity_coefficient * total_sparsity_loss)
        
        return aggregated_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'num_steps': self.num_steps,
            'num_shared_layers': self.num_shared_layers,
            'num_decision_layers': self.num_decision_layers,
            'relaxation_factor': self.relaxation_factor,
            'virtual_batch_size': self.virtual_batch_size,
            'momentum': self.momentum,
            'sparsity_coefficient': self.sparsity_coefficient
        })
        return config
