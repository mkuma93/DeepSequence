"""
Hierarchical Attention Architecture for DeepSequence PWL

Three-level attention mechanism for sparse, interpretable forecasting:

1. PWL-level (Within PWL Components):
   - For Trend: Attention on changepoint features
     Example: "SKU uses changepoints 1, 5, 8 (early growth phase)"
   - For Holiday: Attention on distance ranges  
     Example: "SKU sensitive to 7-14 days before holiday only"
   
2. Feature-level (Within Each Component):
   - Attention on hidden features after PWL/Dense layers
   - Example: "In trend, features 3, 7, 12 matter; rest can be ignored"
   
3. Component-level (Across Components):
   - Attention across trend/seasonal/holiday/regressor
   - Example: "For this SKU: trend=60%, seasonal=30%, holiday=10%, regressor=0%"

Benefits:
- Multi-level sparsity (entmax at each level)
- Interpretable (can inspect attention at each level)
- SKU-specific (different SKUs use different patterns)
- Efficient (zeros out irrelevant features/components)

Architecture Flow:
┌─────────────────────────────────────────────────────────────┐
│ Input: [time_feature, seasonal_features, holiday_distance] │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┴────────────┬──────────────┬──────────────┐
    │                         │              │              │
┌───▼───────┐         ┌───────▼──┐    ┌─────▼─────┐  ┌────▼────┐
│   Trend   │         │ Seasonal │    │  Holiday  │  │Regressor│
│    PWL    │         │  Dense   │    │    PWL    │  │  Dense  │
│[batch,32] │         │[batch,32]│    │[batch,32] │  │[batch,32]│
└───┬───────┘         └────┬─────┘    └─────┬─────┘  └────┬────┘
    │                      │                 │             │
    │ PWL Attention        │                 │ PWL Attn    │
    ▼ (Level 1)            │                 ▼ (Level 1)   │
┌───────────┐              │            ┌──────────┐       │
│Masked PWL │              │            │Masked PWL│       │
└───┬───────┘              │            └─────┬────┘       │
    ▼                      │                  ▼             │
┌───────────┐         ┌────▼─────┐      ┌─────────┐  ┌────▼────┐
│   Dense   │         │  Dense   │      │  Dense  │  │  Dense  │
│[batch,32] │         │[batch,32]│      │[batch,32]│  │[batch,32]│
└───┬───────┘         └────┬─────┘      └─────┬───┘  └────┬────┘
    │                      │                   │            │
    │ Feature Attention (Level 2 for all components)        │
    ▼                      ▼                   ▼            ▼
┌───────────┐         ┌─────────┐        ┌─────────┐  ┌─────────┐
│Masked Feat│         │Masked F.│        │Masked F.│  │Masked F.│
└───┬───────┘         └────┬────┘        └─────┬───┘  └────┬────┘
    │                      │                    │            │
    │ Component Attention (Level 3 - across components)      │
    │                      │                    │            │
    ▼                      ▼                    ▼            ▼
┌───────────┐         ┌─────────┐        ┌─────────┐  ┌─────────┐
│×0.6 (60%) │         │×0.3(30%)│        │×0.1(10%)│  │×0.0 (0%)│
└───┬───────┘         └────┬────┘        └─────┬───┘  └────┬────┘
    │                      │                    │            │
    └──────────────────────┴────────────────────┴────────────┘
                           │
                    ┌──────▼───────┐
                    │ Concatenate  │
                    │  [batch,128] │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Zero Prob    │
                    │   Network    │
                    └──────────────┘
"""

import numpy as np
import tensorflow as tf
from tf_keras.layers import (
    Input, Dense, Embedding, Concatenate, Add, Multiply, Lambda, Dropout
)
from tf_keras.models import Model
from tf_keras import regularizers
from typing import List, Tuple, Optional

# Try importing TensorFlow Lattice for PWL
try:
    import tensorflow_lattice as tfl
    LATTICE_AVAILABLE = True
except ImportError:
    LATTICE_AVAILABLE = False
    print("Warning: tensorflow_lattice not available. PWL calibration disabled.")


class Entmax15(tf.keras.layers.Layer):
    """
    Entmax 1.5 activation - sparse softmax alternative.
    
    TensorFlow implementation of sparsemax (similar to entmax with alpha=2).
    Produces sparser attention weights than softmax, can output exact zeros.
    
    Reference: "From Softmax to Sparsemax" (Martins & Astudillo, 2016)
    """
    
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        """Apply sparsemax activation."""
        return self._sparsemax(inputs)
    
    def _sparsemax(self, logits):
        """
        Sparsemax activation function.
        
        Computes a sparse probability distribution that can have exact zeros.
        More sparse than softmax, suitable for attention mechanisms.
        """
        # Sort logits in descending order
        sorted_logits = tf.sort(logits, axis=self.axis, direction='DESCENDING')
        
        # Compute cumulative sum
        cumsum = tf.cumsum(sorted_logits, axis=self.axis)
        
        # Create range tensor [1, 2, 3, ..., n]
        shape = tf.shape(sorted_logits)
        rng = tf.range(1, shape[self.axis] + 1, dtype=logits.dtype)
        
        # Reshape range to broadcast correctly
        rank = len(logits.shape)
        if self.axis == -1 or self.axis == rank - 1:
            # Last axis - prepend ones
            new_shape = [1] * (rank - 1) + [-1]
            rng = tf.reshape(rng, new_shape)
        
        # Compute support: k(z) = max{k : 1 + k * z_k > sum_{j=1}^{k} z_j}
        support = sorted_logits * rng > (cumsum - 1.0)
        
        # Find k: the size of the support
        k = tf.reduce_sum(tf.cast(support, logits.dtype), axis=self.axis, keepdims=True)
        
        # Compute threshold: tau(z) = (sum_{j=1}^{k} z_j - 1) / k
        # Only sum up to k elements
        cumsum_masked = tf.where(support, cumsum, tf.zeros_like(cumsum))
        tau_sum = tf.reduce_sum(cumsum_masked, axis=self.axis, keepdims=True)
        tau = (tau_sum - 1.0) / tf.maximum(k, 1.0)
        
        # Apply sparsemax: max(logit - tau, 0)
        output = tf.maximum(logits - tau, 0.0)
        
        return output


class ComponentBuilder:
    """Base class for component builders."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
    
    def get_activation_fn(self):
        """Get activation function."""
        if self.activation == 'mish':
            return lambda x: x * tf.nn.tanh(tf.nn.softplus(x))
        return self.activation


class TrendComponentBuilder(ComponentBuilder):
    """Trend Component with PWL calibration for time features and shift-and-scale for SKU-specific effects."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_pwl: bool = True,
        n_changepoints: int = 10,
        time_min: float = None,
        time_max: float = None
    ):
        super().__init__(hidden_units, activation, dropout)
        self.use_pwl = use_pwl
        self.n_changepoints = n_changepoints
        self.time_min = time_min
        self.time_max = time_max
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build trend component with PWL calibration.
        
        Args:
            main_input: Full feature tensor
            id_embedding: SKU embedding
            feature_indices: Should contain just the time feature index (e.g., [0])
        
        Returns:
            (trend_hidden, trend_forecast)
        """
        # Extract time feature (typically just 1 feature: time in days)
        time_feature = Lambda(
            lambda x: tf.gather(x, feature_indices, axis=-1),
            name='time_feature'
        )(main_input)
        
        # Apply PWL calibration if available
        if self.use_pwl and LATTICE_AVAILABLE and self.time_min is not None:
            # PWL calibration learns changepoints automatically
            # Output units = hidden_units (creates multiple calibrated features)
            keypoints = np.linspace(
                self.time_min,
                self.time_max,
                num=self.n_changepoints
            ).astype(np.float32)
            
            with tf.device('/CPU:0'):  # PWL runs on CPU
                trend_pwl = tfl.layers.PWLCalibration(
                    input_keypoints=keypoints,
                    units=self.hidden_units,  # Output multiple calibrated features
                    output_min=-2.0,
                    output_max=2.0,
                    monotonicity='none',  # Allow flexible trends
                    kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                    name='trend_pwl'
                )(time_feature)
            # trend_pwl shape: [batch, hidden_units]
            
            # Dense layer on PWL features (attention applied later in intermittent handler)
            trend_hidden = Dense(
                self.hidden_units,
                activation=self.get_activation_fn(),
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                name='trend_hidden'
            )(trend_pwl)
        else:
            # Fallback: regular Dense layer
            trend_hidden = Dense(
                self.hidden_units,
                activation=self.get_activation_fn(),
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                name='trend_hidden'
            )(time_feature)
        
        trend_hidden = Dropout(self.dropout)(trend_hidden)
        
        # Shift-and-scale using ID embedding
        id_trend_beta = Dense(
            self.hidden_units,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='id_trend_beta'
        )(id_embedding)
        
        id_trend_alpha = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='id_trend_alpha'
        )(id_embedding)
        
        # Apply shift-and-scale
        trend_out = Multiply(name='trend_scale')([trend_hidden, id_trend_beta])
        trend_out = Add(name='trend_shift')([trend_out, id_trend_alpha])
        
        # Forecast
        trend_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='trend_forecast'
        )(trend_out)
        
        return trend_out, trend_forecast


class SeasonalComponentBuilder(ComponentBuilder):
    """Seasonal Component with shift-and-scale."""
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Build seasonal component."""
        seasonal_features = Lambda(
            lambda x: tf.gather(x, feature_indices, axis=-1),
            name='seasonal_features'
        )(main_input)
        
        seasonal_hidden = Dense(
            self.hidden_units,
            activation=self.get_activation_fn(),
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='seasonal_hidden'
        )(seasonal_features)
        seasonal_hidden = Dropout(self.dropout)(seasonal_hidden)
        
        # Shift-and-scale
        id_seasonal_beta = Dense(
            self.hidden_units,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='id_seasonal_beta'
        )(id_embedding)
        
        id_seasonal_alpha = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='id_seasonal_alpha'
        )(id_embedding)
        
        seasonal_out = Multiply(name='seasonal_scale')([seasonal_hidden, id_seasonal_beta])
        seasonal_out = Add(name='seasonal_shift')([seasonal_out, id_seasonal_alpha])
        
        seasonal_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='seasonal_forecast'
        )(seasonal_out)
        
        return seasonal_out, seasonal_forecast


class HolidayComponentBuilder(ComponentBuilder):
    """Holiday Component with PWL calibration and shift-and-scale."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_pwl: bool = True,
        n_keypoints: int = 37,
        keypoint_range: float = 365.0,
        data_frequency: str = 'daily'
    ):
        super().__init__(hidden_units, activation, dropout)
        self.use_pwl = use_pwl
        self.n_keypoints = n_keypoints
        self.keypoint_range = keypoint_range
        self.data_frequency = data_frequency
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_index: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build holiday component with PWL calibration.
        
        Args:
            main_input: Full feature tensor
            id_embedding: SKU embedding
            feature_index: Index of holiday distance feature
        
        Returns:
            (holiday_hidden, holiday_forecast)
        """
        holiday_feature = Lambda(
            lambda x: tf.gather(x, [feature_index], axis=-1),
            name='holiday_feature'
        )(main_input)
        
        # Apply PWL calibration if available
        if self.use_pwl and LATTICE_AVAILABLE:
            # Adapt keypoint range based on data frequency
            if self.data_frequency == 'daily':
                kp_range = 365  # ±1 year
                num_kp = 37
            elif self.data_frequency == 'weekly':
                kp_range = 364  # ±52 weeks
                num_kp = 27
            elif self.data_frequency == 'monthly':
                kp_range = 365  # ±12 months
                num_kp = 13
            else:
                kp_range = self.keypoint_range
                num_kp = self.n_keypoints
            
            keypoints = np.linspace(-kp_range, kp_range, num=num_kp).astype(np.float32)
            
            with tf.device('/CPU:0'):  # PWL runs on CPU
                holiday_pwl = tfl.layers.PWLCalibration(
                    input_keypoints=keypoints,
                    units=self.hidden_units,  # Output multiple calibrated features
                    output_min=-2.0,
                    output_max=2.0,
                    monotonicity='none',  # Flexible holiday effects
                    kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                    name='holiday_pwl'
                )(holiday_feature)
            # holiday_pwl shape: [batch, hidden_units]
            
            # Dense layer on PWL features (attention applied later in intermittent handler)
            holiday_hidden = Dense(
                self.hidden_units,
                activation=self.get_activation_fn(),
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                name='holiday_hidden'
            )(holiday_pwl)
        else:
            # Fallback: regular Dense layer
            holiday_hidden = Dense(
                self.hidden_units,
                activation=self.get_activation_fn(),
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                name='holiday_hidden'
            )(holiday_feature)
        
        holiday_hidden = Dropout(self.dropout)(holiday_hidden)
        
        # Shift-and-scale
        id_holiday_beta = Dense(
            self.hidden_units,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='id_holiday_beta'
        )(id_embedding)
        
        id_holiday_alpha = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='id_holiday_alpha'
        )(id_embedding)
        
        holiday_out = Multiply(name='holiday_scale')([holiday_hidden, id_holiday_beta])
        holiday_out = Add(name='holiday_shift')([holiday_out, id_holiday_alpha])
        
        holiday_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='holiday_forecast'
        )(holiday_out)
        
        return holiday_out, holiday_forecast


class RegressorComponentBuilder(ComponentBuilder):
    """Regressor Component."""
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Build regressor component."""
        regressor_features = Lambda(
            lambda x: tf.gather(x, feature_indices, axis=-1),
            name='regressor_features'
        )(main_input)
        
        regressor_hidden = Dense(
            self.hidden_units,
            activation=self.get_activation_fn(),
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='regressor_hidden'
        )(regressor_features)
        regressor_hidden = Dropout(self.dropout)(regressor_hidden)
        
        # Simple additive residual
        id_regressor_residual = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='id_regressor_residual'
        )(id_embedding)
        
        regressor_out = Add(name='regressor_combined')([regressor_hidden, id_regressor_residual])
        
        regressor_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='regressor_forecast'
        )(regressor_out)
        
        return regressor_out, regressor_forecast


class HierarchicalAttentionIntermittentHandler(ComponentBuilder):
    """
    Intermittent Handler with Hierarchical Attention.
    
    Two-level attention:
    1. Component-level: Learn importance of trend/seasonal/holiday/regressor
    2. Feature-level: Within each component, learn importance of features
    """
    
    def __init__(
        self,
        hidden_units: int = 64,
        hidden_layers: int = 2,
        activation: str = 'mish',
        dropout: float = 0.2,
        component_hidden_units: int = 32  # NEW: component output dimensions
    ):
        super().__init__(hidden_units, activation, dropout)
        self.hidden_layers = hidden_layers
        self.component_hidden_units = component_hidden_units
    
    def build(
        self,
        component_outputs: List[tf.Tensor],
        id_embedding: tf.Tensor,
        base_forecast: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build intermittent handler with hierarchical attention.
        
        Args:
            component_outputs: List of [trend_out, seasonal_out, holiday_out, regressor_out]
            id_embedding: SKU embedding tensor
            base_forecast: Additive combination of component forecasts
            
        Returns:
            (zero_probability, final_forecast)
        """
        num_components = len(component_outputs)
        component_names = ['trend', 'seasonal', 'holiday', 'regressor']
        
        # LEVEL 1: Component-level attention
        # Learn which components matter for zero probability prediction
        component_attention_logits = Dense(
            num_components,
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='component_attention_logits'
        )(id_embedding)
        
        component_weights = Entmax15(name='component_attention')(component_attention_logits)
        # component_weights: [batch, 4] - sparse weights summing to ~1
        
        # LEVEL 2: Feature-level attention within each component
        weighted_components = []
        
        for i, (comp, name) in enumerate(zip(component_outputs, component_names)):
            # Component-level weight (scalar for entire component)
            comp_weight = Lambda(
                lambda x: tf.expand_dims(x[:, i], -1),
                name=f'{name}_component_weight'
            )(component_weights)  # [batch, 1]
            
            # Feature-level attention WITHIN this component
            # Learn which features within this component matter
            feature_attention_logits = Dense(
                self.component_hidden_units,  # Match component output dimensions
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                name=f'{name}_feature_attention_logits'
            )(comp)
            
            feature_weights = Entmax15(name=f'{name}_feature_attention')(feature_attention_logits)
            # feature_weights: [batch, hidden_units] - sparse weights per feature
            
            # Apply BOTH levels of attention:
            # Step 1: Feature-level masking (which features within component)
            feature_masked = Multiply(name=f'{name}_feature_mask')([comp, feature_weights])
            
            # Step 2: Component-level scaling (how important is this component)
            component_scaled = Multiply(name=f'{name}_component_scale')([feature_masked, comp_weight])
            
            weighted_components.append(component_scaled)
        
        # Concatenate all hierarchically-weighted components
        combined_features = Concatenate(name='hierarchical_combined')(weighted_components)
        
        # Zero probability network
        zero_hidden = combined_features
        for i in range(self.hidden_layers):
            zero_hidden = Dense(
                self.hidden_units,
                activation=self.get_activation_fn(),
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                name=f'zero_prob_hidden_{i+1}'
            )(zero_hidden)
            zero_hidden = Dropout(self.dropout)(zero_hidden)
        
        # Zero probability output
        zero_probability = Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='zero_probability'
        )(zero_hidden)
        
        # Final forecast: base_forecast × (1 - zero_probability)
        final_forecast = Lambda(
            lambda x: x[0] * (1 - x[1]),
            name='final_forecast'
        )([base_forecast, zero_probability])
        
        return zero_probability, final_forecast


class DeepSequencePWLHierarchical:
    """
    DeepSequence PWL with Hierarchical Attention.
    
    Architecture:
    - Components: Trend, Seasonal, Holiday, Regressor (with shift-and-scale)
    - Hierarchical Attention: Component-level + Feature-level
    - Intermittent Handler: Two-stage with hierarchical attention
    """
    
    def __init__(
        self,
        num_skus: int,
        n_features: int,
        enable_intermittent_handling: bool = True,
        id_embedding_dim: int = 8,
        component_hidden_units: int = 32,
        component_dropout: float = 0.2,
        zero_prob_hidden_units: int = 64,
        zero_prob_hidden_layers: int = 2,
        zero_prob_dropout: float = 0.2,
        activation: str = 'mish',
        data_frequency: str = 'daily'
    ):
        self.num_skus = num_skus
        self.n_features = n_features
        self.enable_intermittent_handling = enable_intermittent_handling
        self.id_embedding_dim = id_embedding_dim
        self.component_hidden_units = component_hidden_units
        self.component_dropout = component_dropout
        self.zero_prob_hidden_units = zero_prob_hidden_units
        self.zero_prob_hidden_layers = zero_prob_hidden_layers
        self.zero_prob_dropout = zero_prob_dropout
        self.activation = activation
        self.data_frequency = data_frequency
        
        # Initialize component builders
        self.trend_builder = TrendComponentBuilder(
            hidden_units=component_hidden_units,
            activation=activation,
            dropout=component_dropout
        )
        self.seasonal_builder = SeasonalComponentBuilder(
            hidden_units=component_hidden_units,
            activation=activation,
            dropout=component_dropout
        )
        self.holiday_builder = HolidayComponentBuilder(
            hidden_units=component_hidden_units,
            activation=activation,
            dropout=component_dropout
        )
        self.regressor_builder = RegressorComponentBuilder(
            hidden_units=component_hidden_units,
            activation=activation,
            dropout=component_dropout
        )
        self.intermittent_builder = HierarchicalAttentionIntermittentHandler(
            hidden_units=zero_prob_hidden_units,
            hidden_layers=zero_prob_hidden_layers,
            activation=activation,
            dropout=zero_prob_dropout,
            component_hidden_units=component_hidden_units
        )
    
    def build_model(
        self,
        trend_feature_indices: Optional[List[int]] = None,
        seasonal_feature_indices: Optional[List[int]] = None,
        holiday_feature_index: Optional[int] = None,
        regressor_feature_indices: Optional[List[int]] = None
    ) -> Tuple[Model, Model, Model, Model, Model]:
        """
        Build the complete model with hierarchical attention.
        
        Returns:
            (main_model, trend_model, seasonal_model, holiday_model, regressor_model)
        """
        # Input layers
        main_input = Input(shape=(self.n_features,), name='main_input')
        sku_input = Input(shape=(1,), dtype='int32', name='sku_input')
        
        # SKU embedding
        id_embedding = Embedding(
            input_dim=self.num_skus,
            output_dim=self.id_embedding_dim,
            embeddings_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='sku_embedding'
        )(sku_input)
        id_embedding = Lambda(lambda x: tf.squeeze(x, axis=1))(id_embedding)
        
        # Build components
        trend_out, trend_forecast = self.trend_builder.build(
            main_input, id_embedding, trend_feature_indices or []
        )
        
        seasonal_out, seasonal_forecast = self.seasonal_builder.build(
            main_input, id_embedding, seasonal_feature_indices or []
        )
        
        holiday_out, holiday_forecast = self.holiday_builder.build(
            main_input, id_embedding, holiday_feature_index or 0
        )
        
        regressor_out, regressor_forecast = self.regressor_builder.build(
            main_input, id_embedding, regressor_feature_indices or []
        )
        
        # Additive combination
        base_forecast = Add(name='base_forecast')([
            trend_forecast,
            seasonal_forecast,
            holiday_forecast,
            regressor_forecast
        ])
        
        # Intermittent handling with hierarchical attention
        if self.enable_intermittent_handling:
            zero_probability, final_forecast = self.intermittent_builder.build(
                [trend_out, seasonal_out, holiday_out, regressor_out],
                id_embedding,
                base_forecast
            )
            
            outputs = {
                'base_forecast': base_forecast,
                'final_forecast': final_forecast,
                'zero_probability': zero_probability
            }
        else:
            outputs = {
                'base_forecast': base_forecast,
                'final_forecast': base_forecast
            }
        
        # Build main model
        self.model = Model(
            inputs=[main_input, sku_input],
            outputs=outputs,
            name='DeepSequencePWL_Hierarchical'
        )
        
        # Build component sub-models
        self.trend_model = Model(
            inputs=[main_input, sku_input],
            outputs=trend_forecast,
            name='TrendComponent'
        )
        
        self.seasonal_model = Model(
            inputs=[main_input, sku_input],
            outputs=seasonal_forecast,
            name='SeasonalComponent'
        )
        
        self.holiday_model = Model(
            inputs=[main_input, sku_input],
            outputs=holiday_forecast,
            name='HolidayComponent'
        )
        
        self.regressor_model = Model(
            inputs=[main_input, sku_input],
            outputs=regressor_forecast,
            name='RegressorComponent'
        )
        
        return (
            self.model,
            self.trend_model,
            self.seasonal_model,
            self.holiday_model,
            self.regressor_model
        )
    
    def get_model(self) -> Model:
        """Get the main model."""
        if not hasattr(self, 'model'):
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model
