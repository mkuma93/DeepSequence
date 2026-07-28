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
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Concatenate, Add, Multiply, Lambda, Dropout, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import UnitNorm
from typing import List, Tuple, Optional

# Import TabNet
from .tabnet import TabNetEncoder

# Try importing TensorFlow Recommenders for DCN layers
try:
    import tensorflow_recommenders as tfrs
    TFRS_AVAILABLE = True
except ImportError:
    TFRS_AVAILABLE = False
    print("Warning: tensorflow_recommenders not available. Cross layers disabled.")


# Use TFRS Cross layer if available, otherwise use simple fallback
if TFRS_AVAILABLE:
    CrossLayer = tfrs.layers.dcn.Cross
else:
    # Fallback implementation if TFRS not available
    class CrossLayer(tf.keras.layers.Layer):
        """Fallback Cross layer implementation."""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._x0 = None
        
        def build(self, input_shape):
            self.dim = input_shape[-1]
            self.w = self.add_weight(
                name='cross_weight',
                shape=(self.dim,),
                initializer='glorot_normal',
                trainable=True
            )
            self.b = self.add_weight(
                name='cross_bias',
                shape=(self.dim,),
                initializer='zeros',
                trainable=True
            )
            super().build(input_shape)
        
        def call(self, x):
            if self._x0 is None:
                self._x0 = x
            x_w = tf.reduce_sum(x * self.w, axis=-1, keepdims=True)
            x0_x_w = self._x0 * x_w
            output = x0_x_w + self.b + x
            return output
        
        def get_config(self):
            config = super().get_config()
            return config


class SparseAttention(tf.keras.layers.Layer):
    """
    Sparse attention mechanism using low-temperature softmax.
    
    Instead of complex entmax/sparsemax with numerical stability issues,
    we use softmax with low temperature (high sharpness) to achieve sparsity.
    
    This is numerically stable and differentiable, producing concentrated
    attention distributions similar to sparse attention but without NaN issues.
    
    Temperature τ controls sparsity:
    - τ → 0: very sparse (approaches one-hot)
    - τ = 1: standard softmax
    - τ → ∞: uniform distribution
    """
    
    def __init__(self, axis=-1, temperature=0.5, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.temperature = temperature
    
    def call(self, inputs):
        """Apply temperature-scaled softmax for sparse attention."""
        # Scale by temperature (lower temp = sharper/sparser)
        scaled_inputs = inputs / self.temperature
        
        # Subtract max for numerical stability (prevents overflow)
        scaled_inputs = scaled_inputs - tf.reduce_max(
            scaled_inputs, axis=self.axis, keepdims=True
        )
        
        # Apply softmax
        exp_inputs = tf.exp(scaled_inputs)
        sum_exp = tf.reduce_sum(exp_inputs, axis=self.axis, keepdims=True)
        
        # Prevent division by zero
        sum_exp = tf.maximum(sum_exp, 1e-10)
        
        output = exp_inputs / sum_exp
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'temperature': self.temperature
        })
        return config


class ComponentBuilder:
    """Base class for component builders."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_unit_norm: bool = True
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_unit_norm = use_unit_norm
        self.dropout = dropout
    
    def get_kernel_constraint(self):
        """Get kernel constraint based on use_unit_norm flag."""
        return UnitNorm(axis=0) if self.use_unit_norm else None
    
    def get_activation_fn(self):
        """Get activation function."""
        if self.activation == 'mish':
            return lambda x: x * tf.nn.tanh(tf.nn.softplus(x))
        return self.activation


class ChangepointReLU(tf.keras.layers.Layer):
    """Vectorised ReLU transform with learnable changepoints and L1/L2 penalties on the outputs."""

    def __init__(
        self,
        changepoints,
        l1: float = 1e-4,
        l2: float = 1e-4,
        learnable: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._initial_changepoints = np.asarray(changepoints, dtype=np.float32)
        self.n_changepoints = len(self._initial_changepoints)
        self.l1 = l1
        self.l2 = l2
        self.learnable = learnable
        if (l1 or 0.0) > 0.0 or (l2 or 0.0) > 0.0:
            self._regularizer = regularizers.l1_l2(
                l1=l1 or 0.0,
                l2=l2 or 0.0
            )
        else:
            self._regularizer = None
    
    def build(self, input_shape):
        if self.learnable:
            # Create trainable changepoints initialized from provided values
            self.changepoints = self.add_weight(
                name='changepoints',
                shape=(self.n_changepoints,),
                initializer=tf.keras.initializers.Constant(self._initial_changepoints),
                trainable=True,
                dtype=tf.float32
            )
        else:
            # Fixed changepoints (original behavior)
            self.changepoints = tf.constant(self._initial_changepoints, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        cp = tf.reshape(self.changepoints, (1, -1))
        relu_features = tf.nn.relu(inputs - cp)
        if self._regularizer is not None:
            self.add_loss(self._regularizer(relu_features))
        return relu_features

    def get_config(self):
        config = super().get_config()
        config.update({
            'changepoints': self._initial_changepoints.tolist(),
            'l1': self.l1,
            'l2': self.l2,
            'learnable': self.learnable
        })
        return config


class TrendComponentBuilder(ComponentBuilder):
    """Trend component with shift-and-scale for SKU-specific effects."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_attention: bool = True,
        n_changepoints: int = 10,
        time_min: float = None,
        time_max: float = None
    ):
        super().__init__(hidden_units, activation, dropout)
        self.use_attention = use_attention
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
        
        # Dense transform on input time feature
        trend_hidden = Dense(
            self.hidden_units,
            activation=self.get_activation_fn(),
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='trend_hidden'
        )(time_feature)

        # Apply sparse attention on input features if enabled
        if self.use_attention:
            attention_scores = Dense(
                self.hidden_units,
                activation='linear',
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
                name='trend_attention_scores'
            )(trend_hidden)

            attention_weights = SparseAttention(
                name='trend_attention'
            )(attention_scores)
            trend_hidden = Multiply(name='trend_attended')(
                [trend_hidden, attention_weights]
            )
        
        trend_hidden = Dropout(self.dropout)(trend_hidden)
        
        # Shift-and-scale using ID embedding
        id_trend_beta = Dense(
            self.hidden_units,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_trend_beta'
        )(id_embedding)
        
        id_trend_alpha = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
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
            kernel_constraint=self.get_kernel_constraint(),
            name='trend_forecast'
        )(trend_out)
        
        return trend_out, trend_forecast


class SeasonalComponentBuilder(ComponentBuilder):
    """Seasonal Component with Entmax attention and shift-and-scale."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__(hidden_units, activation, dropout)
        self.use_attention = use_attention
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Build seasonal component."""
        if not feature_indices:
            # No seasonal features - return zeros
            batch_size = tf.shape(main_input)[0]
            seasonal_out = tf.zeros((batch_size, self.hidden_units))
            seasonal_forecast = tf.zeros((batch_size, 1))
            return seasonal_out, seasonal_forecast
        
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
        
        # Apply Entmax attention on input features if enabled
        if self.use_attention:
            # LayerNorm before attention for stability
            seasonal_hidden_norm = LayerNormalization(name='seasonal_attention_prenorm')(seasonal_hidden)
            
            attention_scores = Dense(
                self.hidden_units,
                activation='linear',
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
                name='seasonal_attention_scores'
            )(seasonal_hidden_norm)
            
            attention_weights = SparseAttention(name='seasonal_attention')(attention_scores)
            seasonal_hidden = Multiply(name='seasonal_attended')([seasonal_hidden, attention_weights])
        
        seasonal_hidden = Dropout(self.dropout)(seasonal_hidden)
        
        # Shift-and-scale
        id_seasonal_beta = Dense(
            self.hidden_units,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_seasonal_beta'
        )(id_embedding)
        
        id_seasonal_alpha = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_seasonal_alpha'
        )(id_embedding)
        
        seasonal_out = Multiply(name='seasonal_scale')([seasonal_hidden, id_seasonal_beta])
        seasonal_out = Add(name='seasonal_shift')([seasonal_out, id_seasonal_alpha])
        
        seasonal_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='seasonal_forecast'
        )(seasonal_out)
        
        return seasonal_out, seasonal_forecast


class HolidayComponentBuilder(ComponentBuilder):
    """Holiday Component with PWL calibration (for distance features) or direct Dense (for binary features) and Entmax attention."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_attention: bool = True,
        n_keypoints: int = 37,
        keypoint_range: float = 365.0,
        data_frequency: str = 'daily'
    ):
        super().__init__(hidden_units, activation, dropout)
        self.use_attention = use_attention
        self.n_keypoints = n_keypoints
        self.keypoint_range = keypoint_range
        self.data_frequency = data_frequency
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build holiday component with multiple holiday features and Entmax attention.
        
        Args:
            main_input: Full feature tensor
            id_embedding: SKU embedding
            feature_indices: List of holiday feature indices
        
        Returns:
            (holiday_hidden, holiday_forecast)
        """
        if not feature_indices:
            # No holiday features - return zeros
            batch_size = tf.shape(main_input)[0]
            holiday_out = tf.zeros((batch_size, self.hidden_units))
            holiday_forecast = tf.zeros((batch_size, 1))
            return holiday_out, holiday_forecast
        
        holiday_features = Lambda(
            lambda x: tf.gather(x, feature_indices, axis=-1),
            name='holiday_features'
        )(main_input)
        
        # Dense projection on raw holiday features
        holiday_hidden = Dense(
            self.hidden_units,
            activation=self.get_activation_fn(),
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='holiday_hidden'
        )(holiday_features)

        # Apply sparse attention on holiday signals if enabled
        if self.use_attention:
            attention_scores = Dense(
                self.hidden_units,
                activation='linear',
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
                name='holiday_attention_scores'
            )(holiday_hidden)

            attention_weights = SparseAttention(
                name='holiday_attention'
            )(attention_scores)
            holiday_hidden = Multiply(name='holiday_attended')(
                [holiday_hidden, attention_weights]
            )
        
        holiday_hidden = Dropout(self.dropout)(holiday_hidden)
        
        # Shift-and-scale
        id_holiday_beta = Dense(
            self.hidden_units,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_holiday_beta'
        )(id_embedding)
        
        id_holiday_alpha = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_holiday_alpha'
        )(id_embedding)
        
        holiday_out = Multiply(name='holiday_scale')([holiday_hidden, id_holiday_beta])
        holiday_out = Add(name='holiday_shift')([holiday_out, id_holiday_alpha])
        
        holiday_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='holiday_forecast'
        )(holiday_out)
        
        return holiday_out, holiday_forecast


class RegressorComponentBuilder(ComponentBuilder):
    """Regressor Component with Entmax attention."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__(hidden_units, activation, dropout)
        self.use_attention = use_attention
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Build regressor component."""
        if not feature_indices:
            # No regressor features - return zeros
            batch_size = tf.shape(main_input)[0]
            regressor_out = tf.zeros((batch_size, self.hidden_units))
            regressor_forecast = tf.zeros((batch_size, 1))
            return regressor_out, regressor_forecast
        
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
        
        # Apply Entmax attention on input features if enabled
        if self.use_attention:
            # LayerNorm before attention for stability
            regressor_hidden_norm = LayerNormalization(name='regressor_attention_prenorm')(regressor_hidden)
            
            attention_scores = Dense(
                self.hidden_units,
                activation='linear',
                kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
                name='regressor_attention_scores'
            )(regressor_hidden_norm)
            
            attention_weights = SparseAttention(name='regressor_attention')(attention_scores)
            regressor_hidden = Multiply(name='regressor_attended')([regressor_hidden, attention_weights])
        
        regressor_hidden = Dropout(self.dropout)(regressor_hidden)
        
        # Simple additive residual
        id_regressor_residual = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_regressor_residual'
        )(id_embedding)
        
        regressor_out = Add(name='regressor_combined')([regressor_hidden, id_regressor_residual])
        
        regressor_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='regressor_forecast'
        )(regressor_out)
        
        return regressor_out, regressor_forecast


# ============================================================================
# TabNet-based Component Builders
# ============================================================================

class TrendComponentBuilderSimple(ComponentBuilder):
    """
    Trend Component with changepoints + TabNet.
    
    Creates piecewise linear features using ReLU at predefined changepoints,
    then applies TabNet to select relevant changepoints per SKU.
    
    Approach:
    1. Create N changepoints via np.linspace over time range
    2. Generate ReLU features: max(0, time - changepoint) for each
    3. Apply TabNet to select which changepoints matter
    """
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_unit_norm: bool = True,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        time_min: float = None,
        time_max: float = None,
        tabnet_feature_dim: int = 32,
        tabnet_steps: int = 3,
        virtual_batch_size: int = 128,
        changepoint_regularizer_l1: float = 1e-4,
        changepoint_regularizer_l2: float = 1e-4
    ):
        super().__init__(hidden_units, activation, dropout, use_unit_norm)
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.time_min = time_min
        self.time_max = time_max
        self.tabnet_feature_dim = tabnet_feature_dim
        self.tabnet_steps = tabnet_steps
        self.virtual_batch_size = virtual_batch_size
        self.changepoint_reg_l1 = changepoint_regularizer_l1
        self.changepoint_reg_l2 = changepoint_regularizer_l2
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build trend component with changepoint ReLU features + TabNet.
        
        Args:
            main_input: Full feature tensor
            id_embedding: SKU embedding
            feature_indices: Should contain just the time feature index
        
        Returns:
            (trend_hidden, trend_forecast)
        """
        # Extract time feature (numeric: days since epoch)
        time_feature = Lambda(
            lambda x: tf.gather(x, feature_indices, axis=-1),
            name='time_feature'
        )(main_input)
        
        # Create changepoints using np.linspace
        if self.time_min is not None and self.time_max is not None:
            # Use changepoint_range (e.g., 0.8 = first 80% of time range)
            time_range = self.time_max - self.time_min
            cp_start = self.time_min
            cp_end = self.time_min + (time_range * self.changepoint_range)
            
            changepoints = np.linspace(
                cp_start, cp_end, num=self.n_changepoints
            ).astype(np.float32)
        else:
            # Fallback: create normalized changepoints
            changepoints = np.linspace(
                0.0, 1.0, num=self.n_changepoints
            ).astype(np.float32)
        
        # Create ReLU features for all changepoints in one vectorized operation
        # ReLU(time - cp) creates piecewise linear segment after cp
        # Shape: [batch, 1] - [n_changepoints] -> [batch, n_changepoints]
        changepoint_features = ChangepointReLU(
            changepoints=changepoints,
            l1=self.changepoint_reg_l1,
            l2=self.changepoint_reg_l2,
            name='trend_changepoint_relu'
        )(time_feature)
        
        # Apply TabNet to select relevant changepoints
        trend_hidden = TabNetEncoder(
            feature_dim=self.tabnet_feature_dim,
            output_dim=self.hidden_units,
            num_steps=self.tabnet_steps,
            virtual_batch_size=self.virtual_batch_size,
            name='trend_tabnet'
        )(changepoint_features)
        
        # Apply CrossLayer for feature interactions, then normalize
        trend_hidden = CrossLayer(name='trend_cross_layer')(trend_hidden)
        trend_hidden = LayerNormalization(name='trend_layer_norm')(trend_hidden)
        trend_hidden = Dropout(self.dropout)(trend_hidden)
        
        # Shift-and-scale using ID embedding
        id_trend_beta = Dense(
            self.hidden_units,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_trend_beta'
        )(id_embedding)
        
        id_trend_alpha = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
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
            kernel_constraint=self.get_kernel_constraint(),
            name='trend_forecast'
        )(trend_out)
        
        return trend_out, trend_forecast


class SeasonalComponentBuilderTabNet(ComponentBuilder):
    """Seasonal Component using TabNet for feature selection."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_unit_norm: bool = True,
        tabnet_feature_dim: int = None,  # Auto-scale if None
        tabnet_steps: int = 3,
        virtual_batch_size: int = 128
    ):
        super().__init__(hidden_units, activation, dropout, use_unit_norm)
        self.tabnet_feature_dim = tabnet_feature_dim
        self.tabnet_steps = tabnet_steps
        self.virtual_batch_size = virtual_batch_size
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Build seasonal component with TabNet."""
        if not feature_indices:
            # No seasonal features - return zeros
            batch_size = tf.shape(main_input)[0]
            seasonal_out = tf.zeros((batch_size, self.hidden_units))
            seasonal_forecast = tf.zeros((batch_size, 1))
            return seasonal_out, seasonal_forecast
        
        seasonal_features = Lambda(
            lambda x: tf.gather(x, feature_indices, axis=-1),
            name='seasonal_features'
        )(main_input)
        
        # Check for NaN in Fourier features (common issue with date calculations)
        seasonal_features = Lambda(
            lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x),
            name='seasonal_nan_check'
        )(seasonal_features)
        
        # Scale TabNet params based on input dimension
        n_features = len(feature_indices)
        feature_dim = (
            self.tabnet_feature_dim if self.tabnet_feature_dim
            else max(8, min(n_features, 32))
        )
        
        # Apply TabNet: replaces Dense + Attention
        seasonal_hidden = TabNetEncoder(
            feature_dim=feature_dim,
            output_dim=self.hidden_units,
            num_steps=self.tabnet_steps,
            virtual_batch_size=self.virtual_batch_size,
            name='seasonal_tabnet'
        )(seasonal_features)
        
        # Apply CrossLayer for feature interactions, then normalize
        seasonal_hidden = CrossLayer(name='seasonal_cross_layer')(seasonal_hidden)
        seasonal_hidden = LayerNormalization(name='seasonal_layer_norm')(seasonal_hidden)
        seasonal_hidden = Dropout(self.dropout)(seasonal_hidden)
        
        # Shift-and-scale with ID embedding
        id_seasonal_beta = Dense(
            self.hidden_units,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_seasonal_beta'
        )(id_embedding)
        
        id_seasonal_alpha = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_seasonal_alpha'
        )(id_embedding)
        
        seasonal_out = Multiply(name='seasonal_scale')([seasonal_hidden, id_seasonal_beta])
        seasonal_out = Add(name='seasonal_shift')([seasonal_out, id_seasonal_alpha])
        
        seasonal_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='seasonal_forecast'
        )(seasonal_out)
        
        return seasonal_out, seasonal_forecast


class HolidayComponentBuilderTabNet(ComponentBuilder):
    """Holiday Component using two-level TabNet with per-holiday changepoints.
    
    Level 1 (Per-Holiday): For each holiday, creates changepoints on distance
                          (focused on ±7 days) → ReLU features → Small TabNet → Signal
    Level 2 (Cross-Holiday): Concatenates all holiday signals → TabNet → Final encoding
    
    This captures non-linear distance effects per holiday and learns which holidays
    interact or dominate the forecast.
    """
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        use_unit_norm: bool = True,
        n_changepoints_per_holiday: int = 7,
        changepoint_range_days: float = 14.0,
        per_holiday_output_dim: int = 8,
        per_holiday_tabnet_steps: int = 2,
        cross_holiday_tabnet_steps: int = 3,
        tabnet_feature_dim: int = None,  # Auto-scale if None
        virtual_batch_size: int = 128,
        changepoint_regularizer_l1: float = 1e-4,
        changepoint_regularizer_l2: float = 1e-4
    ):
        """Initialize Holiday Component Builder with two-level TabNet.
        
        Args:
            hidden_units: Hidden units for final output
            activation: Activation function (not used with TabNet)
            dropout: Dropout rate
            use_unit_norm: Apply UnitNorm constraint to Dense layer kernels
            n_changepoints_per_holiday: Changepoints per holiday (default: 7 for ±7 days)
            changepoint_range_days: Distance range for changepoints in days (default: 14 = ±7 days)
            per_holiday_output_dim: Output dimension per holiday signal (default: 8)
            per_holiday_tabnet_steps: TabNet steps per holiday (default: 2)
            cross_holiday_tabnet_steps: TabNet steps for cross-holiday (default: 3)
            tabnet_feature_dim: Feature dimension for TabNet encoders
            virtual_batch_size: Virtual batch size for Ghost BN
        """
        super().__init__(hidden_units, activation, dropout, use_unit_norm)
        self.n_changepoints = n_changepoints_per_holiday
        self.changepoint_range = changepoint_range_days
        self.per_holiday_dim = per_holiday_output_dim
        self.per_holiday_steps = per_holiday_tabnet_steps
        self.cross_holiday_steps = cross_holiday_tabnet_steps
        self.tabnet_feature_dim = tabnet_feature_dim
        self.virtual_batch_size = virtual_batch_size
        self.changepoint_reg_l1 = changepoint_regularizer_l1
        self.changepoint_reg_l2 = changepoint_regularizer_l2
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Build two-level holiday component with changepoint-based TabNet.
        
        Args:
            main_input: Main input tensor [batch, features]
            id_embedding: ID embedding tensor [batch, embedding_dim]
            feature_indices: Indices for holiday distance features (15 holidays)
            
        Returns:
            tuple: (holiday_hidden [batch, hidden_units], holiday_forecast [batch, 1])
        """
        if not feature_indices:
            # No holiday features - return zeros
            batch_size = tf.shape(main_input)[0]
            holiday_out = tf.zeros((batch_size, self.hidden_units))
            holiday_forecast = tf.zeros((batch_size, 1))
            return holiday_out, holiday_forecast
        
        # LEVEL 1: Per-holiday changepoint processing
        holiday_signals = []
        
        # Create changepoints: focus on ±7 days (most relevant range)
        # Symmetric range around holiday: -7, -5, -3, -1, 0, +1, +3, +5, +7 days
        changepoints = np.linspace(
            -self.changepoint_range / 2.0,
            self.changepoint_range / 2.0,
            num=self.n_changepoints
        ).astype(np.float32)
        
        for i, feature_idx in enumerate(feature_indices):
            # Extract distance for this holiday
            holiday_distance = Lambda(
                lambda x: tf.gather(x, [feature_idx], axis=-1),
                name=f'holiday_h{i}_extract'
            )(main_input)
            
            # Check for NaN in holiday distance features
            holiday_distance = Lambda(
                lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x),
                name=f'holiday_h{i}_nan_check'
            )(holiday_distance)
            
            # Create ReLU changepoint features in one vectorized operation
            # Shape: [batch, 1] - [n_changepoints] -> [batch, n_changepoints]
            holiday_changepoint_features = ChangepointReLU(
                changepoints=changepoints,
                l1=self.changepoint_reg_l1,
                l2=self.changepoint_reg_l2,
                name=f'holiday_h{i}_changepoint_relu'
            )(holiday_distance)
            
            # Auto-scale per-holiday TabNet based on changepoints
            per_holiday_feat_dim = (
                self.tabnet_feature_dim if self.tabnet_feature_dim 
                else max(8, min(self.n_changepoints, 24))
            )
            
            # Apply small TabNet to select relevant distance ranges for this holiday
            holiday_signal = TabNetEncoder(
                feature_dim=per_holiday_feat_dim,
                output_dim=self.per_holiday_dim,
                num_steps=self.per_holiday_steps,
                virtual_batch_size=self.virtual_batch_size,
                name=f'holiday_h{i}_tabnet'
            )(holiday_changepoint_features)
            
            holiday_signals.append(holiday_signal)
        
        # LEVEL 2: Cross-holiday TabNet
        # Concatenate all holiday signals [batch, num_holidays * per_holiday_dim]
        all_holiday_signals = Concatenate(
            name='holiday_all_signals'
        )(holiday_signals)
        
        # Auto-scale cross-holiday TabNet based on total signal size
        n_holidays = len(feature_indices)
        cross_signal_dim = n_holidays * self.per_holiday_dim
        cross_feat_dim = (
            self.tabnet_feature_dim if self.tabnet_feature_dim 
            else max(16, min(cross_signal_dim, 32))
        )
        
        # Apply TabNet to learn cross-holiday interactions
        holiday_hidden = TabNetEncoder(
            feature_dim=cross_feat_dim,
            output_dim=self.hidden_units,
            num_steps=self.cross_holiday_steps,
            virtual_batch_size=self.virtual_batch_size,
            name='holiday_cross_tabnet'
        )(all_holiday_signals)
        
        # Apply CrossLayer for feature interactions, then normalize
        holiday_hidden = CrossLayer(name='holiday_cross_layer')(holiday_hidden)
        holiday_hidden = LayerNormalization(name='holiday_layer_norm')(holiday_hidden)
        holiday_hidden = Dropout(self.dropout)(holiday_hidden)
        
        # Shift-and-scale
        id_holiday_beta = Dense(
            self.hidden_units,
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_holiday_beta'
        )(id_embedding)
        
        id_holiday_alpha = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_holiday_alpha'
        )(id_embedding)
        
        holiday_out = Multiply(name='holiday_scale')([holiday_hidden, id_holiday_beta])
        holiday_out = Add(name='holiday_shift')([holiday_out, id_holiday_alpha])
        
        holiday_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='holiday_forecast'
        )(holiday_out)
        
        return holiday_out, holiday_forecast


class RegressorComponentBuilderTabNet(ComponentBuilder):
    """Regressor Component using TabNet for feature selection."""
    
    def __init__(
        self,
        hidden_units: int = 32,
        activation: str = 'mish',
        dropout: float = 0.2,
        tabnet_feature_dim: int = 32,
        tabnet_steps: int = 3,
        virtual_batch_size: int = 128
    ):
        super().__init__(hidden_units, activation, dropout)
        self.tabnet_feature_dim = tabnet_feature_dim
        self.tabnet_steps = tabnet_steps
        self.virtual_batch_size = virtual_batch_size
    
    def build(
        self,
        main_input: tf.Tensor,
        id_embedding: tf.Tensor,
        feature_indices: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Build regressor component with TabNet."""
        if not feature_indices:
            # No regressor features - return zeros
            batch_size = tf.shape(main_input)[0]
            regressor_out = tf.zeros((batch_size, self.hidden_units))
            regressor_forecast = tf.zeros((batch_size, 1))
            return regressor_out, regressor_forecast
        
        regressor_features = Lambda(
            lambda x: tf.gather(x, feature_indices, axis=-1),
            name='regressor_features'
        )(main_input)
        
        # Apply TabNet: replaces Dense + Attention
        regressor_hidden = TabNetEncoder(
            feature_dim=self.tabnet_feature_dim,
            output_dim =self.hidden_units,
            num_steps=self.tabnet_steps,
            virtual_batch_size=self.virtual_batch_size,
            name='regressor_tabnet'
        )(regressor_features)
        
        regressor_hidden = Dropout(self.dropout)(regressor_hidden)
        
        # Simple additive residual with ID embedding
        id_regressor_residual = Dense(
            self.hidden_units,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
            name='id_regressor_residual'
        )(id_embedding)
        
        regressor_out = Add(name='regressor_combined')([regressor_hidden, id_regressor_residual])
        
        regressor_forecast = Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            kernel_constraint=self.get_kernel_constraint(),
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
        component_hidden_units: int = 32,  # NEW: component output dimensions
        num_cross_layers: int = 2  # DCN cross layers
    ):
        super().__init__(hidden_units, activation, dropout)
        self.hidden_layers = hidden_layers
        self.component_hidden_units = component_hidden_units
        self.num_cross_layers = num_cross_layers
    
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
            kernel_constraint=self.get_kernel_constraint(),
            name='component_attention_logits'
        )(id_embedding)
        
        component_weights = SparseAttention(name='component_attention')(component_attention_logits)
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
            kernel_constraint=self.get_kernel_constraint(),
                name=f'{name}_feature_attention_logits'
            )(comp)
            
            feature_weights = SparseAttention(name=f'{name}_feature_attention')(feature_attention_logits)
            # feature_weights: [batch, hidden_units] - sparse weights per feature
            
            # Apply BOTH levels of attention:
            # Step 1: Feature-level masking (which features within component)
            feature_masked = Multiply(name=f'{name}_feature_mask')([comp, feature_weights])
            
            # Step 2: Component-level scaling (how important is this component)
            component_scaled = Multiply(name=f'{name}_component_scale')([feature_masked, comp_weight])
            
            weighted_components.append(component_scaled)
        
        # Concatenate all hierarchically-weighted components
        combined_features = Concatenate(name='hierarchical_combined')(weighted_components)
        
        # Apply DCN-style cross layers for explicit feature interactions
        if self.num_cross_layers > 0:
            x = combined_features
            
            # Apply cross layers (TFRS Cross automatically handles x0 storage)
            for i in range(self.num_cross_layers):
                x = CrossLayer(name=f'cross_layer_{i+1}')(x)
            
            # Use cross layer output for zero probability network
            zero_hidden = x
        else:
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
            kernel_constraint=self.get_kernel_constraint(),
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
        num_cross_layers: int = 2,
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
            dropout=component_dropout,
            use_attention=True,
            data_frequency=data_frequency
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
            component_hidden_units=component_hidden_units,
            num_cross_layers=num_cross_layers
        )
    
    def build_model(
        self,
        trend_feature_indices: Optional[List[int]] = None,
        seasonal_feature_indices: Optional[List[int]] = None,
        holiday_feature_indices: Optional[List[int]] = None,
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
        
        # Normalize input features for numerical stability
        # This ensures all features are on similar scale before processing
        main_input_normalized = LayerNormalization(name='input_normalization')(main_input)
        
        # SKU embedding
        id_embedding = Embedding(
            input_dim=self.num_skus,
            output_dim=self.id_embedding_dim,
            embeddings_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
            name='sku_embedding'
        )(sku_input)
        id_embedding = Lambda(lambda x: tf.squeeze(x, axis=1))(id_embedding)
        
        # Build components (use normalized inputs)
        trend_out, trend_forecast = self.trend_builder.build(
            main_input_normalized, id_embedding, trend_feature_indices or []
        )
        
        seasonal_out, seasonal_forecast = self.seasonal_builder.build(
            main_input_normalized, id_embedding, seasonal_feature_indices or []
        )
        
        holiday_out, holiday_forecast = self.holiday_builder.build(
            main_input_normalized, id_embedding, holiday_feature_indices or []
        )
        
        regressor_out, regressor_forecast = self.regressor_builder.build(
            main_input_normalized, id_embedding, regressor_feature_indices or []
        )
        
        # Learnable softmax-weighted combination of component forecasts
        # Each component gets a learned weight that sums to 1 (via softmax)
        component_weight_logits = Dense(
            4,  # 4 components: trend, seasonal, holiday, regressor
            name='component_weight_logits',
            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4)
        )(id_embedding)
        
        # Apply softmax with temperature for smooth weights
        component_weights = Lambda(
            lambda x: tf.nn.softmax(x / 0.5, axis=-1),  # Temperature = 0.5
            name='component_weights'
        )(component_weight_logits)
        
        # Extract individual weights
        trend_weight = Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1), name='trend_weight')(component_weights)
        seasonal_weight = Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1), name='seasonal_weight')(component_weights)
        holiday_weight = Lambda(lambda x: tf.expand_dims(x[:, 2], axis=-1), name='holiday_weight')(component_weights)
        regressor_weight = Lambda(lambda x: tf.expand_dims(x[:, 3], axis=-1), name='regressor_weight')(component_weights)
        
        # Weighted sum of component forecasts
        trend_weighted = Multiply(name='trend_forecast_weighted')([trend_forecast, trend_weight])
        seasonal_weighted = Multiply(name='seasonal_forecast_weighted')([seasonal_forecast, seasonal_weight])
        holiday_weighted = Multiply(name='holiday_forecast_weighted')([holiday_forecast, holiday_weight])
        regressor_weighted = Multiply(name='regressor_forecast_weighted')([regressor_forecast, regressor_weight])
        
        # Final additive combination (now weighted by softmax)
        base_forecast = Add(name='base_forecast')([
            trend_weighted,
            seasonal_weighted,
            holiday_weighted,
            regressor_weighted
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
