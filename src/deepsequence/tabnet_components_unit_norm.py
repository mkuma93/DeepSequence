"""
TabNet-based Components with Unit Normalization for DeepSequence.

Implements TabNet encoder for all components (Trend, Seasonal, Holiday, Regressor)
with unit normalization applied to each component output before combination.

Unit normalization ensures each component contributes proportionally and
prevents any single component from dominating the final prediction.

Components are combined using ADDITIVE mode (as in original DeepFuture):
    ŷ = norm(T) + norm(S) + norm(H) + norm(R)

Where norm(x) = x / ||x||₂ (L2 normalization)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class CrossLayer(layers.Layer):
    """
    Cross Network Layer for explicit feature crossing.
    
    Implements the cross layer from Deep & Cross Network (DCN):
        x_{l+1} = x_0 * x_l^T * w_l + b_l + x_l
    
    Where:
    - x_0 is the input to the first cross layer
    - x_l is the input to the current layer
    - w_l and b_l are trainable weights
    """
    
    def __init__(self, **kwargs):
        """Initialize cross layer."""
        super(CrossLayer, self).__init__(**kwargs)
        self.w = None
        self.b = None
    
    def build(self, input_shape):
        """Build layer weights."""
        # Handle both single input and list of inputs
        if isinstance(input_shape, list):
            # Use the shape of the second input (xl)
            shape = input_shape[1]
        else:
            shape = input_shape
        
        # Extract dimension, handling TensorShape objects
        if hasattr(shape, 'as_list'):
            dim = shape.as_list()[-1]
        else:
            dim = shape[-1]
        
        self.w = self.add_weight(
            name='cross_weight',
            shape=(dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='cross_bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Apply cross operation.
        
        Args:
            inputs: Tuple of (x0, xl) where x0 is the initial input
                    and xl is the current layer input
                    OR single tensor (in which case x0 = xl)
        
        Returns:
            Crossed output tensor
        """
        if isinstance(inputs, (list, tuple)):
            x0, xl = inputs
        else:
            x0 = xl = inputs
        
        # x_l^T * w_l
        xl_w = tf.matmul(xl, self.w)
        
        # x_0 * (x_l^T * w_l)
        x0_xl_w = x0 * xl_w
        
        # x_0 * (x_l^T * w_l) + b_l + x_l
        output = x0_xl_w + self.b + xl
        
        return output
    
    def get_config(self):
        config = super().get_config()
        return config


class UnitNormLayer(layers.Layer):
    """
    L2 Unit Normalization Layer.
    
    Normalizes the input tensor to have unit L2 norm:
        output = input / sqrt(sum(input²) + epsilon)
    
    This ensures each component output has equal weight in combination.
    """
    
    def __init__(self, epsilon=1e-12, **kwargs):
        """
        Initialize unit norm layer.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        super(UnitNormLayer, self).__init__(**kwargs)
        self.epsilon = epsilon
    
    def call(self, inputs):
        """
        Apply L2 normalization.
        
        Args:
            inputs: Input tensor of shape (batch_size, features)
            
        Returns:
            Unit-normalized tensor
        """
        # Calculate L2 norm: sqrt(sum(x^2))
        norm = tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)
        # Normalize: x / ||x||
        return inputs / norm
    
    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


class TabNetEncoder(layers.Layer):
    """
    TabNet Encoder for feature selection and representation learning.
    
    Based on "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, 2020).
    
    Key features:
    - Sequential attention mechanism for feature selection
    - Sparse feature masks for interpretability
    - Multi-step decision making
    """
    
    def __init__(self,
                 feature_dim=64,
                 output_dim=32,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 bn_momentum=0.98,
                 sparsity_coefficient=1e-5,
                 **kwargs):
        """
        Initialize TabNet encoder.
        
        Args:
            feature_dim: Dimension of feature transformation
            output_dim: Dimension of output representation
            num_decision_steps: Number of sequential attention steps
            relaxation_factor: Factor for feature mask relaxation
            bn_momentum: Batch normalization momentum
            sparsity_coefficient: L1 regularization on attention masks
        """
        super(TabNetEncoder, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.bn_momentum = bn_momentum
        self.sparsity_coefficient = sparsity_coefficient
        
        # Will be built in build()
        self.initial_bn = None
        self.feature_transforms = []
        self.attentive_transforms = []
        
    def build(self, input_shape):
        """Build layer components."""
        # Initial batch normalization
        self.initial_bn = layers.BatchNormalization(
            momentum=self.bn_momentum,
            name=f'{self.name}_initial_bn'
        )
        
        # Create feature transform and attention layers for each step
        for step in range(self.num_decision_steps):
            # Feature transformer
            self.feature_transforms.append(
                layers.Dense(
                    self.feature_dim,
                    activation='relu',
                    name=f'{self.name}_feat_transform_{step}'
                )
            )
            
            # Attentive transformer for mask generation
            self.attentive_transforms.append(
                layers.Dense(
                    input_shape[-1],
                    activation=None,
                    name=f'{self.name}_attn_transform_{step}'
                )
            )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass through TabNet encoder.
        
        Args:
            inputs: Input tensor (batch_size, features)
            training: Training mode flag
            
        Returns:
            Encoded representation tensor
        """
        # Initial normalization
        x = self.initial_bn(inputs, training=training)
        
        # Initialize prior scale and aggregated mask
        batch_size = tf.shape(x)[0]
        num_features = tf.shape(x)[1]
        prior_scales = tf.ones((batch_size, num_features))
        aggregated_output = tf.zeros((batch_size, self.output_dim))
        
        # Sequential decision steps
        for step in range(self.num_decision_steps):
            # Compute attention mask using prior
            mask_values = self.attentive_transforms[step](x)
            mask_values = mask_values * prior_scales
            
            # Sparsemax activation (approximate with softmax for simplicity)
            attention_mask = tf.nn.softmax(mask_values, axis=-1)
            
            # Apply mask to select features
            masked_features = x * attention_mask
            
            # Transform selected features
            transformed = self.feature_transforms[step](masked_features)
            
            # Accumulate output
            if step == 0:
                aggregated_output = transformed[:, :self.output_dim]
            else:
                aggregated_output += transformed[:, :self.output_dim]
            
            # Update prior scales (feature usage history)
            prior_scales = prior_scales * (self.relaxation_factor - attention_mask)
        
        return aggregated_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'num_decision_steps': self.num_decision_steps,
            'relaxation_factor': self.relaxation_factor,
            'bn_momentum': self.bn_momentum,
            'sparsity_coefficient': self.sparsity_coefficient
        })
        return config


class TabNetTrendComponent:
    """
    TabNet-based Trend Component with Unit Normalization.
    
    Captures long-term trends using TabNet architecture.
    Applies unit normalization to output before combination.
    """
    
    def __init__(self,
                 num_features,
                 id_embedding_dim=16,
                 tabnet_feature_dim=64,
                 tabnet_output_dim=32,
                 num_decision_steps=3,
                 num_cross_layers=2,
                 hidden_units=[64, 32],
                 dropout_rate=0.1):
        """
        Initialize TabNet trend component.
        
        Args:
            num_features: Number of input features
            id_embedding_dim: Dimension of ID embeddings
            tabnet_feature_dim: TabNet feature transformation dimension
            tabnet_output_dim: TabNet output dimension
            num_decision_steps: Number of TabNet attention steps
            num_cross_layers: Number of cross layers after TabNet
            hidden_units: List of hidden layer units
            dropout_rate: Dropout rate
        """
        self.num_features = num_features
        self.id_embedding_dim = id_embedding_dim
        self.tabnet_feature_dim = tabnet_feature_dim
        self.tabnet_output_dim = tabnet_output_dim
        self.num_decision_steps = num_decision_steps
        self.num_cross_layers = num_cross_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self, num_ids):
        """
        Build TabNet trend model.
        
        Args:
            num_ids: Number of unique IDs for embedding
            
        Returns:
            Keras Model
        """
        # Inputs
        id_input = layers.Input(shape=(1,), dtype='int32', name='trend_id_input')
        time_input = layers.Input(shape=(1,), dtype='float32', name='trend_time_input')
        
        # ID embedding
        id_embed = layers.Embedding(
            input_dim=num_ids,
            output_dim=self.id_embedding_dim,
            name='trend_id_embedding'
        )(id_input)
        id_embed = layers.Flatten()(id_embed)
        
        # Concatenate all features
        concat = layers.Concatenate()([id_embed, time_input])
        
        # TabNet encoder
        tabnet_output = TabNetEncoder(
            feature_dim=self.tabnet_feature_dim,
            output_dim=self.tabnet_output_dim,
            num_decision_steps=self.num_decision_steps,
            name='trend_tabnet'
        )(concat)
        
        # Hidden layers
        x = tabnet_output
        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units, activation='relu', name=f'trend_dense_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'trend_dropout_{i}')(x)
        
        # Output layer (pre-normalization)
        trend_output = layers.Dense(1, activation=None, name='trend_output')(x)
        
        # Unit normalization
        trend_normalized = UnitNormLayer(name='trend_unit_norm')(trend_output)
        
        # Create model
        self.model = Model(
            inputs=[id_input, time_input],
            outputs=trend_normalized,
            name='tabnet_trend_component'
        )
        
        return self.model


class TabNetSeasonalComponent:
    """
    TabNet-based Seasonal Component with Unit Normalization.
    
    Captures seasonal patterns using TabNet architecture with Fourier features.
    Applies unit normalization to output before combination.
    """
    
    def __init__(self,
                 num_fourier_features,
                 id_embedding_dim=16,
                 tabnet_feature_dim=64,
                 tabnet_output_dim=32,
                 num_decision_steps=3,
                 num_cross_layers=2,
                 hidden_units=[64, 32],
                 dropout_rate=0.1):
        """
        Initialize TabNet seasonal component.
        
        Args:
            num_fourier_features: Number of Fourier features
            id_embedding_dim: Dimension of ID embeddings
            tabnet_feature_dim: TabNet feature transformation dimension
            tabnet_output_dim: TabNet output dimension
            num_decision_steps: Number of TabNet attention steps
            num_cross_layers: Number of cross layers after TabNet
            hidden_units: List of hidden layer units
            dropout_rate: Dropout rate
        """
        self.num_fourier_features = num_fourier_features
        self.id_embedding_dim = id_embedding_dim
        self.tabnet_feature_dim = tabnet_feature_dim
        self.tabnet_output_dim = tabnet_output_dim
        self.num_decision_steps = num_decision_steps
        self.num_cross_layers = num_cross_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self, num_ids):
        """
        Build TabNet seasonal model.
        
        Args:
            num_ids: Number of unique IDs for embedding
            
        Returns:
            Keras Model
        """
        # Inputs
        id_input = layers.Input(shape=(1,), dtype='int32', name='seasonal_id_input')
        fourier_input = layers.Input(
            shape=(self.num_fourier_features,),
            dtype='float32',
            name='seasonal_fourier_input'
        )
        
        # ID embedding
        id_embed = layers.Embedding(
            input_dim=num_ids,
            output_dim=self.id_embedding_dim,
            name='seasonal_id_embedding'
        )(id_input)
        id_embed = layers.Flatten()(id_embed)
        
        # Concatenate all features
        concat = layers.Concatenate()([id_embed, fourier_input])
        
        # TabNet encoder
        tabnet_output = TabNetEncoder(
            feature_dim=self.tabnet_feature_dim,
            output_dim=self.tabnet_output_dim,
            num_decision_steps=self.num_decision_steps,
            name='seasonal_tabnet'
        )(concat)
        
        # Hidden layers
        x = tabnet_output
        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units, activation='relu', name=f'seasonal_dense_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'seasonal_dropout_{i}')(x)
        
        # Output layer (pre-normalization)
        seasonal_output = layers.Dense(1, activation=None, name='seasonal_output')(x)
        
        # Unit normalization
        seasonal_normalized = UnitNormLayer(name='seasonal_unit_norm')(seasonal_output)
        
        # Create model
        self.model = Model(
            inputs=[id_input, fourier_input],
            outputs=seasonal_normalized,
            name='tabnet_seasonal_component'
        )
        
        return self.model


class TabNetHolidayComponent:
    """
    TabNet-based Holiday Component with Unit Normalization.
    
    Captures holiday effects using TabNet architecture.
    Applies unit normalization to output before combination.
    """
    
    def __init__(self,
                 id_embedding_dim=16,
                 tabnet_feature_dim=64,
                 tabnet_output_dim=32,
                 num_decision_steps=3,
                 num_cross_layers=2,
                 hidden_units=[32, 16],
                 dropout_rate=0.1):
        """
        Initialize TabNet holiday component.
        
        Args:
            id_embedding_dim: Dimension of ID embeddings
            tabnet_feature_dim: TabNet feature transformation dimension
            tabnet_output_dim: TabNet output dimension
            num_decision_steps: Number of TabNet attention steps
            num_cross_layers: Number of cross layers after TabNet
            hidden_units: List of hidden layer units
            dropout_rate: Dropout rate
        """
        self.id_embedding_dim = id_embedding_dim
        self.tabnet_feature_dim = tabnet_feature_dim
        self.tabnet_output_dim = tabnet_output_dim
        self.num_decision_steps = num_decision_steps
        self.num_cross_layers = num_cross_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self, num_ids):
        """
        Build TabNet holiday model.
        
        Args:
            num_ids: Number of unique IDs for embedding
            
        Returns:
            Keras Model
        """
        # Inputs
        id_input = layers.Input(shape=(1,), dtype='int32', name='holiday_id_input')
        holiday_input = layers.Input(shape=(1,), dtype='float32', name='holiday_input')
        
        # ID embedding
        id_embed = layers.Embedding(
            input_dim=num_ids,
            output_dim=self.id_embedding_dim,
            name='holiday_id_embedding'
        )(id_input)
        id_embed = layers.Flatten()(id_embed)
        
        # Concatenate all features
        concat = layers.Concatenate()([id_embed, holiday_input])
        
        # TabNet encoder
        tabnet_output = TabNetEncoder(
            feature_dim=self.tabnet_feature_dim,
            output_dim=self.tabnet_output_dim,
            num_decision_steps=self.num_decision_steps,
            name='holiday_tabnet'
        )(concat)
        
        # Hidden layers
        x = tabnet_output
        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units, activation='relu', name=f'holiday_dense_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'holiday_dropout_{i}')(x)
        
        # Output layer (pre-normalization)
        holiday_output = layers.Dense(1, activation=None, name='holiday_output')(x)
        
        # Unit normalization
        holiday_normalized = UnitNormLayer(name='holiday_unit_norm')(holiday_output)
        
        # Create model
        self.model = Model(
            inputs=[id_input, holiday_input],
            outputs=holiday_normalized,
            name='tabnet_holiday_component'
        )
        
        return self.model


class TabNetRegressorComponent:
    """
    TabNet-based Regressor Component with Unit Normalization.
    
    Captures effects from exogenous regressors using TabNet architecture.
    Applies unit normalization to output before combination.
    """
    
    def __init__(self,
                 num_regressor_features,
                 id_embedding_dim=16,
                 tabnet_feature_dim=64,
                 tabnet_output_dim=32,
                 num_decision_steps=3,
                 num_cross_layers=2,
                 hidden_units=[64, 32],
                 dropout_rate=0.1):
        """
        Initialize TabNet regressor component.
        
        Args:
            num_regressor_features: Number of regressor features
            id_embedding_dim: Dimension of ID embeddings
            tabnet_feature_dim: TabNet feature transformation dimension
            tabnet_output_dim: TabNet output dimension
            num_decision_steps: Number of TabNet attention steps
            num_cross_layers: Number of cross layers after TabNet
            hidden_units: List of hidden layer units
            dropout_rate: Dropout rate
        """
        self.num_regressor_features = num_regressor_features
        self.id_embedding_dim = id_embedding_dim
        self.tabnet_feature_dim = tabnet_feature_dim
        self.tabnet_output_dim = tabnet_output_dim
        self.num_decision_steps = num_decision_steps
        self.num_cross_layers = num_cross_layers
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_model(self, num_ids):
        """
        Build TabNet regressor model.
        
        Args:
            num_ids: Number of unique IDs for embedding
            
        Returns:
            Keras Model
        """
        # Inputs
        id_input = layers.Input(shape=(1,), dtype='int32', name='regressor_id_input')
        regressor_input = layers.Input(
            shape=(self.num_regressor_features,),
            dtype='float32',
            name='regressor_input'
        )
        
        # ID embedding
        id_embed = layers.Embedding(
            input_dim=num_ids,
            output_dim=self.id_embedding_dim,
            name='regressor_id_embedding'
        )(id_input)
        id_embed = layers.Flatten()(id_embed)
        
        # Concatenate all features
        concat = layers.Concatenate()([id_embed, regressor_input])
        
        # TabNet encoder
        tabnet_output = TabNetEncoder(
            feature_dim=self.tabnet_feature_dim,
            output_dim=self.tabnet_output_dim,
            num_decision_steps=self.num_decision_steps,
            name='regressor_tabnet'
        )(concat)
        
        # Hidden layers
        x = tabnet_output
        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units, activation='relu', name=f'regressor_dense_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'regressor_dropout_{i}')(x)
        
        # Output layer (pre-normalization)
        regressor_output = layers.Dense(1, activation=None, name='regressor_output')(x)
        
        # Unit normalization
        regressor_normalized = UnitNormLayer(name='regressor_unit_norm')(regressor_output)
        
        # Create model
        self.model = Model(
            inputs=[id_input, regressor_input],
            outputs=regressor_normalized,
            name='tabnet_regressor_component'
        )
        
        return self.model


class TabNetDeepSequenceModel:
    """
    Complete DeepSequence Model with TabNet components and Unit Normalization.
    
    Combines all components using ADDITIVE mode (as in original DeepFuture):
        ŷ = norm(Trend) + norm(Seasonal) + norm(Holiday) + norm(Regressor)
    
    Where norm(x) applies L2 unit normalization to each component.
    """
    
    def __init__(self,
                 num_ids,
                 num_fourier_features,
                 num_regressor_features=0,
                 include_holiday=False,
                 tabnet_config=None):
        """
        Initialize TabNet DeepSequence model.
        
        Args:
            num_ids: Number of unique IDs
            num_fourier_features: Number of Fourier seasonal features
            num_regressor_features: Number of regressor features (0 if not used)
            include_holiday: Whether to include holiday component
            tabnet_config: Dictionary with TabNet configuration parameters
        """
        self.num_ids = num_ids
        self.num_fourier_features = num_fourier_features
        self.num_regressor_features = num_regressor_features
        self.include_holiday = include_holiday
        
        # Default TabNet configuration
        self.tabnet_config = tabnet_config or {
            'id_embedding_dim': 16,
            'tabnet_feature_dim': 64,
            'tabnet_output_dim': 32,
            'num_decision_steps': 3,
            'dropout_rate': 0.1
        }
        
        self.model = None
    
    def build(self):
        """
        Build complete model with all components.
        
        Returns:
            Keras Model
        """
        # Create shared inputs
        id_input = layers.Input(shape=(1,), dtype='int32', name='id_input')
        time_input = layers.Input(shape=(1,), dtype='float32', name='time_input')
        fourier_input = layers.Input(
            shape=(self.num_fourier_features,),
            dtype='float32',
            name='fourier_input'
        )
        
        # Build trend component
        # ID embedding
        id_embed_trend = layers.Embedding(
            input_dim=self.num_ids,
            output_dim=self.tabnet_config['id_embedding_dim'],
            name='trend_id_embedding'
        )(id_input)
        id_embed_trend = layers.Flatten()(id_embed_trend)
        
        trend_concat = layers.Concatenate()([id_embed_trend, time_input])
        trend_tabnet = TabNetEncoder(
            feature_dim=self.tabnet_config['tabnet_feature_dim'],
            output_dim=self.tabnet_config['tabnet_output_dim'],
            num_decision_steps=self.tabnet_config['num_decision_steps'],
            name='trend_tabnet'
        )(trend_concat)
        
        # Apply cross layers after TabNet
        x = trend_tabnet
        x0_trend = x  # Store initial input for cross network
        num_cross_layers = self.tabnet_config.get('num_cross_layers', 2)
        for i in range(num_cross_layers):
            x = CrossLayer(name=f'trend_cross_{i}')([x0_trend, x])
        
        # Dense layers after cross network
        for i, units in enumerate([64, 32]):
            x = layers.Dense(
                units, activation='relu', name=f'trend_dense_{i}'
            )(x)
            x = layers.Dropout(
                self.tabnet_config['dropout_rate'],
                name=f'trend_dropout_{i}'
            )(x)
        
        trend_output = layers.Dense(1, activation=None, name='trend_output')(x)
        trend_norm = UnitNormLayer(name='trend_unit_norm')(trend_output)
        
        # Build seasonal component
        # ID embedding
        id_embed_seasonal = layers.Embedding(
            input_dim=self.num_ids,
            output_dim=self.tabnet_config['id_embedding_dim'],
            name='seasonal_id_embedding'
        )(id_input)
        id_embed_seasonal = layers.Flatten()(id_embed_seasonal)
        
        seasonal_concat = layers.Concatenate()([
            id_embed_seasonal, fourier_input
        ])
        seasonal_tabnet = TabNetEncoder(
            feature_dim=self.tabnet_config['tabnet_feature_dim'],
            output_dim=self.tabnet_config['tabnet_output_dim'],
            num_decision_steps=self.tabnet_config['num_decision_steps'],
            name='seasonal_tabnet'
        )(seasonal_concat)
        
        # Apply cross layers after TabNet
        x = seasonal_tabnet
        x0_seasonal = x  # Store initial input for cross network
        num_cross_layers = self.tabnet_config.get('num_cross_layers', 2)
        for i in range(num_cross_layers):
            x = CrossLayer(name=f'seasonal_cross_{i}')([x0_seasonal, x])
        
        # Dense layers after cross network
        for i, units in enumerate([64, 32]):
            x = layers.Dense(
                units, activation='relu', name=f'seasonal_dense_{i}'
            )(x)
            x = layers.Dropout(
                self.tabnet_config['dropout_rate'],
                name=f'seasonal_dropout_{i}'
            )(x)
        
        seasonal_output = layers.Dense(1, activation=None, name='seasonal_output')(x)
        seasonal_norm = UnitNormLayer(name='seasonal_unit_norm')(seasonal_output)
        
        # Additive combination (as in original DeepFuture)
        combined = layers.Add(name='trend_seasonal_add')([
            trend_norm, seasonal_norm
        ])
        
        # Collect input list
        input_list = [id_input, time_input, fourier_input]
        
        # Add holiday component if requested
        if self.include_holiday:
            holiday_input = layers.Input(
                shape=(1,), dtype='float32', name='holiday_input'
            )
            input_list.append(holiday_input)
            
            id_embed_holiday = layers.Embedding(
                input_dim=self.num_ids,
                output_dim=self.tabnet_config['id_embedding_dim'],
                name='holiday_id_embedding'
            )(id_input)
            id_embed_holiday = layers.Flatten()(id_embed_holiday)
            
            holiday_concat = layers.Concatenate()([
                id_embed_holiday, holiday_input
            ])
            holiday_tabnet = TabNetEncoder(
                feature_dim=self.tabnet_config['tabnet_feature_dim'],
                output_dim=self.tabnet_config['tabnet_output_dim'],
                num_decision_steps=self.tabnet_config['num_decision_steps'],
                name='holiday_tabnet'
            )(holiday_concat)
            
            # Apply cross layers after TabNet
            x = holiday_tabnet
            x0_holiday = x  # Store initial input for cross network
            num_cross_layers = self.tabnet_config.get('num_cross_layers', 2)
            for i in range(num_cross_layers):
                x = CrossLayer(name=f'holiday_cross_{i}')([x0_holiday, x])
            
            # Dense layers after cross network
            for i, units in enumerate([32, 16]):
                x = layers.Dense(
                    units, activation='relu', name=f'holiday_dense_{i}'
                )(x)
                x = layers.Dropout(
                    self.tabnet_config['dropout_rate'],
                    name=f'holiday_dropout_{i}'
                )(x)
            
            holiday_output = layers.Dense(
                1, activation=None, name='holiday_output'
            )(x)
            holiday_norm = UnitNormLayer(name='holiday_unit_norm')(
                holiday_output
            )
            combined = layers.Add(name='with_holiday_add')([
                combined, holiday_norm
            ])
        
        # Add regressor component if features provided
        if self.num_regressor_features > 0:
            regressor_input = layers.Input(
                shape=(self.num_regressor_features,),
                dtype='float32',
                name='regressor_input'
            )
            input_list.append(regressor_input)
            
            id_embed_regressor = layers.Embedding(
                input_dim=self.num_ids,
                output_dim=self.tabnet_config['id_embedding_dim'],
                name='regressor_id_embedding'
            )(id_input)
            id_embed_regressor = layers.Flatten()(id_embed_regressor)
            
            regressor_concat = layers.Concatenate()([
                id_embed_regressor, regressor_input
            ])
            regressor_tabnet = TabNetEncoder(
                feature_dim=self.tabnet_config['tabnet_feature_dim'],
                output_dim=self.tabnet_config['tabnet_output_dim'],
                num_decision_steps=self.tabnet_config['num_decision_steps'],
                name='regressor_tabnet'
            )(regressor_concat)
            
            # Apply cross layers after TabNet
            x = regressor_tabnet
            x0_regressor = x  # Store initial input for cross network
            num_cross_layers = self.tabnet_config.get('num_cross_layers', 2)
            for i in range(num_cross_layers):
                x = CrossLayer(name=f'regressor_cross_{i}')([
                    x0_regressor, x
                ])
            
            # Dense layers after cross network
            for i, units in enumerate([64, 32]):
                x = layers.Dense(
                    units, activation='relu', name=f'regressor_dense_{i}'
                )(x)
                x = layers.Dropout(
                    self.tabnet_config['dropout_rate'],
                    name=f'regressor_dropout_{i}'
                )(x)
            
            regressor_output = layers.Dense(
                1, activation=None, name='regressor_output'
            )(x)
            regressor_norm = UnitNormLayer(name='regressor_unit_norm')(
                regressor_output
            )
            combined = layers.Add(name='with_regressor_add')([
                combined, regressor_norm
            ])
        
        # Final output (already normalized components, additive combination)
        final_output = layers.Dense(
            1,
            activation='relu',  # Ensure non-negative forecasts
            name='final_output'
        )(combined)
        
        # Create complete model
        self.model = Model(
            inputs=input_list,
            outputs=final_output,
            name='tabnet_deepsequence_unit_norm'
        )
        
        return self.model
    
    def compile(self, learning_rate=0.001):
        """
        Compile model with optimizer and loss.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build() first.")
