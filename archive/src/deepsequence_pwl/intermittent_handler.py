"""
Intermittent Handler for DeepSequence.

Handles sparse/intermittent demand with two-stage prediction:
1. Zero probability: P(demand = 0)
2. Candidate forecast: E[demand | demand > 0]
Final forecast = (1 - zero_prob) * candidate_forecast
"""

import tensorflow as tf


def _setup_precision():
    """Configure dtype and mixed precision based on available device."""
    try:
        mps_devices = tf.config.list_physical_devices('GPU')
        if mps_devices:
            from tf_keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            return tf.float32
    except Exception:
        pass
    return tf.float32


DTYPE = _setup_precision()


# Try tf_keras first (Keras 2), fallback to tensorflow.keras
try:
    from tf_keras.layers import (
        Dense, Concatenate, Dropout, Multiply, Add
    )
    from tf_keras import regularizers
except ImportError:
    from tf_keras.layers import (
        Dense, Concatenate, Dropout, Multiply, Add
    )
    from tf_keras import regularizers


class IntermittentHandler:
    """
    Two-stage intermittent demand handler.
    
    Architecture:
    1. Takes combined outputs from ALL components
       (trend, seasonal, holiday, regressor)
    2. Predicts TWO outputs:
       - Zero probability: P(demand = 0) via sigmoid
       - Candidate forecast: E[demand | demand > 0] via linear
    3. Final forecast = (1 - zero_prob) * candidate_forecast
    
    This handles sparse/intermittent demand patterns effectively.
    """
    
    def __init__(self,
                 hidden_units: int = 32,
                 hidden_layers: int = 2,
                 activation: str = 'relu',
                 dropout: float = 0.2,
                 l1_reg: float = 0.01,
                 use_cross_layer: bool = False):
        """
        Initialize intermittent handler.
        
        Args:
            hidden_units: Number of units in hidden layers
            hidden_layers: Number of hidden layers
            activation: Activation function for hidden layers
            dropout: Dropout rate
            l1_reg: L1 regularization factor
            use_cross_layer: [OPTIONAL] Apply cross-layer to component outputs
                           (default: False, usually minimal benefit)
        """
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.l1_reg = l1_reg
        self.use_cross_layer = use_cross_layer
        self.model = None
    
    def build_model(self, component_outputs):
        """
        Build two-stage intermittent handler model.
        
        Takes combined outputs from all components and predicts:
        1. Zero probability: P(demand = 0)
        2. Candidate forecast: E[demand | demand > 0]
        
        Args:
            component_outputs: Concatenated tensor from all components
                             (trend + seasonal + holiday + regressor)
            
        Returns:
            Tuple of (zero_probability, candidate_forecast, final_forecast)
        """
        # Optional cross-layer for component interactions
        x = component_outputs
        if self.use_cross_layer:
            for i in range(self.hidden_layers):
                residual = x
                x = Dense(
                    self.hidden_units,
                    activation=self.activation,
                    kernel_regularizer=regularizers.l1(self.l1_reg),
                    name=f'cross_layer_{i+1}'
                )(x)
                if self.dropout > 0:
                    x = Dropout(self.dropout)(x)
                # Residual connection
                x = Add()([x, residual])
        
        # Branch 1: Zero probability P(demand = 0)
        zero_branch = x
        for i in range(self.hidden_layers):
            zero_branch = Dense(
                self.hidden_units,
                activation=self.activation,
                kernel_regularizer=regularizers.l1(self.l1_reg),
                name=f'zero_prob_hidden_{i+1}'
            )(zero_branch)
            if self.dropout > 0:
                zero_branch = Dropout(
                    self.dropout,
                    name=f'zero_prob_dropout_{i+1}'
                )(zero_branch)
        
        zero_probability = Dense(
            1,
            activation='sigmoid',
            name='zero_probability'
        )(zero_branch)
        
        # Branch 2: Candidate forecast E[demand | demand > 0]
        candidate_branch = x
        for i in range(self.hidden_layers):
            candidate_branch = Dense(
                self.hidden_units,
                activation=self.activation,
                kernel_regularizer=regularizers.l1(self.l1_reg),
                name=f'candidate_hidden_{i+1}'
            )(candidate_branch)
            if self.dropout > 0:
                candidate_branch = Dropout(
                    self.dropout,
                    name=f'candidate_dropout_{i+1}'
                )(candidate_branch)
        
        candidate_forecast = Dense(
            1,
            activation='linear',
            name='candidate_forecast'
        )(candidate_branch)
        
        # Final forecast = (1 - zero_prob) * candidate_forecast
        # Subtract from 1 to get non-zero probability
        one_minus_zero_prob = tf.subtract(
            1.0, zero_probability, name='non_zero_probability'
        )
        
        final_forecast = Multiply(name='final_forecast')([
            one_minus_zero_prob, candidate_forecast
        ])
        
        return zero_probability, candidate_forecast, final_forecast
    
    def get_probability_layer(self):
        """
        Get the probability output layer for integration with main model.
        
        Returns:
            Output layer (probability)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model.output


def create_intermittent_forecast(trend_output, seasonal_output,
                                 holiday_output, regressor_output,
                                 hidden_units=32, hidden_layers=2,
                                 use_cross_layer=False):
    """
    Create complete intermittent forecasting model.
    
    Combines all component outputs and applies two-stage prediction:
    1. Zero probability: P(demand = 0)
    2. Candidate forecast: E[demand | demand > 0]
    Final = (1 - zero_prob) * candidate
    
    Args:
        trend_output: Output from trend component
        seasonal_output: Output from seasonal component
        holiday_output: Output from holiday component
        regressor_output: Output from regressor component
        hidden_units: Hidden layer size
        hidden_layers: Number of hidden layers
        use_cross_layer: Apply cross-layer for component interactions
        
    Returns:
        Tuple of (zero_prob, candidate_forecast, final_forecast)
    """
    # Concatenate all component outputs
    combined = Concatenate(name='all_components')([
        trend_output,
        seasonal_output,
        holiday_output,
        regressor_output
    ])
    
    # Build intermittent handler
    handler = IntermittentHandler(
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        use_cross_layer=use_cross_layer
    )
    
    zero_prob, candidate, final = handler.build_model(combined)
    
    return zero_prob, candidate, final
