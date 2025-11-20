"""
Main DeepSequence model combining seasonal, trend, and regression components.
Fully decomposed architecture inspired by Prophet.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import List, Optional
import numpy as np

from .seasonal_component import SeasonalComponent
from .trend_component import TrendComponent
from .regressor_component import RegressorComponent
from .holiday_component import HolidayComponent
from .intermittent_handler import IntermittentHandler, apply_intermittent_mask
from .tabnet_encoder import TabNetEncoder
from .unit_norm import UnitNorm
from .cross_layer import CrossNetwork
from .combination_layer import CombinationLayer, ScalableCombination
from .additive_multiplicative import SequentialCombiner
from .config import DEFAULT_LEARNING_RATE, TRAINING_PARAMS


class DeepSequenceModel:
    """
    DeepSequence: Complete forecasting model with full decomposition.
    
    Architecture: ŷ = f(seasonal, trend, regressor)
    
    Components:
    - Seasonal: Captures periodic patterns (weekly, monthly, yearly)
    - Trend: Models long-term growth/decline (like Prophet)
    - Regressor: Handles exogenous variables and special effects
    
    Combination Modes:
    - 'additive': seasonal + trend + regressor
    - 'multiplicative': seasonal × trend (+ regressor)
    - 'seasonal_mult_trend': seasonal × trend + regressor
    - 'hybrid_add': seasonal × regressor + trend
    - 'hybrid_mult': seasonal × trend + regressor
    - 'three_way_mult': seasonal × trend × regressor
    - 'weighted_additive': learnable weights for combination
    - 'weighted_multiplicative': learnable seasonal × weighted(others)
    """

    def __init__(self, mode: str = 'additive', use_intermittent: bool = False,
                 use_tabnet: bool = False, learnable_combination: bool = False,
                 use_trend: bool = True):
        self.mode = mode
        self.use_intermittent = use_intermittent
        self.use_tabnet = use_tabnet
        self.learnable_combination = learnable_combination
        self.use_trend = use_trend
        self.seasonal_model = None
        self.trend_model = None
        self.holiday_model = None
        self.regressor_model = None
        self.intermittent_handler = None
        self.seasonal_tabnet = None
        self.trend_tabnet = None
        self.regressor_tabnet = None
        self.full_model = None

    def build(self, seasonal_component: SeasonalComponent,
              trend_component: Optional[TrendComponent] = None,
              holiday_component: Optional[HolidayComponent] = None,
              regressor_component: Optional[RegressorComponent] = None,
              use_sequential_combination: bool = True,
              intermittent_config: dict = None,
              tabnet_config: dict = None):
        """
        Build the full DeepSequence model with shared ID input architecture.
        
        The Trend component creates the shared ID input which is then passed
        to Seasonal, Holiday, and Regressor components for architectural
        coupling.
        
        Combination Modes:
        - Sequential (use_sequential_combination=True):
          T → T+S → (T+S)+H → (T+S+H)+R
        - Parallel (use_sequential_combination=False):
          Original parallel combination of components
        
        Args:
            seasonal_component: Seasonal component (required)
            trend_component: Trend component (optional, creates shared ID)
            holiday_component: Holiday component (optional, requires shared ID)
            regressor_component: Regressor component (optional)
            use_sequential_combination: Whether to use sequential combination
                (default: True)
            intermittent_config: Configuration for intermittent demand
            tabnet_config: Configuration for TabNet encoders
        """
        # Build Trend first (creates shared ID input)
        if trend_component is not None and self.use_trend:
            self.trend_model = trend_component.t_model
            shared_id_input = trend_component.id_input
        else:
            shared_id_input = None
        
        # Build Seasonal with shared ID
        if shared_id_input is not None:
            # Rebuild seasonal model with shared ID input
            self.seasonal_model = seasonal_component.seasonal_model(
                id_input=shared_id_input
            )
        else:
            self.seasonal_model = seasonal_component.s_model
        
        # Build Holiday with shared ID (REQUIRED)
        if holiday_component is not None:
            if shared_id_input is None:
                raise ValueError(
                    "Holiday component requires Trend component to provide "
                    "shared ID input"
                )
            # Build holiday model with shared ID input
            self.holiday_model = holiday_component.holiday_model(
                id_input=shared_id_input
            )
        else:
            self.holiday_model = None
        
        # Build Regressor with shared ID
        if regressor_component is not None:
            if shared_id_input is not None:
                # Rebuild regressor model with shared ID input
                self.regressor_model = regressor_component.reg_model(
                    id_input=shared_id_input
                )
            else:
                self.regressor_model = regressor_component.combined_reg_model
        
        # Get outputs
        seasonal_output = self.seasonal_model.output
        
        trend_output = None
        if self.trend_model is not None:
            trend_output = self.trend_model.output
        
        regressor_output = None
        if self.regressor_model is not None:
            regressor_output = self.regressor_model.output

        # Apply TabNet encoders if enabled
        if self.use_tabnet:
            if tabnet_config is None:
                tabnet_config = {}
            
            # TabNet for seasonal component
            self.seasonal_tabnet = TabNetEncoder(
                output_dim=tabnet_config.get('output_dim', 32),
                feature_dim=tabnet_config.get('feature_dim', 32),
                n_steps=tabnet_config.get('n_steps', 3),
                n_shared=tabnet_config.get('n_shared', 2),
                n_independent=tabnet_config.get('n_independent', 2),
                name='seasonal_tabnet'
            )
            seasonal_output = self.seasonal_tabnet(seasonal_output)
            
            # Apply cross layer for feature interactions
            seasonal_output = CrossNetwork(
                num_layers=tabnet_config.get('cross_layers', 2),
                use_bias=True,
                name='seasonal_cross'
            )(seasonal_output)
            
            # Apply unit normalization
            seasonal_output = UnitNorm(name='seasonal_unit_norm')(seasonal_output)
            
            # Final projection to forecast dimension
            seasonal_output = layers.Dense(1, activation='linear',
                                          name='seasonal_tabnet_output')(seasonal_output)
            
            # TabNet for regressor component
            self.regressor_tabnet = TabNetEncoder(
                output_dim=tabnet_config.get('output_dim', 32),
                feature_dim=tabnet_config.get('feature_dim', 32),
                n_steps=tabnet_config.get('n_steps', 3),
                n_shared=tabnet_config.get('n_shared', 2),
                n_independent=tabnet_config.get('n_independent', 2),
                name='regressor_tabnet'
            )
            regressor_output = self.regressor_tabnet(regressor_output)
            
            # Apply cross layer for feature interactions
            regressor_output = CrossNetwork(
                num_layers=tabnet_config.get('cross_layers', 2),
                use_bias=True,
                name='regressor_cross'
            )(regressor_output)
            
            # Apply unit normalization
            regressor_output = UnitNorm(name='regressor_unit_norm')(
                regressor_output)
            
            # Final projection to forecast dimension
            regressor_output = layers.Dense(
                1, activation='linear',
                name='regressor_tabnet_output')(regressor_output)

        # Get holiday output
        holiday_output = None
        if self.holiday_model is not None:
            holiday_output = self.holiday_model.output
        
        # Combine components
        if use_sequential_combination:
            # Sequential combination: T → T+S → (T+S)+H → (T+S+H)+R
            if trend_output is None:
                raise ValueError(
                    "Sequential combination requires Trend component"
                )
            
            # Create sequential combiner
            combiner = SequentialCombiner(mode=self.mode)
            
            # Apply sequential combination
            combined_output = combiner.combine_all(
                trend_output=trend_output,
                seasonal_model=self.seasonal_model,
                holiday_model=self.holiday_model,
                regressor_model=self.regressor_model
            )
        else:
            # Original parallel combination logic
            components = [seasonal_output]
            
            if trend_output is not None:
                components.append(trend_output)
            
            if holiday_output is not None:
                components.append(holiday_output)
            
            if regressor_output is not None:
                components.append(regressor_output)
            
            # Combine based on mode
            if len(components) == 1:
                # Only seasonal
                combined_output = seasonal_output
            elif self.learnable_combination:
                combined_output = ScalableCombination(
                    mode=self.mode if 'weighted' in self.mode
                    else 'weighted_additive',
                    use_bias=True,
                    name='learnable_forecast'
                )(components)
            elif self.mode == 'additive':
                combined_output = layers.Add(
                    name='additive_forecast'
                )(components)
            elif self.mode == 'multiplicative':
                if len(components) == 2:
                    combined_output = layers.Multiply(
                        name='multiplicative_forecast'
                    )(components)
                else:
                    combined_output = CombinationLayer(
                        mode='three_way_mult',
                        name='multiplicative_forecast'
                    )(components)
            else:
                # Default to additive
                combined_output = layers.Add(
                    name='additive_forecast'
                )(components)

        # Build final model with shared ID input handling
        all_inputs = []
        seen_input_names = set()
        
        # Collect inputs, avoiding duplicates (shared ID input)
        models_to_collect = [
            self.trend_model,
            self.seasonal_model,
            self.holiday_model,
            self.regressor_model
        ]
        for model in models_to_collect:
            if model is not None:
                model_inputs = (model.input if isinstance(model.input, list)
                                else [model.input])
                for inp in model_inputs:
                    if inp.name not in seen_input_names:
                        all_inputs.append(inp)
                        seen_input_names.add(inp.name)
        
        # Apply intermittent handler if enabled
        if self.use_intermittent:
            if intermittent_config is None:
                intermittent_config = {}
            
            # Create intermittent handler
            self.intermittent_handler = IntermittentHandler(**intermittent_config)
            
            # Build handler model using TabNet outputs if available
            if self.use_tabnet:
                # Use TabNet encoded features for intermittent handler
                seasonal_for_intermittent = self.seasonal_tabnet(
                    self.seasonal_model.output
                )
                if self.regressor_model is not None:
                    regressor_for_intermittent = self.regressor_tabnet(
                        self.regressor_model.output
                    )
                else:
                    regressor_for_intermittent = None
            else:
                # Use original component outputs
                seasonal_for_intermittent = self.seasonal_model.output
                if self.regressor_model is not None:
                    regressor_for_intermittent = self.regressor_model.output
                else:
                    regressor_for_intermittent = None
            
            # Concatenate encoded features for intermittent handler
            concat_inputs = [seasonal_for_intermittent]
            if regressor_for_intermittent is not None:
                concat_inputs.append(regressor_for_intermittent)
                
            if len(concat_inputs) > 1:
                intermittent_input = layers.Concatenate(name='intermittent_concat')(concat_inputs)
            else:
                intermittent_input = concat_inputs[0]
            
            # Apply cross layer for seasonal-regressor interactions
            intermittent_input = CrossNetwork(
                num_layers=intermittent_config.get('cross_layers', 2),
                use_bias=True,
                name='intermittent_cross'
            )(intermittent_input)
            
            # Build probability prediction network with unit norm
            prob_hidden = intermittent_input
            for i in range(intermittent_config.get('hidden_layers', 2)):
                prob_hidden = layers.Dense(
                    intermittent_config.get('hidden_units', 32),
                    activation=intermittent_config.get('activation', 'relu'),
                    kernel_regularizer=tf.keras.regularizers.l1(
                        intermittent_config.get('l1_reg', 0.01)
                    ),
                    name=f'intermittent_hidden_{i}'
                )(prob_hidden)
                # Apply unit normalization after activation
                prob_hidden = UnitNorm(
                    name=f'intermittent_unit_norm_{i}'
                )(prob_hidden)
                prob_hidden = layers.Dropout(
                    intermittent_config.get('dropout', 0.2),
                    name=f'intermittent_dropout_{i}'
                )(prob_hidden)
            
            probability = layers.Dense(
                1, activation='sigmoid',
                name='intermittent_probability'
            )(prob_hidden)
            
            # Apply mask: multiply forecast with probability
            combined_output = layers.Multiply(
                name='intermittent_masked_forecast'
            )([combined_output, probability])

        self.full_model = Model(
            inputs=all_inputs,
            outputs=combined_output,
            name='deepsequence_net'
        )

        return self.full_model

    def compile(self, loss='mape', learning_rate: float = DEFAULT_LEARNING_RATE):
        if loss == 'mape':
            loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
        elif loss == 'mae':
            loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif loss == 'mse':
            loss_fn = tf.keras.losses.MeanSquaredError()
        else:
            loss_fn = loss

        self.full_model.compile(
            loss=loss_fn,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        )

    def fit(self, train_input: List, train_target: np.ndarray, val_input: List = None, val_target: np.ndarray = None,
            epochs: int = 500, batch_size: int = 512, checkpoint_path: str = None, patience: int = 10, verbose: int = 1):
        callbacks = []
        early_stop = EarlyStopping(
            monitor='val_loss' if val_input else 'loss',
            mode=TRAINING_PARAMS['mode'],
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stop)

        if checkpoint_path:
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if val_input else 'loss',
                save_best_only=TRAINING_PARAMS['save_best_only'],
                mode=TRAINING_PARAMS['mode']
            )
            callbacks.append(checkpoint)

        validation_data = None
        if val_input is not None and val_target is not None:
            validation_data = (val_input, val_target)

        history = self.full_model.fit(
            train_input,
            train_target,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, inputs: List) -> np.ndarray:
        return self.full_model.predict(inputs)

    def save(self, path: str):
        self.full_model.save(path)

    @staticmethod
    def load(path: str, custom_objects: dict = None):
        model = DeepSequenceModel()
        model.full_model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        return model

    def summary(self):
        if self.full_model:
            self.full_model.summary()
        else:
            print("Model not built yet. Call build() first.")
