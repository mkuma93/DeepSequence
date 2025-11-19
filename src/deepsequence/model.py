"""
Main DeepSequence model combining seasonal and regression components.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import List
import numpy as np

from .seasonal_component import SeasonalComponent
from .regressor_component import RegressorComponent
from .intermittent_handler import IntermittentHandler, apply_intermittent_mask
from .tabnet_encoder import TabNetEncoder
from .unit_norm import UnitNorm
from .cross_layer import CrossNetwork
from .config import DEFAULT_LEARNING_RATE, TRAINING_PARAMS


class DeepSequenceModel:
    """
    DeepSequence: Complete forecasting model combining seasonal and
    regression components with optional TabNet encoders.
    """

    def __init__(self, mode: str = 'additive', use_intermittent: bool = False,
                 use_tabnet: bool = False):
        self.mode = mode
        self.use_intermittent = use_intermittent
        self.use_tabnet = use_tabnet
        self.seasonal_model = None
        self.regressor_model = None
        self.intermittent_handler = None
        self.seasonal_tabnet = None
        self.regressor_tabnet = None
        self.full_model = None

    def build(self, seasonal_component: SeasonalComponent,
              regressor_component: RegressorComponent,
              intermittent_config: dict = None,
              tabnet_config: dict = None):
        self.seasonal_model = seasonal_component.s_model
        self.regressor_model = regressor_component.combined_reg_model

        seasonal_output = self.seasonal_model.output
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

        # Combine seasonal and regressor outputs
        if self.mode == 'additive':
            combined_output = layers.Add(name='additive_forecast')([
                seasonal_output, regressor_output
            ])
        else:
            combined_output = layers.Multiply(name='multiplicative_forecast')([
                seasonal_output, regressor_output
            ])

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
                regressor_for_intermittent = self.regressor_tabnet(
                    self.regressor_model.output
                )
            else:
                # Use original component outputs
                seasonal_for_intermittent = self.seasonal_model.output
                regressor_for_intermittent = self.regressor_model.output
            
            # Concatenate encoded features for intermittent handler
            intermittent_input = layers.Concatenate(name='intermittent_concat')([
                seasonal_for_intermittent, regressor_for_intermittent
            ])
            
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
                    kernel_regularizer=layers.regularizers.l1(
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
            inputs=[self.seasonal_model.input, self.regressor_model.input],
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
