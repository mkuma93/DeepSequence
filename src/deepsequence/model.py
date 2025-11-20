"""
Main DeepSequence model with 4 components + Intermittent Handler.

Latest Architecture (November 2025):
- 4 Components: Seasonal (with Fourier), Trend, Regressor, Holiday
- Shared ID Embedding (16-dim) reused across all components
- TabNet + CrossNetwork + UnitNorm on Seasonal, Regressor, Holiday
- Simplified Trend: Dense + ReLU for changepoint modeling
- Cross-Component Interactions: CrossNetwork on all component outputs
- Intermittent Handler: Predicts zero vs non-zero probability
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import List, Optional, Dict
import numpy as np

from .seasonal_component import SeasonalComponent
from .regressor_component import RegressorComponent
from .tabnet_encoder import TabNetEncoder
from .unit_norm import UnitNorm
from .cross_layer import CrossNetwork
from .config import DEFAULT_LEARNING_RATE, TRAINING_PARAMS


class DeepSequenceModel:
    """
    DeepSequence: Full architecture with 4 components + Intermittent Handler.
    
    Architecture:
        1. Shared ID Embedding (16-dim)
        2. Seasonal: embeddings + 8 Fourier → TabNet → CrossNetwork → UnitNorm → Dense(1, L2 reg)
        3. Trend: id_embed + time → Dense(32,relu) → Dense(16,relu) → Dense(1)
        4. Regressor: id_embed + lags → TabNet → CrossNetwork → UnitNorm → Dense(1)
        5. Holiday: id_embed + indicator → TabNet → CrossNetwork → UnitNorm → Dense(1)
        6. Combined: seasonal + trend + regressor + holiday (additive)
        7. Cross-Component: component_outputs → CrossNetwork(2) → component_cross
        8. Intermittent Handler: component_cross → Dense → Sigmoid → probability
        9. Final: combined_forecast × probability
    """

    def __init__(self, mode: str = 'additive', use_intermittent: bool = True,
                 use_fourier: bool = True, use_cross_component: bool = True):
        self.mode = mode
        self.use_intermittent = use_intermittent
        self.use_fourier = use_fourier
        self.use_cross_component = use_cross_component
        self.full_model = None

    def build_full_architecture(
        self,
        n_ids: int,
        seasonal_cols: list,
        seasonal_n_unique: dict,
        fourier_names: list = None,
        lag_features: list = [1, 2, 4, 8],
        tabnet_config: dict = None,
        intermittent_config: dict = None
    ):
        """
        Build the full 4-component DeepSequence architecture.
        
        Latest Architecture (Nov 2025):
        - Shared ID Embedding (16-dim)
        - Seasonal: embeddings + Fourier → TabNet → Cross → UnitNorm
        - Trend: Dense + ReLU (simplified, no TabNet)
        - Regressor: lags → TabNet → Cross → UnitNorm
        - Holiday: indicator → TabNet → Cross → UnitNorm
        - Cross-Component: CrossNetwork on all outputs
        - Intermittent Handler: probability prediction
        
        Args:
            n_ids: Number of unique IDs
            seasonal_cols: List of seasonal feature column names
            seasonal_n_unique: Dict mapping seasonal cols to n_unique values
            fourier_names: List of Fourier feature names (default: 8)
            lag_features: List of lag periods (default: [1,2,4,8])
            tabnet_config: TabNet configuration
            intermittent_config: Intermittent handler configuration
        """
        if tabnet_config is None:
            tabnet_config = {
                'output_dim': 32, 'feature_dim': 32,
                'n_steps': 3, 'n_shared': 2, 'n_independent': 2
            }
        
        if intermittent_config is None:
            intermittent_config = {
                'hidden_units': [32, 16], 'dropout': 0.2
            }
        
        if fourier_names is None:
            fourier_names = [
                'fourier_weekly_sin1', 'fourier_weekly_cos1',
                'fourier_monthly_sin1', 'fourier_monthly_cos1',
                'fourier_quarterly_sin1', 'fourier_quarterly_cos1',
                'fourier_yearly_sin1', 'fourier_yearly_cos1'
            ]
        
        # 1. Shared ID Embedding
        id_input = layers.Input(shape=(1,), name='shared_id_input')
        id_embed = layers.Embedding(
            n_ids + 1, 16, name='shared_id_embed'
        )(id_input)
        id_embed = layers.Flatten()(id_embed)
        
        # 2. Seasonal Component with Fourier features
        seasonal_inputs = []
        seasonal_embeds = []
        for col in seasonal_cols:
            n_unique = seasonal_n_unique[col]
            feat_in = layers.Input(shape=(1,), name=f'seasonal_{col}')
            feat_embed = layers.Embedding(
                n_unique + 1, min(16, n_unique)
            )(feat_in)
            feat_embed = layers.Flatten()(feat_embed)
            seasonal_inputs.append(feat_in)
            seasonal_embeds.append(feat_embed)
        
        # Add Fourier inputs
        fourier_inputs = []
        if self.use_fourier and fourier_names:
            for fourier_name in fourier_names:
                fourier_in = layers.Input(shape=(1,), name=fourier_name)
                fourier_inputs.append(fourier_in)
        
        # Combine seasonal features
        seasonal_combined = layers.Concatenate()(
            [id_embed] + seasonal_embeds + fourier_inputs
        )
        
        # TabNet → CrossNetwork → UnitNorm → Dense
        l2_reg = tf.keras.regularizers.l2(0.01)
        seasonal_tabnet = TabNetEncoder(
            32, 32, 3, 2, 2, name='seasonal_tabnet'
        )(seasonal_combined)
        seasonal_cross = CrossNetwork(2, name='seasonal_cross')(
            seasonal_tabnet
        )
        seasonal_norm = UnitNorm(name='seasonal_unitnorm')(seasonal_cross)
        seasonal_output = layers.Dense(
            1, activation='linear', kernel_regularizer=l2_reg,
            name='seasonal_forecast'
        )(seasonal_norm)
        
        # 3. Trend Component (simplified, no TabNet)
        trend_time_in = layers.Input(shape=(1,), name='trend_time')
        trend_combined = layers.Concatenate()([id_embed, trend_time_in])
        trend_hidden = layers.Dense(32, activation='relu')(trend_combined)
        trend_hidden = layers.Dropout(0.2)(trend_hidden)
        trend_hidden = layers.Dense(16, activation='relu')(trend_hidden)
        trend_hidden = layers.Dropout(0.2)(trend_hidden)
        trend_output = layers.Dense(
            1, activation='linear', name='trend_forecast'
        )(trend_hidden)
        
        # 4. Regressor Component
        regressor_inputs = []
        for lag in lag_features:
            lag_in = layers.Input(shape=(1,), name=f'regressor_lag_{lag}')
            regressor_inputs.append(lag_in)
        
        regressor_combined = layers.Concatenate()(
            [id_embed] + regressor_inputs
        )
        regressor_tabnet = TabNetEncoder(
            32, 32, 3, 2, 2, name='regressor_tabnet'
        )(regressor_combined)
        regressor_cross = CrossNetwork(2, name='regressor_cross')(
            regressor_tabnet
        )
        regressor_norm = UnitNorm(name='regressor_unitnorm')(
            regressor_cross
        )
        regressor_output = layers.Dense(
            1, activation='linear', name='regressor_forecast'
        )(regressor_norm)
        
        # 5. Holiday Component
        holiday_in = layers.Input(shape=(1,), name='holiday_indicator')
        holiday_combined = layers.Concatenate()([id_embed, holiday_in])
        holiday_tabnet = TabNetEncoder(
            16, 16, 2, 1, 1, name='holiday_tabnet'
        )(holiday_combined)
        holiday_cross = CrossNetwork(1, name='holiday_cross')(
            holiday_tabnet
        )
        holiday_norm = UnitNorm(name='holiday_unitnorm')(holiday_cross)
        holiday_output = layers.Dense(
            1, activation='linear', name='holiday_forecast'
        )(holiday_norm)
        
        # 6. Combine Components (Additive)
        combined_forecast = layers.Add(name='combined_forecast')([
            seasonal_output, trend_output, regressor_output, holiday_output
        ])
        
        # 7. Cross-Component Interactions
        component_concat = layers.Concatenate(name='component_concat')([
            seasonal_output, trend_output, regressor_output, holiday_output
        ])
        
        if self.use_cross_component:
            component_cross = CrossNetwork(
                2, name='component_cross'
            )(component_concat)
        else:
            component_cross = component_concat
        
        # 8. Intermittent Handler
        if self.use_intermittent:
            intermittent_features = component_cross
            
            # Hidden layers with UnitNorm
            hidden_units = intermittent_config.get('hidden_units', [32, 16])
            dropout = intermittent_config.get('dropout', 0.2)
            
            intermittent_hidden = intermittent_features
            for i, units in enumerate(hidden_units):
                intermittent_hidden = layers.Dense(
                    units, activation='relu',
                    name=f'intermittent_hidden{i+1}'
                )(intermittent_hidden)
                intermittent_hidden = UnitNorm(
                    name=f'intermittent_unitnorm{i+1}'
                )(intermittent_hidden)
                intermittent_hidden = layers.Dropout(
                    dropout, name=f'intermittent_dropout{i+1}'
                )(intermittent_hidden)
            
            probability = layers.Dense(
                1, activation='sigmoid', name='intermittent_probability'
            )(intermittent_hidden)
            
            # 9. Final Forecast: combined × probability
            final_forecast = layers.Multiply(name='final_forecast')([
                combined_forecast, probability
            ])
        else:
            final_forecast = combined_forecast
        
        # Build model
        all_inputs = (
            [id_input] + seasonal_inputs + fourier_inputs +
            [trend_time_in] + regressor_inputs + [holiday_in]
        )
        
        self.full_model = Model(
            inputs=all_inputs,
            outputs=final_forecast,
            name='FullDeepSequence'
        )
        
        return self.full_model
    
    def compile(self, loss='mae', learning_rate=DEFAULT_LEARNING_RATE):
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

    def fit(
        self, train_input: List, train_target: np.ndarray,
        val_input: List = None, val_target: np.ndarray = None,
        epochs: int = 50, batch_size: int = 64,
        checkpoint_path: str = None, patience: int = 10, verbose: int = 1
    ):
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
        model.full_model = tf.keras.models.load_model(
            path, custom_objects=custom_objects
        )
        return model
    
    @staticmethod
    def create_fourier_features(data, index_col=None):
        """
        Create 8 Fourier features for seasonal patterns.
        
        Args:
            data: DataFrame
            index_col: Column to use as index for time-based features
        
        Returns:
            DataFrame with added Fourier features
        """
        if index_col:
            t = np.arange(len(data[index_col]))
        else:
            t = data.index.values
        
        # Weekly (7), Monthly (30), Quarterly (90), Yearly (365)
        data['fourier_weekly_sin1'] = np.sin(2 * np.pi * t / 7)
        data['fourier_weekly_cos1'] = np.cos(2 * np.pi * t / 7)
        data['fourier_monthly_sin1'] = np.sin(2 * np.pi * t / 30)
        data['fourier_monthly_cos1'] = np.cos(2 * np.pi * t / 30)
        data['fourier_quarterly_sin1'] = np.sin(2 * np.pi * t / 90)
        data['fourier_quarterly_cos1'] = np.cos(2 * np.pi * t / 90)
        data['fourier_yearly_sin1'] = np.sin(2 * np.pi * t / 365)
        data['fourier_yearly_cos1'] = np.cos(2 * np.pi * t / 365)
        
        return data

    def summary(self):
        if self.full_model:
            self.full_model.summary()
        else:
            print("Model not built yet. Call build() first.")
