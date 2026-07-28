"""
Main model builder for Hierarchical Attention DeepSequence.

This module provides the high-level API for creating and training
the hierarchical attention model for intermittent demand forecasting.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Tuple, List, Optional, Dict
import numpy as np

from .components import (
    DeepSequencePWLHierarchical,
    TrendComponentBuilder,
    SeasonalComponentBuilder,
    HolidayComponentBuilder,
    RegressorComponentBuilder
)


def create_hierarchical_model(
    num_skus: int,
    n_features: int,
    id_embedding_dim: int = 8,
    component_hidden_units: int = 32,
    component_dropout: float = 0.2,
    zero_prob_hidden_units: int = 64,
    zero_prob_hidden_layers: int = 2,
    zero_prob_dropout: float = 0.2,
    activation: str = 'mish',
    data_frequency: str = 'daily',
    trend_feature_indices: Optional[List[int]] = None,
    seasonal_feature_indices: Optional[List[int]] = None,
    holiday_feature_index: Optional[int] = None,
    regressor_feature_indices: Optional[List[int]] = None,
    time_min: Optional[float] = None,
    time_max: Optional[float] = None,
    n_changepoints: int = 10,
    n_holiday_keypoints: int = 37,
    holiday_keypoint_range: float = 365.0
) -> Tuple[Model, Model, Model, Model, Model]:
    """
    Create a hierarchical attention model for intermittent demand forecasting.
    
    Args:
        num_skus: Number of unique SKUs in the dataset
        n_features: Total number of input features
        id_embedding_dim: Dimension of SKU embedding
        component_hidden_units: Hidden units for each component
        component_dropout: Dropout rate for components
        zero_prob_hidden_units: Hidden units for zero probability network
        zero_prob_hidden_layers: Number of hidden layers in zero probability network
        zero_prob_dropout: Dropout rate for zero probability network
        activation: Activation function ('mish', 'relu', 'swish', etc.)
        data_frequency: Data frequency ('daily', 'weekly', 'monthly')
        trend_feature_indices: Indices of time features
        seasonal_feature_indices: Indices of seasonal/Fourier features
        holiday_feature_index: Index of holiday distance feature
        regressor_feature_indices: Indices of additional regressor features
        time_min: Minimum time value for PWL calibration
        time_max: Maximum time value for PWL calibration
        n_changepoints: Number of changepoints for trend PWL
        n_holiday_keypoints: Number of keypoints for holiday PWL
        holiday_keypoint_range: Range for holiday PWL keypoints
    
    Returns:
        Tuple of (main_model, trend_model, seasonal_model, holiday_model, regressor_model)
    """
    # Create model instance
    model = DeepSequencePWLHierarchical(
        num_skus=num_skus,
        n_features=n_features,
        enable_intermittent_handling=True,
        id_embedding_dim=id_embedding_dim,
        component_hidden_units=component_hidden_units,
        component_dropout=component_dropout,
        zero_prob_hidden_units=zero_prob_hidden_units,
        zero_prob_hidden_layers=zero_prob_hidden_layers,
        zero_prob_dropout=zero_prob_dropout,
        activation=activation,
        data_frequency=data_frequency
    )
    
    # Configure PWL parameters if time range provided
    if time_min is not None and time_max is not None:
        model.trend_builder = TrendComponentBuilder(
            hidden_units=component_hidden_units,
            activation=activation,
            dropout=component_dropout,
            use_pwl=True,
            n_changepoints=n_changepoints,
            time_min=time_min,
            time_max=time_max
        )
    
    # Configure holiday PWL
    model.holiday_builder = HolidayComponentBuilder(
        hidden_units=component_hidden_units,
        activation=activation,
        dropout=component_dropout,
        use_pwl=True,
        n_keypoints=n_holiday_keypoints,
        keypoint_range=holiday_keypoint_range,
        data_frequency=data_frequency
    )
    
    # Build the model
    main_model, trend_model, seasonal_model, holiday_model, regressor_model = model.build_model(
        trend_feature_indices=trend_feature_indices,
        seasonal_feature_indices=seasonal_feature_indices,
        holiday_feature_index=holiday_feature_index,
        regressor_feature_indices=regressor_feature_indices
    )
    
    return main_model, trend_model, seasonal_model, holiday_model, regressor_model


def compile_hierarchical_model(
    model: Model,
    learning_rate: float = 0.005,
    loss_weights: Optional[Dict[str, float]] = None
) -> Model:
    """
    Compile the hierarchical attention model with appropriate loss and metrics.
    
    Args:
        model: The model to compile
        learning_rate: Initial learning rate
        loss_weights: Dictionary of loss weights for different outputs
    
    Returns:
        Compiled model
    """
    if loss_weights is None:
        loss_weights = {
            'base_forecast': 1.0,
            'final_forecast': 1.0,
            'zero_probability': 1.0
        }
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'base_forecast': 'mse',
            'final_forecast': 'mse',
            'zero_probability': 'binary_crossentropy'
        },
        loss_weights=loss_weights,
        metrics={
            'base_forecast': ['mae'],
            'final_forecast': ['mae'],
            'zero_probability': ['accuracy']
        }
    )
    
    return model


def get_training_callbacks(
    checkpoint_path: str,
    patience: int = 20,
    reduce_lr_patience: int = 5,
    min_lr: float = 1e-7
) -> List:
    """
    Get standard training callbacks for hierarchical attention model.
    
    Args:
        checkpoint_path: Path to save model checkpoints
        patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        min_lr: Minimum learning rate
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    
    return callbacks


__all__ = [
    'create_hierarchical_model',
    'compile_hierarchical_model',
    'get_training_callbacks'
]
