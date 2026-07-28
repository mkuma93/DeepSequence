"""
Autoregressive prediction utilities for hierarchical attention models.

This module provides functionality to:
1. Create lag features from historical demand
2. Make multi-step predictions by feeding forecasts back as lag inputs
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Tuple, Optional


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = 'demand',
    sku_col: str = 'sku_id',
    date_col: str = 'date',
    lags: List[int] = [1, 2, 7],
    fill_method: str = 'mean'
) -> pd.DataFrame:
    """
    Create lag features for each SKU.
    
    Args:
        df: DataFrame with demand data
        target_col: Name of target column
        sku_col: Name of SKU identifier column
        date_col: Name of date column
        lags: List of lag periods (e.g., [1, 2, 7] for 1-day, 2-day, weekly)
        fill_method: How to fill missing lags ('mean', 'median', 'zero', 'forward')
    
    Returns:
        DataFrame with added lag features (lag_1, lag_7, etc.)
    """
    df = df.copy()
    df = df.sort_values([sku_col, date_col]).reset_index(drop=True)
    
    lag_features = {}
    
    for lag in lags:
        lag_col = f'lag_{lag}'
        # Create lag within each SKU group
        df[lag_col] = df.groupby(sku_col)[target_col].shift(lag)
        lag_features[lag_col] = lag
    
    # Handle missing values
    if fill_method == 'mean':
        for lag_col in lag_features.keys():
            # Fill with per-SKU mean
            sku_means = df.groupby(sku_col)[target_col].transform('mean')
            df[lag_col] = df[lag_col].fillna(sku_means)
    elif fill_method == 'median':
        for lag_col in lag_features.keys():
            sku_medians = df.groupby(sku_col)[target_col].transform('median')
            df[lag_col] = df[lag_col].fillna(sku_medians)
    elif fill_method == 'zero':
        for lag_col in lag_features.keys():
            df[lag_col] = df[lag_col].fillna(0.0)
    elif fill_method == 'forward':
        for lag_col in lag_features.keys():
            df[lag_col] = df.groupby(sku_col)[lag_col].fillna(method='ffill')
            # If still NaN (beginning of series), use mean
            sku_means = df.groupby(sku_col)[target_col].transform('mean')
            df[lag_col] = df[lag_col].fillna(sku_means)
    
    return df


class AutoregressivePredictor:
    """
    Wrapper for making autoregressive predictions with hierarchical attention models.
    
    For multi-step forecasting, predictions are fed back as lag features.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        lag_feature_indices: List[int],
        lags: List[int],
        n_skus: int
    ):
        """
        Args:
            model: Trained hierarchical attention model
            lag_feature_indices: Indices in feature matrix where lag features are located
            lags: List of lag periods used (e.g., [1, 7, 14])
            n_skus: Total number of SKUs
        """
        self.model = model
        self.lag_feature_indices = lag_feature_indices
        self.lags = lags
        self.n_skus = n_skus
        
        # Get n_features from model input shape
        # Model has 2 inputs: [features, sku_ids]
        n_features = model.input[0].shape[1]
        
        # Validate
        assert len(lag_feature_indices) == len(lags), \
            f"Mismatch: {len(lag_feature_indices)} indices vs {len(lags)} lags"
        
        # Create tf.function wrapper for efficient prediction
        # Use input_signature to avoid retracing with multiple outputs
        @tf.function(
            reduce_retracing=True,
            input_signature=[
                tf.TensorSpec(shape=[None, n_features],
                              dtype=tf.float32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.int32)
            ]
        )
        def _predict_fn(X_features, X_sku):
            """Optimized prediction with reduced retracing."""
            return self.model([X_features, X_sku], training=False)
        
        self._predict_fn = _predict_fn
    
    def predict_single_step(
        self,
        X_features: np.ndarray,
        X_sku: np.ndarray
    ) -> np.ndarray:
        """
        Make single-step predictions (standard inference).
        
        Args:
            X_features: Feature matrix [batch_size, n_features]
            X_sku: SKU indices [batch_size, 1]
        
        Returns:
            Predictions [batch_size, 1]
        """
        # Convert to tensors for tf.function
        X_features_tf = tf.convert_to_tensor(X_features, dtype=tf.float32)
        X_sku_tf = tf.convert_to_tensor(X_sku, dtype=tf.int32)
        
        # Use optimized tf.function prediction
        predictions = self._predict_fn(X_features_tf, X_sku_tf)
        
        # Extract final forecast from model outputs
        if isinstance(predictions, dict):
            result = predictions['final_forecast'].numpy()
        elif isinstance(predictions, (list, tuple)):
            # Assume final forecast is first output
            result = predictions[0].numpy()
        else:
            result = predictions.numpy()
        
        return result
    
    def predict_multi_step(
        self,
        X_features_init: np.ndarray,
        X_sku: np.ndarray,
        n_steps: int,
        historical_demand: Optional[Dict[int, List[float]]] = None
    ) -> np.ndarray:
        """
        Make multi-step autoregressive predictions.
        
        For each step:
        1. Make prediction with current features
        2. Update lag features using the prediction
        3. Move to next step
        
        Args:
            X_features_init: Initial feature matrix [batch_size, n_features]
            X_sku: SKU indices [batch_size, 1] (constant across steps)
            n_steps: Number of steps to forecast
            historical_demand: Optional dict {sku_idx: [recent_demands]} for initialization
        
        Returns:
            Predictions [batch_size, n_steps]
        """
        batch_size = X_features_init.shape[0]
        n_features = X_features_init.shape[1]
        
        # Store predictions for each step
        all_predictions = np.zeros((batch_size, n_steps))
        
        # Initialize lag buffer for each sample
        # lag_buffers[i] = deque of recent predictions for sample i
        from collections import deque
        lag_buffers = {}
        
        for i in range(batch_size):
            sku_idx = X_sku[i, 0]
            
            if historical_demand is not None and sku_idx in historical_demand:
                # Initialize with historical data
                buffer = deque(historical_demand[sku_idx], maxlen=max(self.lags))
            else:
                # Initialize with zeros or current lag features
                buffer = deque(maxlen=max(self.lags))
                for lag_idx in self.lag_feature_indices:
                    buffer.append(X_features_init[i, lag_idx])
            
            lag_buffers[i] = buffer
        
        # Current features (will be updated each step)
        X_features_current = X_features_init.copy()
        
        # Make predictions step by step
        for step in range(n_steps):
            # Predict current step
            step_predictions = self.predict_single_step(
                X_features_current,
                X_sku
            ).flatten()
            
            # Store predictions
            all_predictions[:, step] = step_predictions
            
            # Update lag features for next step
            if step < n_steps - 1:  # Don't update after last step
                for i in range(batch_size):
                    # Add new prediction to buffer
                    lag_buffers[i].append(step_predictions[i])
                    
                    # Update lag features in feature matrix
                    for j, lag in enumerate(self.lags):
                        lag_idx = self.lag_feature_indices[j]
                        
                        # Get lag value from buffer
                        buffer_list = list(lag_buffers[i])
                        if len(buffer_list) >= lag:
                            # Get the lag-th most recent value
                            X_features_current[i, lag_idx] = buffer_list[-lag]
                        else:
                            # Not enough history, use most recent available
                            X_features_current[i, lag_idx] = buffer_list[-1] if buffer_list else 0.0
        
        return all_predictions
    
    def predict_multi_step_iterative(
        self,
        unique_skus: np.ndarray,
        sku_to_idx: Dict,
        start_date: pd.Timestamp,
        n_steps: int,
        historical_demand: Optional[Dict[int, List[float]]] = None,
        holiday_data: Optional[pd.DataFrame] = None,
        feature_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Make multi-step autoregressive predictions iteratively (row-by-row).
        
        For each SKU:
        1. Start from start_date, predict t+1
        2. Use t+1 prediction as lag, advance to t+2, predict t+2
        3. Continue for n_steps
        
        This generates a dataset with one row per (sku, date, forecast).
        
        Args:
            unique_skus: Array of unique SKU IDs to forecast
            sku_to_idx: Dict mapping SKU ID to integer index
            start_date: Starting date (will predict start_date+1, +2, ...)
            n_steps: Number of days to forecast ahead
            historical_demand: Dict {sku_idx: [recent_demands]} for lag init
            holiday_data: DataFrame with holiday features (date, holiday_cols)
            feature_config: Dict with feature indices (see below)
        
        Returns:
            DataFrame with columns: [sku_id, date, forecast]
        """
        from collections import deque
        
        # Default feature configuration
        if feature_config is None:
            feature_config = {
                'holiday_indices': list(range(15)),
                'fourier_indices': list(range(15, 25)),
                'lag_indices': [25, 26, 27],
                'date_indices': [28, 29, 30, 31],
                'time_index': 32,
                'reference_date': pd.Timestamp('2009-01-01'),
                'n_fourier': 5
            }
        
        # Prepare holiday lookup if available
        holiday_lookup = {}
        if holiday_data is not None:
            for _, row in holiday_data.iterrows():
                holiday_lookup[row['date']] = row[1:].values
        
        # Storage for results
        results = []
        
        # Iterate through each SKU
        for sku_id in unique_skus:
            sku_idx = sku_to_idx[sku_id]
            
            # Initialize lag buffer for this SKU
            if historical_demand and sku_idx in historical_demand:
                lag_buffer = deque(historical_demand[sku_idx], 
                                 maxlen=max(self.lags))
            else:
                lag_buffer = deque([0.0] * max(self.lags), 
                                 maxlen=max(self.lags))
            
            # Forecast n_steps ahead for this SKU
            current_date = start_date
            
            for step in range(n_steps):
                # Advance date
                current_date = current_date + pd.Timedelta(days=1)
                
                # Build feature vector for this (sku, date)
                features = np.zeros(33)  # Total features
                
                # 1. Holiday features
                if current_date in holiday_lookup:
                    features[feature_config['holiday_indices']] = \
                        holiday_lookup[current_date]
                
                # 2. Fourier features
                day_of_year = current_date.dayofyear
                period = 365.25
                fourier_idx = 0
                for k in range(1, feature_config['n_fourier'] + 1):
                    sin_val = np.sin(2 * np.pi * k * day_of_year / period)
                    cos_val = np.cos(2 * np.pi * k * day_of_year / period)
                    features[feature_config['fourier_indices'][fourier_idx]] = sin_val
                    features[feature_config['fourier_indices'][fourier_idx+1]] = cos_val
                    fourier_idx += 2
                
                # 3. Lag features from buffer
                buffer_list = list(lag_buffer)
                for j, lag in enumerate(self.lags):
                    if len(buffer_list) >= lag:
                        features[self.lag_feature_indices[j]] = buffer_list[-lag]
                    else:
                        features[self.lag_feature_indices[j]] = \
                            buffer_list[-1] if buffer_list else 0.0
                
                # 4. Date features
                features[feature_config['date_indices']] = [
                    current_date.dayofweek,
                    current_date.day,
                    current_date.month,
                    current_date.quarter
                ]
                
                # 5. Time feature
                days_since_ref = (current_date - 
                                feature_config['reference_date']).days
                features[feature_config['time_index']] = days_since_ref
                
                # Make prediction (batch size = 1)
                X_features = features.reshape(1, -1)
                X_sku = np.array([[sku_idx]])
                
                prediction = self.predict_single_step(X_features, X_sku)[0, 0]
                
                # Store result
                results.append({
                    'sku_id': sku_id,
                    'date': current_date,
                    'forecast': prediction
                })
                
                # Update lag buffer with prediction
                lag_buffer.append(prediction)
        
        # Convert to DataFrame
        return pd.DataFrame(results)
    
    def predict_with_mode(
        self,
        X_features: np.ndarray,
        X_sku: np.ndarray,
        mode: str = 'single',
        n_steps: int = 1,
        historical_demand: Optional[Dict[int, List[float]]] = None
    ) -> np.ndarray:
        """
        Unified prediction interface.
        
        Args:
            X_features: Feature matrix
            X_sku: SKU indices
            mode: 'single' or 'multi' step prediction
            n_steps: Number of steps (for multi-step mode)
            historical_demand: Historical demand buffer (for multi-step mode)
        
        Returns:
            Predictions - shape depends on mode:
                - single: [batch_size, 1]
                - multi: [batch_size, n_steps]
        """
        if mode == 'single':
            return self.predict_single_step(X_features, X_sku)
        elif mode == 'multi':
            return self.predict_multi_step(
                X_features, X_sku, n_steps, historical_demand
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'multi'")


def prepare_historical_demand_buffer(
    df: pd.DataFrame,
    sku_col: str = 'sku_id',
    target_col: str = 'demand',
    date_col: str = 'date',
    max_lag: int = 7,
    sku_to_idx: Optional[Dict] = None
) -> Dict[int, List[float]]:
    """
    Prepare historical demand buffer for autoregressive prediction initialization.
    
    Args:
        df: DataFrame with historical data
        sku_col: SKU identifier column
        target_col: Demand column
        date_col: Date column for sorting
        max_lag: Maximum lag to keep in buffer
        sku_to_idx: Optional mapping from sku_id to integer index
    
    Returns:
        Dict mapping SKU index to list of recent demands
    """
    df = df.sort_values([sku_col, date_col]).reset_index(drop=True)
    
    historical_buffer = {}
    
    for sku, group in df.groupby(sku_col):
        # Get most recent demands (up to max_lag)
        recent_demands = group[target_col].tail(max_lag).tolist()
        
        # Map to SKU index if provided
        if sku_to_idx is not None:
            sku_idx = sku_to_idx[sku]
        else:
            sku_idx = sku
        
        historical_buffer[sku_idx] = recent_demands
    
    return historical_buffer


# Prefer the shared causal server from intermittent_features
from .intermittent_features import CausalInferenceFeatureServer  # noqa: E402
