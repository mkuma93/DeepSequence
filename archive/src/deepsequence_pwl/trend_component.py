"""
Trend Component for DeepSequence.
Models long-term growth/decline patterns with PWL calibration and lattice layers.
Based on original DeepFuture v1/v2 implementation.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Embedding, Flatten, Reshape,
                                    Dropout, Concatenate, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from datetime import datetime
from typing import List, Optional

# Check for tensorflow_lattice availability
LATTICE_AVAILABLE = False
try:
    import tensorflow_lattice as tfl  # noqa: F401
    LATTICE_AVAILABLE = True
except ImportError:
    pass


class TrendComponent:
    """
    Trend component using PWL (Piecewise Linear) Calibration and Lattice layers.
    
    Key features matching original DeepFuture implementation:
    - PWLCalibration layer for learned changepoints (not manual ReLU features)
    - Absolute time representation (days since epoch)
    - Lattice layers with proper reshaping
    - Residual connections back to calibrator
    - Per-ID embeddings for different SKU trends
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 target: List[str],
                 id_var: str,
                 time_col: str = 'ds',
                 use_changepoints: bool = True,
                 n_changepoints: int = 25,
                 changepoint_range: float = 0.8,
                 lat_unit: int = 32,
                 lattice_size: int = 2,
                 hidden_unit: int = 32,
                 hidden_layer: int = 1,
                 drop_out: float = 0.2,
                 L1: float = 0.05,
                 hidden_act = 'relu',
                 output_act = 'relu',
                 embed_size: int = 50):
        """
        Initialize trend component with PWL calibration and lattice layers.
        
        Args:
            data: Input DataFrame with time series data
            target: List of target column names
            id_var: ID variable column name
            time_col: Time column name (datetime)
            use_changepoints: Use PWL calibration with changepoints (default: True)
            n_changepoints: Number of keypoints for PWL calibration
            changepoint_range: Portion of history for changepoints (0.8 = first 80%)
            lat_unit: Output dimensionality of PWL calibration (e.g., 32, 64)
            lattice_size: Size of each lattice dimension (e.g., 2 for 2x2x... grid)
            hidden_unit: Hidden layer units
            hidden_layer: Number of hidden layers
            drop_out: Dropout rate
            L1: L1 regularization strength
            hidden_act: Hidden activation function
            output_act: Output activation function
            embed_size: Embedding size for ID variable
        """
        self.data = data.copy()
        self.data[time_col] = pd.to_datetime(self.data[time_col])
        self.data = self.data.sort_values(by=[time_col])
        
        self.target = target
        self.id_var = id_var
        self.time_col = time_col
        self.use_changepoints = use_changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.lat_unit = lat_unit
        self.lattice_size = lattice_size
        self.hidden_unit = hidden_unit
        self.hidden_layer = hidden_layer
        self.drop_out = drop_out
        self.L1 = L1
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.embed_size = embed_size
        
        self.t_model = None
        self.id_input = None  # Shared ID input tensor (exposed for other components)
        self.time_min_days = None
        self.time_max_days = None
        self.n_ids = None
        self.c_size = None  # Actual number of changepoints used
    
    def _compute_time_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert datetime to days since epoch (1970-01-01).
        
        Args:
            df: DataFrame with time column
            
        Returns:
            Array of time in days since epoch
        """
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        
        # Days since epoch (matches original implementation)
        epoch = pd.Timestamp('1970-01-01')
        time_days = (df[self.time_col] - epoch).dt.total_seconds() / (24 * 3600)
        
        # Store min/max for future predictions
        if self.time_min_days is None:
            self.time_min_days = time_days.min()
            self.time_max_days = time_days.max()
            
            # Calculate actual changepoints to use
            if self.use_changepoints:
                if self.id_var is not None:
                    # Max length per ID + buffer
                    max_size = df.groupby([self.id_var])[self.time_col].count().max()
                else:
                    max_size = len(df)
                
                self.c_size = int(np.floor(max_size * self.changepoint_range))
                if self.c_size > self.n_changepoints:
                    self.c_size = self.n_changepoints
                
                print(f"Trend: Using {self.c_size} changepoints for PWL calibration")
        
        return time_days.values.astype(np.float32)
    
    def create_trend_features(self, future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare data for trend model (no manual feature creation needed).
        PWLCalibration will learn the piecewise linear function automatically.
        
        Args:
            future_df: Optional future DataFrame for prediction
            
        Returns:
            DataFrame with time in days since epoch
        """
        if future_df is None:
            df = self.data.copy()
        else:
            df = future_df.copy()
        
        # Compute time features
        time_days = self._compute_time_features(df)
        
        # Create output dataframe
        result = pd.DataFrame()
        result[self.id_var] = df[self.id_var].values
        result[self.time_col] = df[self.time_col].values
        result['time_days'] = time_days
        
        return result
    
    def trend_model(self, id_input: Optional[tf.Tensor] = None):
        """
        Build trend model with PWL calibration, lattice layers, residuals.
        
        Original DeepFuture architecture:
        1. Time input → PWLCalibration (learned changepoints)
        2. Concatenate with ID embedding
        3. Dense → Reshape → Lattice
        4. Hidden layers with residual connections back to calibrator
        5. Output
        
        Args:
            id_input: Optional shared ID input tensor
            
        Returns:
            Keras Model for trend component
        """
        if not LATTICE_AVAILABLE:
            raise ValueError(
                "tensorflow_lattice required. "
                "Install: pip install tensorflow_lattice"
            )
        
        inputs = []
        
        # Time input (days since epoch)
        time_input = Input(shape=(1,), name='time_input')
        inputs.append(time_input)
        
        # PWL Calibration with learned changepoints
        if self.use_changepoints and self.c_size is not None:
            # Create keypoints for piecewise linear function
            keypoints = np.linspace(
                self.time_min_days,
                self.time_max_days,
                num=self.c_size
            )
            
            calibrator = tfl.layers.PWLCalibration(
                input_keypoints=keypoints,
                units=self.lat_unit,
                dtype=tf.float32,
                is_cyclic=True,  # For time series
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.L1),
                name='time_calibrator'
            )(time_input)
        else:
            # Fallback: simple dense layer
            calibrator = Dense(
                self.lat_unit,
                activation='sigmoid',
                kernel_regularizer=l1(self.L1),
                name='time_calibrator'
            )(time_input)
        
        # Handle ID input: create new or use provided (for sharing)
        if self.id_var is not None:
            if self.n_ids is None:
                self.n_ids = self.data[self.id_var].nunique()
            
            if id_input is None:
                # Create new ID input (will be shared with other components)
                id_in = Input(shape=(1,), name='id')
                inputs.append(id_in)
                self.id_input = id_in  # Store for sharing
            else:
                # Use provided shared ID input
                id_in = id_input
                self.id_input = id_input
                # Add to inputs only if not already there
                if id_input not in inputs:
                    inputs.append(id_in)
            
            # Create ID embedding
            id_embed = Embedding(
                self.n_ids + 1,
                self.embed_size,
                input_length=1,
                name='id_embed'
            )(id_in)
            id_embed = Flatten()(id_embed)
            
            # Concatenate calibrator with ID embedding
            calibrator = Concatenate(name='calibrator_with_id')(
                [calibrator, id_embed]
            )
        
        # Prepare for lattice: Dense → Reshape
        # lattice_sizes determines the grid: e.g., [2,2,2,2] for 4D lattice
        lattice_sizes = [self.lattice_size] * int(np.sqrt(self.lat_unit))
        
        x = Dense(
            self.hidden_unit * len(lattice_sizes),
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.L1),
            name='trend_pre_lattice'
        )(calibrator)
        
        # Reshape for lattice: (batch, hidden_unit, num_lattices)
        x = Reshape(
            target_shape=(self.hidden_unit, len(lattice_sizes)),
            name='trend_reshape'
        )(x)
        
        # Lattice layer
        x = tfl.layers.Lattice(
            lattice_sizes=lattice_sizes,
            units=self.hidden_unit,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.L1),
            name='trend_lattice'
        )(x)
        
        # Reshape back to flat
        x = Reshape(target_shape=(self.hidden_unit,), name='trend_flatten')(x)
        
        # Hidden layers with RESIDUAL connections
        for i in range(self.hidden_layer):
            x = Dense(
                self.hidden_unit,
                activation=self.hidden_act,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.L1),
                name=f'trend_hidden_{i}'
            )(x)
            x = Dropout(self.drop_out, name=f'trend_dropout_{i}')(x)
            x = BatchNormalization(name=f'trend_bn_{i}')(x)
            
            # RESIDUAL: Concatenate back to calibrator output
            x = Concatenate(name=f'trend_residual_{i}')([x, calibrator])
        
        # Output layer
        trend_output = Dense(
            1,
            activation=self.output_act,
            name='trend_output'
        )(x)
        
        # Create model
        self.t_model = Model(
            inputs=inputs,
            outputs=trend_output,
            name='trend_component'
        )
        
        return self.t_model
    
    def get_input_data(self, df: pd.DataFrame) -> dict:
        """
        Prepare input data for the trend model.
        
        Args:
            df: DataFrame with time_days column from create_trend_features()
            
        Returns:
            Dictionary mapping input names to numpy arrays
        """
        if df is None or 'time_days' not in df.columns:
            raise ValueError(
                "Invalid data. Call create_trend_features() first."
            )
        
        input_dict = {}
        
        # Time input (days since epoch)
        input_dict['time_input'] = df['time_days'].values.astype(np.float32)
        
        # ID input (shared name 'id')
        if self.id_var is not None:
            id_input_name = 'id'
            if id_input_name in [inp.name for inp in self.t_model.inputs]:
                input_dict[id_input_name] = df[self.id_var].values
        
        return input_dict
    
    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        """
        Predict trend for future dates.
        
        Args:
            future_df: DataFrame with future dates and IDs
            
        Returns:
            Array of trend predictions
        """
        if self.t_model is None:
            raise ValueError("Model not built. Call trend_model() first.")
        
        # Compute time features for future dates
        future_trend = self.create_trend_features(future_df)
        
        # Get input data
        input_data = self.get_input_data(future_trend)
        
        # Predict
        predictions = self.t_model.predict(input_data)
        
        return predictions.flatten()
    
    def summary(self):
        """Print model summary."""
        if self.t_model is not None:
            print("\n=== Trend Component Summary ===")
            print(f"PWL Calibration: {self.c_size} changepoints")
            print(f"Lattice units: {self.lat_unit}")
            print(f"Lattice size: {self.lattice_size}")
            print(f"Hidden layers: {self.hidden_layer}")
            print(f"Hidden units: {self.hidden_unit}")
            print(f"Time range: {self.time_min_days:.1f} to "
                  f"{self.time_max_days:.1f} days")
            print("\nModel architecture:")
            self.t_model.summary()
        else:
            print("Model not built yet. Call trend_model() first.")
