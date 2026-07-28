"""
Seasonal Component for DeepSequence.
Captures multiple seasonality patterns (weekly, monthly, yearly).
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Optional

# Setup precision for MPS/CPU
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

# Try tf_keras first for TensorFlow Lattice compatibility
LATTICE_AVAILABLE = False
try:
    import tensorflow_lattice as tfl
    import tf_keras as keras
    from tf_keras.layers import (Input, Dense, Embedding, Flatten,
                                  Dropout, Concatenate)
    from tf_keras.models import Model
    from tf_keras.regularizers import l1
    LATTICE_AVAILABLE = True
except ImportError:
    from tf_keras.layers import Input, Dense, Embedding, Flatten, Dropout, Concatenate
    from tf_keras.models import Model
    from tf_keras.regularizers import l1


class SeasonalComponent:
    """
    Seasonal component that models multiple seasonality patterns.
    Inspired by Prophet's additive seasonal components.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 target: List[str],
                 id_var: str,
                 horizon: int = 8,
                 weekly: bool = True,
                 monthly: bool = True,
                 yearly: bool = True,
                 quarterly: bool = False,
                 unit: str = 'w'):
        """
        Initialize seasonal component.
        
        Args:
            data: Input DataFrame with time series data
            target: List of target column names
            id_var: ID variable column name
            horizon: Forecast horizon
            weekly: Include weekly seasonality
            monthly: Include monthly seasonality
            yearly: Include yearly seasonality
            quarterly: Include quarterly seasonality
            unit: Time unit ('w' for week, 'd' for day)
        """
        self.data = data
        self.target = target
        self.id_var = id_var
        self.horizon = horizon
        self.weekly = weekly
        self.monthly = monthly
        self.yearly = yearly
        self.quarterly = quarterly
        self.unit = unit
        
        self.sr_df = None
        self.s_model = None
        
    def seasonal_feature(
        self,
        future_df: Optional[pd.DataFrame] = None,
        use_fourier: bool = True,
        use_categorical: bool = False,
        yearly_order: int = 10,
        monthly_order: int = 4,
        weekly_order: int = 3,
        quarterly_order: int = 5
    ) -> pd.DataFrame:
        """
        Create seasonal features from date column.
        
        Uses Fourier decomposition (sin/cos) for smooth periodic patterns
        and/or integer categorical features.
        
        Args:
            future_df: Optional future DataFrame for prediction
            use_fourier: Use Fourier features (sin/cos)
            use_categorical: Also include categorical features (day, week, etc.)
            yearly_order: Number of yearly Fourier terms (default: 10)
            monthly_order: Number of monthly Fourier terms (default: 4)
            weekly_order: Number of weekly Fourier terms (default: 3)
            quarterly_order: Number of quarterly Fourier terms (default: 5)
            
        Returns:
            DataFrame with seasonal features
        """
        if future_df is None:
            df = self.data.copy()
        else:
            df = future_df.copy()
        
        df['ds'] = pd.to_datetime(df['ds'])
        
        seasonal_features = {}
        seasonal_features[self.id_var] = df[self.id_var]
        seasonal_features['ds'] = df['ds']
        
        if use_fourier:
            # Use Fourier decomposition for smooth periodic patterns
            from .fourier_component import create_fourier_features
            
            fourier_df = create_fourier_features(
                dates=df['ds'],
                yearly=self.yearly,
                monthly=self.monthly,
                weekly=self.weekly,
                quarterly=self.quarterly,
                yearly_order=yearly_order,
                monthly_order=monthly_order,
                weekly_order=weekly_order,
                quarterly_order=quarterly_order
            )
            
            # Add Fourier features to seasonal_features
            for col in fourier_df.columns:
                seasonal_features[col] = fourier_df[col].values
        
        if use_categorical or not use_fourier:
            # Add integer categorical features
            if self.weekly:
                seasonal_features['day_of_week'] = df['ds'].dt.dayofweek.astype('int32')
                seasonal_features['week_no'] = df['ds'].dt.isocalendar().week.astype('int32')
            
            if self.monthly:
                seasonal_features['month'] = df['ds'].dt.month.astype('int32')
                seasonal_features['day_of_month'] = df['ds'].dt.day.astype('int32')
                # Week of month
                seasonal_features['wom'] = df['ds'].apply(lambda x: x.day // 7).astype('int32')
            
            if self.yearly:
                seasonal_features['day_of_year'] = df['ds'].dt.dayofyear.astype('int32')
                seasonal_features['year'] = df['ds'].dt.year.astype('int32')
            
            if self.quarterly:
                seasonal_features['quarter'] = df['ds'].dt.quarter.astype('int32')
        
        result_df = pd.DataFrame(seasonal_features)
        
        if future_df is None:
            self.sr_df = result_df
        
        return result_df
    
    def seasonal_model(self,
                       id_input: Optional[tf.Tensor] = None,
                       hidden: int = 1,
                       hidden_unit: int = 4,
                       hidden_act = 'relu',
                       output_act = 'linear',
                       reg: float = 0.01,
                       embed_size: int = 50,
                       drop_out: float = 0.1,
                       use_pwl_calibration: bool = False,
                       n_pwl_keypoints: int = 10,
                       pwl_on_categorical: bool = False):
        """
        Build seasonal component neural network model.
        
        Args:
            id_input: Optional ID input tensor from another model
            hidden: Number of hidden layers
            hidden_unit: Units per hidden layer
            hidden_act: Hidden layer activation
            output_act: Output activation
            reg: L1 regularization strength
            embed_size: Embedding size for ID variable
            drop_out: Dropout rate
            use_pwl_calibration: [OPTIONAL] PWL on Fourier (default: False)
                               Testing shows NO benefit - keep False
            n_pwl_keypoints: PWL keypoints per feature (default: 10)
            pwl_on_categorical: [OPTIONAL] PWL on categorical (default: False)
                              NOT RECOMMENDED - no improvement
        """
        if self.sr_df is None:
            self.seasonal_feature()
        
        inputs = []
        embeddings = []
        
        # ID input: use shared from Trend or create new
        # Use max ID + 1 for embedding size to handle all possible ID values
        max_id = int(self.sr_df[self.id_var].max())
        n_ids = max_id + 1
        embed_dim = min(embed_size, max(1, n_ids // 2))
        
        if id_input is None:
            # Create new ID input
            id_in = Input(shape=(1,), name='id')
        else:
            # Use shared ID input from Trend component
            id_in = id_input
        
        # Always add to inputs (needed for Model graph tracing)
        inputs.append(id_in)
        
        # Create ID embedding
        id_embed = Embedding(
            n_ids + 1,
            embed_dim,
            input_length=1,
            name='seasonal_id_embed'
        )(id_in)
        id_embed = Flatten()(id_embed)
        embeddings.append(id_embed)
        
        # Seasonal feature inputs
        seasonal_cols = [col for col in self.sr_df.columns if col not in [self.id_var, 'ds', 'year']]
        
        # Check if using Fourier features (continuous) or categorical features
        fourier_cols = [col for col in seasonal_cols if 'sin' in col or 'cos' in col]
        categorical_cols = [col for col in seasonal_cols if col not in fourier_cols]
        
        # Handle Fourier features (continuous, no embedding needed)
        if fourier_cols:
            if use_pwl_calibration and LATTICE_AVAILABLE:
                # Apply PWL calibration to each Fourier feature individually
                print(f"Seasonal: Using PWL calibration for {len(fourier_cols)} Fourier features")
                
                calibrated_features = []
                for col in fourier_cols:
                    # Individual input for each Fourier feature
                    feat_in = Input(shape=(1,), name=col)
                    inputs.append(feat_in)
                    
                    # PWL calibration for Fourier features (range typically [-1, 1])
                    # Use cyclic PWL for periodic seasonal patterns
                    calibrator = tfl.layers.PWLCalibration(
                        input_keypoints=np.linspace(-1.0, 1.0, num=n_pwl_keypoints).astype(DTYPE.as_numpy_dtype),
                        dtype=DTYPE,
                        output_min=-1.0,
                        output_max=1.0,
                        monotonicity='none',  # Fourier features can have any shape
                        kernel_regularizer=tf.keras.regularizers.l1(reg),
                        is_cyclic=True,  # Enable cyclic mode for seasonality
                        name=f'pwl_{col}'
                    )(feat_in)
                    calibrated_features.append(calibrator)
                
                # Concatenate all calibrated Fourier features
                if len(calibrated_features) > 1:
                    fourier_calibrated = Concatenate(name='fourier_calibrated')(calibrated_features)
                else:
                    fourier_calibrated = calibrated_features[0]
                
                embeddings.append(fourier_calibrated)
            else:
                # Standard approach: single input for all Fourier features
                fourier_in = Input(shape=(len(fourier_cols),), name='fourier_features')
                inputs.append(fourier_in)
                embeddings.append(fourier_in)
        
        # Handle categorical features
        if categorical_cols:
            if pwl_on_categorical and LATTICE_AVAILABLE:
                # Apply PWL calibration to categorical features
                print(f"Seasonal: Using PWL on {len(categorical_cols)} categorical features")
                
                for col in categorical_cols:
                    feat_in = Input(shape=(1,), name=col)
                    inputs.append(feat_in)
                    
                    # Get min/max for this categorical feature
                    col_min = float(self.sr_df[col].min())
                    col_max = float(self.sr_df[col].max())
                    
                    # PWL calibration with cyclic mode ONLY for truly periodic features
                    # where continuity across boundaries makes sense
                    # day_of_week: 0-6, wraps to 0 (Sunday->Monday or Sat->Sun)
                    # month: 1-12, wraps to 1 (Dec->Jan)
                    # quarter: 1-4, wraps to 1 (Q4->Q1)
                    # NOT cyclic: year (no wrap), day_of_year (365->1 not continuous),
                    #             day_of_month (28/30/31 varies), week_no (52->1 not smooth)
                    is_cyclic = col in ['day_of_week', 'month', 'quarter']
                    
                    calibrator = tfl.layers.PWLCalibration(
                        input_keypoints=np.linspace(col_min, col_max, 
                                                   num=n_pwl_keypoints).astype(DTYPE.as_numpy_dtype),
                        dtype=DTYPE,
                        output_min=0.0,
                        output_max=1.0,
                        monotonicity='none',
                        kernel_regularizer=tf.keras.regularizers.l1(reg),
                        is_cyclic=is_cyclic,
                        name=f'pwl_{col}'
                    )(feat_in)
                    embeddings.append(calibrator)
            else:
                # Standard embeddings for categorical features
                for col in categorical_cols:
                    n_unique = self.sr_df[col].nunique()
                    feat_in = Input(shape=(1,), name=col)
                    feat_embed = Embedding(n_unique + 1, min(embed_size, n_unique), 
                                          name=f'{col}_embed')(feat_in)
                    feat_embed = Flatten()(feat_embed)
                    inputs.append(feat_in)
                    embeddings.append(feat_embed)
        
        # Concatenate all embeddings
        if len(embeddings) > 1:
            x = Concatenate()(embeddings)
        else:
            x = embeddings[0]
        
        # Hidden layers
        for i in range(hidden):
            x = Dense(hidden_unit, activation=hidden_act, 
                     kernel_regularizer=l1(reg),
                     name=f'seasonal_hidden_{i}')(x)
            x = Dropout(drop_out)(x)
        
        # Output layer
        seasonal_output = Dense(1, activation=output_act, name='seasonal_output')(x)
        
        self.s_model = Model(inputs=inputs, outputs=seasonal_output, name='seasonal_component')
        
        return self.s_model
    
    def get_input_data(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Prepare input data for the seasonal model.
        
        Args:
            df: Optional DataFrame with seasonal features. If None, uses self.sr_df
            
        Returns:
            Dictionary mapping input names to numpy arrays
        """
        if df is None:
            df = self.sr_df
        
        if df is None:
            raise ValueError("No seasonal features available. Call seasonal_feature() first.")
        
        input_dict = {}
        
        # Get all Fourier column names from df
        fourier_cols = [col for col in df.columns if 'sin' in col or 'cos' in col]
        
        # Get all input names from the model
        for input_layer in self.s_model.inputs:
            input_name = input_layer.name.split(':')[0]  # Remove ':0' suffix
            
            if input_name == 'fourier_features':
                # Standard approach: all Fourier features in one input
                if fourier_cols:
                    fourier_data = df[fourier_cols].values.astype(DTYPE.as_numpy_dtype)
                    input_dict[input_name] = fourier_data
            elif input_name in fourier_cols:
                # PWL calibration approach: individual Fourier feature inputs
                input_dict[input_name] = df[input_name].values.reshape(-1, 1).astype(DTYPE.as_numpy_dtype)
            elif input_name == 'id':
                # Shared ID input
                input_dict[input_name] = df[self.id_var].values
            elif input_name in df.columns:
                # Categorical features
                input_dict[input_name] = df[input_name].values
        
        return input_dict
