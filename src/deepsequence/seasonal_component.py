"""
Seasonal Component for DeepSequence.
Captures multiple seasonality patterns (weekly, monthly, yearly).
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from typing import List, Optional


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
        yearly_order: int = 10,
        monthly_order: int = 4,
        weekly_order: int = 3,
        quarterly_order: int = 5
    ) -> pd.DataFrame:
        """
        Create seasonal features from date column.
        
        Uses Fourier decomposition (sin/cos) for smooth periodic patterns
        instead of integer categorical features. This matches the original
        DeepFuture architecture and provides better generalization.
        
        Args:
            future_df: Optional future DataFrame for prediction
            use_fourier: Use Fourier features instead of integer features
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
        else:
            # Legacy: Use integer categorical features
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
                       drop_out: float = 0.1):
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
        """
        if self.sr_df is None:
            self.seasonal_feature()
        
        inputs = []
        embeddings = []
        
        # ID input: use shared from Trend or create new
        n_ids = self.sr_df[self.id_var].nunique()
        embed_dim = min(embed_size, n_ids // 2 + 1)
        
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
            # Create single input for all Fourier features
            fourier_in = Input(shape=(len(fourier_cols),), name='fourier_features')
            inputs.append(fourier_in)
            embeddings.append(fourier_in)
        
        # Handle categorical features (need embeddings)
        for col in categorical_cols:
            n_unique = self.sr_df[col].nunique()
            feat_in = Input(shape=(1,), name=col)
            feat_embed = Embedding(n_unique + 1, min(embed_size, n_unique), name=f'{col}_embed')(feat_in)
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
        
        # Get all input names from the model
        for input_layer in self.s_model.inputs:
            input_name = input_layer.name.split(':')[0]  # Remove ':0' suffix
            
            if input_name == 'fourier_features':
                # Collect all Fourier columns
                fourier_cols = [col for col in df.columns if 'sin' in col or 'cos' in col]
                if fourier_cols:
                    fourier_data = df[fourier_cols].values.astype(np.float32)
                    input_dict[input_name] = fourier_data
            elif input_name == 'id':
                # Shared ID input
                input_dict[input_name] = df[self.id_var].values
            elif input_name in df.columns:
                # Categorical features
                input_dict[input_name] = df[input_name].values
        
        return input_dict
