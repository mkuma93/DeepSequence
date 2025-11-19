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
        
    def seasonal_feature(self, future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create seasonal features from date column.
        
        Args:
            future_df: Optional future DataFrame for prediction
            
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
        
        if self.weekly:
            seasonal_features['day_of_week'] = df['ds'].dt.dayofweek
            seasonal_features['week_no'] = df['ds'].dt.isocalendar().week
        
        if self.monthly:
            seasonal_features['month'] = df['ds'].dt.month
            seasonal_features['day_of_month'] = df['ds'].dt.day
            # Week of month
            seasonal_features['wom'] = df['ds'].apply(lambda x: x.day // 7)
        
        if self.yearly:
            seasonal_features['day_of_year'] = df['ds'].dt.dayofyear
            seasonal_features['year'] = df['ds'].dt.year
        
        if self.quarterly:
            seasonal_features['quarter'] = df['ds'].dt.quarter
        
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
        
        # ID embedding
        if id_input is None:
            n_ids = self.sr_df[self.id_var].nunique()
            id_in = Input(shape=(1,), name=f'{self.id_var}_seasonal')
            id_embed = Embedding(n_ids + 1, embed_size, name=f'{self.id_var}_embed')(id_in)
            id_embed = Flatten()(id_embed)
            inputs.append(id_in)
            embeddings.append(id_embed)
        else:
            embeddings.append(id_input)
        
        # Seasonal feature inputs
        seasonal_cols = [col for col in self.sr_df.columns if col not in [self.id_var, 'ds', 'year']]
        
        for col in seasonal_cols:
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
