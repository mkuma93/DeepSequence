"""
Holiday Component for modeling holiday effects in time series.

Based on the original DeepFuture HolidayComponent.py but modernized.
Uses holiday distance with PWL calibration and Lattice layers.
Can work standalone or integrate with DeepSequence architecture.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Literal
import tensorflow as tf
from tf_keras.layers import (
    Input, Dense, Concatenate, Dropout,
    Embedding, Reshape, Flatten, BatchNormalization, Lambda
)
from tf_keras.models import Model
from tf_keras.regularizers import l1
import tensorflow_lattice as tfl
from pandas.tseries.offsets import Easter
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from dateutil.relativedelta import MO, SU, TH


class ExtendedUSHolidayCalendar(AbstractHolidayCalendar):
    """
    Extended US Holiday calendar including retail/commercial events.
    Used for calculating distance from nearest holiday.
    """
    rules = [
        Holiday("NewYearsDay", month=1, day=1),
        Holiday("MLKDay", month=1, day=1,
                offset=pd.DateOffset(weekday=MO(3))),
        Holiday("PresidentsDay", month=2, day=1,
                offset=pd.DateOffset(weekday=MO(3))),
        Holiday("ValentinesDay", month=2, day=14),
        Holiday("Easter", month=1, day=1, offset=[Easter()]),
        Holiday("MothersDay", month=5, day=1,
                offset=pd.DateOffset(weekday=SU(2))),
        Holiday("MemorialDay", month=5, day=31,
                offset=pd.DateOffset(weekday=MO(-1))),
        Holiday("FathersDay", month=6, day=1,
                offset=pd.DateOffset(weekday=SU(3))),
        Holiday("IndependenceDay", month=7, day=4),
        Holiday("LaborDay", month=9, day=1,
                offset=pd.DateOffset(weekday=MO(1))),
        Holiday("Halloween", month=10, day=31),
        Holiday("Thanksgiving", month=11, day=1,
                offset=pd.DateOffset(weekday=TH(4))),
        Holiday("BlackFriday", month=11, day=1,
                offset=[pd.DateOffset(weekday=TH(4)),
                        pd.DateOffset(days=1)]),
        Holiday("Christmas", month=12, day=25),
        Holiday("NewYearsEve", month=12, day=31),
    ]


class USHolidayCalendar(AbstractHolidayCalendar):
    """
    US Holiday calendar with major holidays including retail events.
    """
    rules = [
        Holiday("new_years_day", month=1, day=1),
        Holiday("mlk_day", month=1, day=1, offset=pd.DateOffset(weekday=MO(3))),
        Holiday("super_bowl", month=2, day=1, offset=pd.DateOffset(weekday=SU(1))),
        Holiday("valentines_day", month=2, day=14),
        Holiday("presidents_day", month=2, day=1, offset=pd.DateOffset(weekday=MO(3))),
        Holiday("easter", month=1, day=1, offset=[Easter()]),
        Holiday("mothers_day", month=5, day=1, offset=pd.DateOffset(weekday=SU(2))),
        Holiday("memorial_day", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))),
        Holiday("july_4th", month=7, day=4),
        Holiday("labor_day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))),
        Holiday("columbus_day", month=10, day=1, offset=pd.DateOffset(weekday=MO(2))),
        Holiday("halloween", month=10, day=31),
        Holiday("veterans_day", month=11, day=11),
        Holiday("thanksgiving", month=11, day=1, offset=pd.DateOffset(weekday=TH(4))),
        Holiday("black_friday", month=11, day=1, 
                offset=[pd.DateOffset(weekday=TH(4)), pd.DateOffset(days=1)]),
        Holiday("cyber_monday", month=11, day=1,
                offset=[pd.DateOffset(weekday=TH(4)), pd.DateOffset(days=4)]),
        Holiday("christmas", month=12, day=25),
    ]


class HolidayFeatureGenerator:
    """
    Generate holiday features with configurable windows.
    
    Creates binary features for days around holidays (e.g., -7 to +7 days).
    Each holiday gets multiple features based on the window size.
    
    Args:
        start_date: Start date for holiday calendar
        end_date: End date for holiday calendar
        lower_window: Days before holiday to include (negative, e.g., -7)
        upper_window: Days after holiday to include (positive, e.g., 7)
        unit: Time unit ('d' for day, 'w' for week)
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        lower_window: int = -7,
        upper_window: int = 7,
        unit: str = 'd'
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.lower_window = lower_window
        self.upper_window = upper_window
        self.unit = unit
        
        # Get holiday calendar
        self.holidays_df = self._get_holidays()
        self.holiday_names = sorted(self.holidays_df['holiday'].unique())
        
    def _get_holidays(self) -> pd.DataFrame:
        """Get holidays from calendar."""
        # Get holidays in date range
        holiday_series = USHolidayCalendar().holidays(
            self.start_date,
            self.end_date,
            return_name=True
        )
        
        # Create DataFrame with all dates
        dates = pd.date_range(self.start_date, self.end_date).to_series()
        holidays_df = pd.concat([dates, holiday_series], axis=1)
        holidays_df.columns = ['ds', 'holiday']
        holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        
        # Aggregate by unit if needed
        if self.unit == 'w':
            holidays_df['ds'] = holidays_df['ds'].dt.to_period('W').dt.to_timestamp()
        elif self.unit == 'm':
            holidays_df['ds'] = holidays_df['ds'].dt.to_period('MS').dt.to_timestamp()
        
        # Remove NaN holidays and sort
        holidays_df = holidays_df.dropna().sort_values(['holiday', 'ds']).reset_index(drop=True)
        
        return holidays_df
    
    def create_holiday_features(self, dates: pd.Series) -> pd.DataFrame:
        """
        Create holiday window features for given dates.
        
        Args:
            dates: Series of dates to create features for
            
        Returns:
            DataFrame with holiday window features (binary indicators)
        """
        df = pd.DataFrame({'ds': pd.to_datetime(dates)})
        
        # Create features for each holiday and window offset
        for holiday_name in self.holiday_names:
            holiday_dates = self.holidays_df[
                self.holidays_df['holiday'] == holiday_name
            ]['ds']
            
            # Create window features
            for offset in range(self.lower_window, self.upper_window + 1):
                feature_name = f'{holiday_name}_offset_{offset}'
                
                if self.unit == 'd':
                    shifted_dates = holiday_dates + pd.Timedelta(days=offset)
                elif self.unit == 'w':
                    shifted_dates = holiday_dates + pd.Timedelta(weeks=offset)
                else:
                    shifted_dates = holiday_dates + pd.DateOffset(months=offset)
                
                # Create binary indicator
                df[feature_name] = df['ds'].isin(shifted_dates).astype(np.float32)
        
        return df.drop(columns=['ds'])


class HolidayComponent:
    """
    Holiday component using PWL calibration for holiday effects.
    
    Models the impact of holidays on the target variable using:
    1. Holiday window features (binary indicators for days around holidays)
    2. PWL calibration layers for each holiday feature
    3. Lattice layer for feature interactions
    4. ID embedding for entity-specific holiday effects
    
    REQUIRES shared ID input from Trend component (raises ValueError if not provided).
    
    Args:
        data: Input DataFrame with time series data
        target: List of target column names
        id_var: ID variable column name
        lower_window: Days before holiday (negative, default: -7)
        upper_window: Days after holiday (positive, default: 7)
        unit: Time unit ('d' for day, 'w' for week)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target: List[str],
        id_var: str,
        lower_window: int = -7,
        upper_window: int = 7,
        unit: str = 'd'
    ):
        self.data = data
        self.target = target
        self.id_var = id_var
        self.lower_window = lower_window
        self.upper_window = upper_window
        self.unit = unit
        
        self.holiday_df = None
        self.h_model = None
        
    def create_holiday_features(
        self,
        future_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create holiday features from date column.
        
        Args:
            future_df: Optional future DataFrame for prediction
            
        Returns:
            DataFrame with holiday features
        """
        if future_df is None:
            df = self.data.copy()
        else:
            df = future_df.copy()
        
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Get date range
        start_date = df['ds'].min()
        end_date = df['ds'].max()
        
        # Create holiday feature generator
        generator = HolidayFeatureGenerator(
            start_date=str(start_date.date()),
            end_date=str(end_date.date()),
            lower_window=self.lower_window,
            upper_window=self.upper_window,
            unit=self.unit
        )
        
        # Generate features
        holiday_features = generator.create_holiday_features(df['ds'])
        
        # Add ID and date
        result_df = pd.DataFrame({
            self.id_var: df[self.id_var].values,
            'ds': df['ds'].values
        })
        
        # Add all holiday features
        for col in holiday_features.columns:
            result_df[col] = holiday_features[col].values
        
        if future_df is None:
            self.holiday_df = result_df
        
        return result_df
    
    def holiday_model(
        self,
        id_input: tf.Tensor,
        lat_unit: int = 16,
        lattice_size: int = 2,
        hidden: int = 2,
        hidden_unit: int = 32,
        hidden_act: str = 'relu',
        output_act: str = 'linear',
        reg: float = 0.01,
        drop_out: float = 0.2,
        embed_size: int = 50
    ) -> Model:
        """
        Build holiday component neural network model with PWL calibration.
        
        Args:
            id_input: REQUIRED shared ID input tensor from Trend component
            lat_unit: Number of lattice units
            lattice_size: Size of each lattice dimension
            hidden: Number of hidden layers
            hidden_unit: Units per hidden layer
            hidden_act: Hidden layer activation
            output_act: Output activation
            reg: L1 regularization strength
            drop_out: Dropout rate
            embed_size: Embedding size for ID variable
            
        Returns:
            Keras Model for holiday component
            
        Raises:
            ValueError: If id_input is None (shared ID is required)
        """
        if id_input is None:
            raise ValueError(
                "Holiday component requires shared ID input from Trend component. "
                "Pass id_input=trend_component.id_input when building holiday model."
            )
        
        if self.holiday_df is None:
            self.create_holiday_features()
        
        inputs = []
        calibrators = []
        
        # Get holiday feature columns
        holiday_cols = [
            col for col in self.holiday_df.columns
            if col not in [self.id_var, 'ds']
        ]
        
        print(f"Holiday: Using {len(holiday_cols)} holiday window features")
        
        # Calculate calibration size
        window_size = abs(self.lower_window) + self.upper_window
        c_size = max(3, window_size)  # At least 3 keypoints
        
        # PWL Calibration for each holiday feature
        for col in holiday_cols:
            feat_input = Input(shape=(1,), name=col)
            inputs.append(feat_input)
            
            # PWL calibration
            calibrated = tfl.layers.PWLCalibration(
                input_keypoints=np.linspace(
                    self.holiday_df[col].min(),
                    self.holiday_df[col].max(),
                    num=c_size
                ),
                dtype=tf.float32,
                units=lat_unit,
                kernel_regularizer=tf.keras.regularizers.l1(reg),
                name=f'{col}_calibrator'
            )(feat_input)
            
            calibrators.append(calibrated)
        
        # Concatenate calibrated features
        if len(calibrators) > 1:
            concat_calibrators = Concatenate()(calibrators)
        else:
            concat_calibrators = calibrators[0]
        
        # Add ID embedding
        n_ids = self.data[self.id_var].nunique()
        embed_dim = min(embed_size, n_ids // 2 + 1)
        
        # Use shared ID input
        inputs.append(id_input)
        id_embed = Embedding(
            n_ids + 1,
            embed_dim,
            input_length=1,
            name='holiday_id_embed'
        )(id_input)
        id_embed = Flatten()(id_embed)
        
        # Combine calibrators with ID embedding
        combined = Concatenate()([concat_calibrators, id_embed])
        
        # Lattice layer
        lattice_sizes = [lattice_size] * int(np.sqrt(lat_unit))
        dense_for_lattice = Dense(
            lat_unit * len(lattice_sizes),
            activation='sigmoid',
            name='holiday_dense'
        )(combined)
        
        reshaped = Reshape(
            target_shape=(lat_unit, len(lattice_sizes))
        )(dense_for_lattice)
        
        lattice_out = tfl.layers.Lattice(
            lattice_sizes=lattice_sizes,
            units=lat_unit,
            kernel_regularizer=tf.keras.regularizers.l1(reg),
            name='holiday_lattice'
        )(reshaped)
        
        # Hidden layers with residual connections
        x = lattice_out
        for i in range(hidden):
            x = Dense(
                hidden_unit,
                activation=hidden_act,
                kernel_regularizer=l1(reg),
                name=f'holiday_hidden_{i}'
            )(x)
            x = Dropout(drop_out)(x)
            x = BatchNormalization()(x)
            # Residual connection
            x = Concatenate()([x, combined])
        
        # Output layer
        holiday_output = Dense(
            1,
            activation=output_act,
            name='holiday_output'
        )(x)
        
        self.h_model = Model(
            inputs=inputs,
            outputs=holiday_output,
            name='holiday_component'
        )
        
        return self.h_model
    
    def get_input_data(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        Prepare input data for the holiday model.
        
        Args:
            df: Optional DataFrame with holiday features
            
        Returns:
            Dictionary mapping input names to numpy arrays
        """
        if df is None:
            df = self.holiday_df
        
        if df is None:
            raise ValueError(
                "No holiday features available. "
                "Call create_holiday_features() first."
            )
        
        input_dict = {}
        
        # Get all input names from the model
        for input_layer in self.h_model.inputs:
            input_name = input_layer.name.split(':')[0]
            
            if input_name == 'id':
                # Shared ID input
                input_dict[input_name] = df[self.id_var].values
            elif input_name in df.columns:
                # Holiday features
                input_dict[input_name] = df[input_name].values
        
        return input_dict
