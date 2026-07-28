"""
Utility functions for DeepSequence.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def create_time_features(df: pd.DataFrame, date_col: str = 'ds') -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['wom'] = df[date_col].apply(lambda x: x.day // 7)
    df['year'] = df[date_col].dt.year
    df['week_no'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    return df


def prepare_data(df: pd.DataFrame,
                 target_col: str,
                 id_col: str,
                 standardize: bool = True,
                 create_lags: bool = True,
                 lag_periods: List[int] = [1, 4, 52]) -> pd.DataFrame:
    df = df.copy()
    if standardize:
        mu = df.groupby([id_col])[target_col].mean().reset_index()
        mu = mu.rename(columns={target_col: 'mean'})
        std = df.groupby([id_col])[target_col].std().reset_index()
        std = std.rename(columns={target_col: 'std'})
        df = df.merge(mu, on=[id_col], how='left')
        df = df.merge(std, on=[id_col], how='left')
        df[f't{target_col}'] = (df[target_col] - df['mean']) / df['std']
        target_col = f't{target_col}'
    if create_lags:
        for lag in lag_periods:
            df[f'lag{lag}'] = df.groupby(id_col)[target_col].shift(lag)
    df.fillna(-1, inplace=True)
    return df


def train_val_test_split(df: pd.DataFrame,
                         date_col: str = 'ds',
                         val_weeks: int = 8,
                         test_weeks: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(weeks=test_weeks)
    val_start = test_start - pd.Timedelta(weeks=val_weeks)
    train = df[df[date_col] < val_start].copy()
    val = df[(df[date_col] >= val_start) & (df[date_col] < test_start)].copy()
    test = df[df[date_col] >= test_start].copy()
    return train, val, test


def inverse_transform(predictions: np.ndarray, mean: float, std: float) -> np.ndarray:
    return predictions * std + mean


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-7) -> float:
    """Calculate MAPE with epsilon to prevent division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def encode_categorical(df: pd.DataFrame, col: str, encoder = None):
    try:
        import category_encoders as ce
    except ImportError:
        raise ImportError("Please install category_encoders: pip install category-encoders")
    df = df.copy()
    if encoder is None:
        encoder = ce.OrdinalEncoder()
        df[f'{col}_encoded'] = encoder.fit_transform(df[col])
    else:
        df[f'{col}_encoded'] = encoder.transform(df[col])
    return df, encoder
