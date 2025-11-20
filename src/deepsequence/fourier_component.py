"""
Fourier Component for extracting seasonal patterns using sin/cos decomposition.

Based on the original DeepFuture FourierComponent.py but modernized and simplified.
Generates Fourier features for capturing periodic patterns in time series data.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Union


class FourierFeatureGenerator:
    """
    Generate Fourier features (sin/cos) for capturing seasonal patterns.
    
    Fourier decomposition represents periodic patterns as combinations of
    sine and cosine functions, providing a smooth and interpretable way
    to model seasonality.
    
    Args:
        period: The period of seasonality (e.g., 365.25 for yearly, 7 for weekly)
        order: Number of Fourier terms (harmonics) to generate
               Higher order captures more complex patterns
               Total features = 2 * order (sin and cos for each harmonic)
    
    Example:
        # Weekly seasonality with 3 harmonics (6 features)
        weekly = FourierFeatureGenerator(period=7, order=3)
        features = weekly.generate_features(dates)
    """
    
    def __init__(self, period: float, order: int):
        """
        Initialize Fourier feature generator.
        
        Args:
            period: Period of the seasonality pattern
            order: Number of Fourier terms (k in original implementation)
        """
        if order < 1:
            raise ValueError("order must be a positive integer")
        if 2 * order > period:
            raise ValueError(
                f"order must not be greater than period//2 "
                f"(got order={order}, period={period})"
            )
        
        self.period = period
        self.order = order
        
        # Precompute frequencies: (1/m, 2/m, ..., k/m)
        self.frequencies = ((np.arange(order) + 1) / period).astype(np.float64)
    
    def generate_features(
        self,
        dates: Union[pd.Series, pd.DatetimeIndex, np.ndarray],
        prefix: str = 'fourier'
    ) -> pd.DataFrame:
        """
        Generate Fourier features for given dates.
        
        Args:
            dates: Dates to generate features for
            prefix: Prefix for feature names
        
        Returns:
            DataFrame with sin/cos Fourier features
            Columns: ['{prefix}_sin_1', '{prefix}_cos_1', '{prefix}_sin_2', ...]
        """
        # Convert dates to numeric time (days since epoch)
        if isinstance(dates, pd.Series):
            dates = dates.values
        if isinstance(dates, pd.DatetimeIndex):
            dates = dates.to_numpy()
        
        # Convert to days since Unix epoch
        if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
            epoch = pd.Timestamp('1970-01-01')
            times = (pd.to_datetime(dates) - epoch).total_seconds() / (3600 * 24)
            times = times.values if hasattr(times, 'values') else times
        else:
            times = dates
        
        # Generate Fourier features
        features = []
        feature_names = []
        
        for i, freq in enumerate(self.frequencies, 1):
            # Sin component
            sin_feature = np.sin(2 * np.pi * freq * times)
            features.append(sin_feature)
            feature_names.append(f'{prefix}_sin_{i}')
            
            # Cos component
            cos_feature = np.cos(2 * np.pi * freq * times)
            features.append(cos_feature)
            feature_names.append(f'{prefix}_cos_{i}')
        
        # Stack features and create DataFrame
        features_array = np.column_stack(features)
        features_df = pd.DataFrame(features_array, columns=feature_names)
        
        return features_df


class SeasonalFourierExtractor:
    """
    Extract multiple seasonal patterns using Fourier decomposition.
    
    Supports common seasonal patterns:
    - Yearly (period=365.25)
    - Monthly (period=30.44)
    - Weekly (period=7)
    - Quarterly (period=91.31)
    
    Args:
        yearly: Number of yearly Fourier terms (default: 10)
        monthly: Number of monthly Fourier terms (default: 4)
        weekly: Number of weekly Fourier terms (default: 3)
        quarterly: Number of quarterly Fourier terms (default: 5)
    
    Example:
        extractor = SeasonalFourierExtractor(yearly=10, weekly=3)
        features = extractor.extract_features(data['ds'])
    """
    
    def __init__(
        self,
        yearly: Optional[int] = None,
        monthly: Optional[int] = None,
        weekly: Optional[int] = None,
        quarterly: Optional[int] = None
    ):
        """
        Initialize seasonal Fourier extractor.
        
        Args:
            yearly: Order for yearly seasonality (None to disable)
            monthly: Order for monthly seasonality (None to disable)
            weekly: Order for weekly seasonality (None to disable)
            quarterly: Order for quarterly seasonality (None to disable)
        """
        self.generators = {}
        
        if yearly is not None:
            self.generators['yearly'] = FourierFeatureGenerator(
                period=365.25,
                order=yearly
            )
        
        if monthly is not None:
            self.generators['monthly'] = FourierFeatureGenerator(
                period=30.44,  # Average days per month
                order=monthly
            )
        
        if weekly is not None:
            self.generators['weekly'] = FourierFeatureGenerator(
                period=7,
                order=weekly
            )
        
        if quarterly is not None:
            self.generators['quarterly'] = FourierFeatureGenerator(
                period=91.31,  # Average days per quarter
                order=quarterly
            )
    
    def extract_features(
        self,
        dates: Union[pd.Series, pd.DatetimeIndex, np.ndarray]
    ) -> pd.DataFrame:
        """
        Extract all configured Fourier features.
        
        Args:
            dates: Dates to generate features for
        
        Returns:
            DataFrame with all Fourier features concatenated
        """
        if len(self.generators) == 0:
            raise ValueError("No seasonal patterns configured")
        
        all_features = []
        
        for season_name, generator in self.generators.items():
            features = generator.generate_features(dates, prefix=season_name)
            all_features.append(features)
        
        # Concatenate all features
        result = pd.concat(all_features, axis=1)
        
        return result
    
    def get_feature_count(self) -> int:
        """Get total number of features that will be generated."""
        return sum(gen.order * 2 for gen in self.generators.values())
    
    def get_feature_info(self) -> dict:
        """Get information about configured features."""
        info = {}
        for name, gen in self.generators.items():
            info[name] = {
                'period': gen.period,
                'order': gen.order,
                'features': gen.order * 2
            }
        return info


def create_fourier_features(
    dates: Union[pd.Series, pd.DatetimeIndex, np.ndarray],
    yearly: bool = True,
    monthly: bool = False,
    weekly: bool = True,
    quarterly: bool = False,
    yearly_order: int = 10,
    monthly_order: int = 4,
    weekly_order: int = 3,
    quarterly_order: int = 5
) -> pd.DataFrame:
    """
    Convenience function to create Fourier features.
    
    Args:
        dates: Dates to generate features for
        yearly: Include yearly seasonality
        monthly: Include monthly seasonality
        weekly: Include weekly seasonality
        quarterly: Include quarterly seasonality
        yearly_order: Order for yearly Fourier terms
        monthly_order: Order for monthly Fourier terms
        weekly_order: Order for weekly Fourier terms
        quarterly_order: Order for quarterly Fourier terms
    
    Returns:
        DataFrame with Fourier features
    
    Example:
        features = create_fourier_features(
            data['ds'],
            yearly=True,
            weekly=True,
            yearly_order=10,
            weekly_order=3
        )
    """
    extractor = SeasonalFourierExtractor(
        yearly=yearly_order if yearly else None,
        monthly=monthly_order if monthly else None,
        weekly=weekly_order if weekly else None,
        quarterly=quarterly_order if quarterly else None
    )
    
    return extractor.extract_features(dates)
