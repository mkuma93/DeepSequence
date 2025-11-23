"""
DeepSequence PWL Implementation

Piecewise Linear (PWL) calibration-based components for intermittent demand forecasting.
"""

from .trend_component import TrendComponent
from .seasonal_component import SeasonalComponent
from .holiday_component import HolidayComponent
from .regressor_component import RegressorComponent
from .intermittent_handler import IntermittentHandler
from .combination_layer import CombinationLayer
from .model import DeepSequencePWL
from . import config
from .utils import *

__version__ = "1.0.0"
__all__ = [
    'TrendComponent',
    'SeasonalComponent',
    'HolidayComponent',
    'RegressorComponent', 
    'IntermittentHandler',
    'CombinationLayer',
    'DeepSequencePWL',
    'config'
]