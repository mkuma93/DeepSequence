"""
DeepSequence: A Prophet-inspired deep learning architecture for time series forecasting.

This package provides modules for building and training the DeepSequence architecture,
which decomposes forecasting into seasonal and regression components.

Author: Mritunjay Kumar
Year: 2025
"""

__version__ = "0.1.0"
__author__ = "Mritunjay Kumar"

from .seasonal_component import SeasonalComponent
from .regressor_component import RegressorComponent
from .intermittent_handler import IntermittentHandler, apply_intermittent_mask
from .tabnet_encoder import TabNetEncoder, create_tabnet_encoder
from .unit_norm import UnitNorm, UnitNormDense, apply_unit_norm
from .cross_layer import CrossLayer, CrossNetwork, apply_cross_network
from .model import DeepSequenceModel
from .utils import create_time_features, prepare_data

__all__ = [
    'SeasonalComponent',
    'RegressorComponent',
    'IntermittentHandler',
    'apply_intermittent_mask',
    'TabNetEncoder',
    'create_tabnet_encoder',
    'UnitNorm',
    'UnitNormDense',
    'apply_unit_norm',
    'CrossLayer',
    'CrossNetwork',
    'apply_cross_network',
    'DeepSequenceModel',
    'create_time_features',
    'prepare_data'
]
