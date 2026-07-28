"""
Temporal modeling components for time series forecasting.

Provides two-stage approach:
1. Base model generates forecasts per timestep (flexible)
2. Temporal calibration model refines forecasts using MHA on sequences
"""

from .temporal_calibration import (
    TemporalCalibrationModel,
    build_temporal_calibration_model,
    prepare_sequences_for_calibration
)

__all__ = [
    'TemporalCalibrationModel',
    'build_temporal_calibration_model',
    'prepare_sequences_for_calibration'
]
