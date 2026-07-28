"""
DeepSequence PWL Implementation

Piecewise Linear (PWL) calibration-based components for intermittent demand forecasting.

Note: Old PWL components (trend, seasonal, holiday, regressor) use tf_keras and are commented out.
For new projects, use the hierarchical_attention module which uses tensorflow.keras.
"""

__version__ = "1.0.0"

from .intermittent_features import (
    SKUDemandState,
    CausalInferenceFeatureServer,
    transform_panel,
    build_states_from_history,
    features_from_state,
    update_state,
    save_states,
    load_states,
    INTERMITTENT_FEATURE_NAMES,
)

__all__ = [
    "SKUDemandState",
    "CausalInferenceFeatureServer",
    "transform_panel",
    "build_states_from_history",
    "features_from_state",
    "update_state",
    "save_states",
    "load_states",
    "INTERMITTENT_FEATURE_NAMES",
]
