"""
Hierarchical Attention DeepSequence for intermittent demand forecasting.

Product surface (v1.6):
1. Lightweight hierarchical DeepSequence (`components_lightweight`)
2. Causal intermittent / lag features (`intermittent_features`)
3. Gated training losses (`losses`)
4. Forecast post-process (`round_forecast`)
"""

from .losses import (
    composite_loss,
    weighted_composite_loss,
    mae_loss,
    three_term_loss_config,
    bce_mae_loss_config,
    inverse_weighted_mae_loss,
    hurdle_poisson_loss_config,
    tweedie_loss_config,
)

from .components_lightweight import build_hierarchical_model_lightweight

from .intermittent_features import (
    SKUDemandState,
    transform_panel,
    build_states_from_history,
    features_from_state,
    update_state,
    save_states,
    load_states,
    INTERMITTENT_FEATURE_NAMES,
    CausalInferenceFeatureServer,
)

from .forecast_postprocess import round_forecast

__version__ = "1.6.0"


def get_feature_config_path():
    """Path to packaged feature_config.yaml (v1.6)."""
    from pathlib import Path

    return Path(__file__).resolve().parent / "feature_config.yaml"


__all__ = [
    # Loss functions
    "composite_loss",
    "weighted_composite_loss",
    "mae_loss",
    "three_term_loss_config",
    "bce_mae_loss_config",
    "inverse_weighted_mae_loss",
    "hurdle_poisson_loss_config",
    "tweedie_loss_config",
    # Model
    "build_hierarchical_model_lightweight",
    # Causal intermittent features
    "SKUDemandState",
    "transform_panel",
    "build_states_from_history",
    "features_from_state",
    "update_state",
    "save_states",
    "load_states",
    "INTERMITTENT_FEATURE_NAMES",
    "CausalInferenceFeatureServer",
    # Post-process
    "round_forecast",
    "get_feature_config_path",
    "__version__",
]
