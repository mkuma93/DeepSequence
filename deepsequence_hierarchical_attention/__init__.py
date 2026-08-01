"""
Hierarchical Attention DeepSequence for intermittent demand forecasting.

Product surface (v1.6):
1. Lightweight hierarchical DeepSequence (`components_lightweight`)
2. Causal intermittent / lag features (`intermittent_features`)
3. Gated training losses (`losses`)
4. Forecast post-process (`round_forecast`)
5. Optional residual causal transformer head (`residual_transformer`)
6. Optional hybrid temporal trunk (`hybrid_temporal`)
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
    spike_aware_loss_config,
    resolve_positive_bce_weight,
)

from .components_lightweight import (
    build_hierarchical_model_lightweight,
    build_component_readout_model,
    predict_with_components,
    estimate_zero_rate_by_sku,
    pos_weight_from_zero_rate,
    bce_sample_weights_from_sku_zero_rates,
    multioutput_bce_sample_weight_dict,
    FOURIER_PERIODS_BY_FREQUENCY,
    fourier_periods_for_frequency,
    default_fourier_periods_for_frequency,
)

from .hybrid_temporal import build_hierarchical_model_hybrid

from .frequency_presets import (
    LAGS_BY_FREQUENCY,
    default_lags_for_frequency,
    normalize_frequency,
)

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

from .residual_transformer import (
    DEFAULT_SEQUENCE_CHANNELS,
    DEFAULT_CHANNEL_COLS,
    P_DS_CHANNEL_INDEX,
    ResidualTrainModel,
    build_residual_transformer,
    build_residual_windows,
    train_residual_transformer,
    predict_residual_transformer,
    mask_predict_step,
)

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
    "spike_aware_loss_config",
    "resolve_positive_bce_weight",
    # Model
    "build_hierarchical_model_lightweight",
    "build_hierarchical_model_hybrid",
    "build_component_readout_model",
    "predict_with_components",
    "estimate_zero_rate_by_sku",
    "pos_weight_from_zero_rate",
    "bce_sample_weights_from_sku_zero_rates",
    "multioutput_bce_sample_weight_dict",
    "FOURIER_PERIODS_BY_FREQUENCY",
    "fourier_periods_for_frequency",
    "default_fourier_periods_for_frequency",
    # Frequency-aware lag / period presets
    "LAGS_BY_FREQUENCY",
    "default_lags_for_frequency",
    "normalize_frequency",
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
    # Optional residual head
    "DEFAULT_SEQUENCE_CHANNELS",
    "DEFAULT_CHANNEL_COLS",
    "P_DS_CHANNEL_INDEX",
    "ResidualTrainModel",
    "build_residual_transformer",
    "build_residual_windows",
    "train_residual_transformer",
    "predict_residual_transformer",
    "mask_predict_step",
    # Post-process
    "round_forecast",
    "get_feature_config_path",
    "__version__",
]
