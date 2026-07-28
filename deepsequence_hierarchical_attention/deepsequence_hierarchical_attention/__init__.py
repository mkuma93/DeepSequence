"""
Hierarchical Attention Architecture for Intermittent Demand Forecasting.

Main pieces:
1. Lightweight hierarchical DeepSequence (components_lightweight.py)
2. TabNet hierarchical attention (components.py)
3. Causal intermittent features (intermittent_features.py)
4. Residual causal transformer head (residual_transformer.py)
"""

# Loss functions
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

# Lightweight implementation (optimized for production)
from .components_lightweight import build_hierarchical_model_lightweight

# TabNet-based hierarchical attention (full feature attention)
from .components import (
    DeepSequencePWLHierarchical,
    HierarchicalAttentionIntermittentHandler,
    TrendComponentBuilder,
    SeasonalComponentBuilder,
    HolidayComponentBuilder,
    RegressorComponentBuilder
)

# TabNet encoder components
from .tabnet import (
    TabNetEncoder,
    GhostBatchNormalization,
    GLUBlock,
    AttentiveTransformer,
    FeatureTransformer
)

# Model creation utilities
from .model import (
    create_hierarchical_model,
    compile_hierarchical_model,
    get_training_callbacks
)

# Causal intermittent / regressor demand-history features
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

# Residual transformer head
from .residual_transformer import (
    DEFAULT_SEQUENCE_CHANNELS,
    DEFAULT_CHANNEL_COLS,
    P_DS_CHANNEL_INDEX,
    build_residual_transformer,
    ResidualTrainModel,
    build_residual_windows,
    train_residual_transformer,
    predict_residual_transformer,
    mask_predict_step,
)

# Forecast post-process
from .forecast_postprocess import round_forecast

__version__ = "1.6.0"

def get_feature_config_path():
    """Path to packaged feature_config.yaml (v1.6)."""
    from pathlib import Path

    return Path(__file__).resolve().parent / "feature_config.yaml"

__all__ = [
    # Loss functions
    'composite_loss',
    'weighted_composite_loss',
    'mae_loss',
    'three_term_loss_config',
    'bce_mae_loss_config',
    'inverse_weighted_mae_loss',
    'hurdle_poisson_loss_config',
    'tweedie_loss_config',

    # Lightweight implementation
    'build_hierarchical_model_lightweight',

    # TabNet-based hierarchical attention
    'DeepSequencePWLHierarchical',
    'HierarchicalAttentionIntermittentHandler',
    'TrendComponentBuilder',
    'SeasonalComponentBuilder',
    'HolidayComponentBuilder',
    'RegressorComponentBuilder',

    # TabNet components
    'TabNetEncoder',
    'GhostBatchNormalization',
    'GLUBlock',
    'AttentiveTransformer',
    'FeatureTransformer',

    # Model utilities
    'create_hierarchical_model',
    'compile_hierarchical_model',
    'get_training_callbacks',

    # Causal intermittent features
    'SKUDemandState',
    'transform_panel',
    'build_states_from_history',
    'features_from_state',
    'update_state',
    'save_states',
    'load_states',
    'INTERMITTENT_FEATURE_NAMES',
    'CausalInferenceFeatureServer',

    # Residual transformer
    'DEFAULT_SEQUENCE_CHANNELS',
    'DEFAULT_CHANNEL_COLS',
    'P_DS_CHANNEL_INDEX',
    'build_residual_transformer',
    'ResidualTrainModel',
    'build_residual_windows',
    'train_residual_transformer',
    'predict_residual_transformer',
    'mask_predict_step',

    # Post-process
    'round_forecast',
    'get_feature_config_path',
    '__version__',
]
