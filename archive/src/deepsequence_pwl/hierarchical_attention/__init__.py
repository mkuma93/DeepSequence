"""
Hierarchical Attention Architecture for Intermittent Demand Forecasting.

This module implements a sophisticated three-level attention mechanism:

Level 1 (PWL-level): Attention on piecewise linear calibration outputs
    - Trend: Attention on changepoint features (which time regions matter)
    - Holiday: Attention on distance range features (which distances from holidays matter)

Level 2 (Feature-level): Attention on hidden features within each component
    - Applied to all components (trend, seasonal, holiday, regressor)
    - Selects which hidden dimensions are important per SKU

Level 3 (Component-level): Attention across all components
    - Learns importance of trend vs seasonal vs holiday vs regressor
    - Different SKUs can have different component importance

Key Features:
- TensorFlow-native sparsemax activation (can output exact zeros)
- PWL calibration for non-linear transformations
- SKU-specific attention through embeddings
- Hierarchical sparsity for interpretability
"""

from .components import (
    SparseAttention,
    CrossLayer,
    TrendComponentBuilder,
    SeasonalComponentBuilder,
    HolidayComponentBuilder,
    RegressorComponentBuilder,
    TrendComponentBuilderSimple,
    SeasonalComponentBuilderTabNet,
    HolidayComponentBuilderTabNet,
    RegressorComponentBuilderTabNet,
    HierarchicalAttentionIntermittentHandler,
    DeepSequencePWLHierarchical
)

from .tabnet import (
    TabNetEncoder,
    GhostBatchNormalization,
    GLUBlock,
    AttentiveTransformer,
    FeatureTransformer
)

from .autoregressive import (
    create_lag_features,
    AutoregressivePredictor,
    prepare_historical_demand_buffer
)

from .model import (
    create_hierarchical_model,
    compile_hierarchical_model,
    get_training_callbacks
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

__all__ = [
    'SparseAttention',
    'CrossLayer',
    'TrendComponentBuilder',
    'SeasonalComponentBuilder',
    'HolidayComponentBuilder',
    'RegressorComponentBuilder',
    'TrendComponentBuilderSimple',
    'SeasonalComponentBuilderTabNet',
    'HolidayComponentBuilderTabNet',
    'RegressorComponentBuilderTabNet',
    'HierarchicalAttentionIntermittentHandler',
    'DeepSequencePWLHierarchical',
    'TabNetEncoder',
    'GhostBatchNormalization',
    'GLUBlock',
    'AttentiveTransformer',
    'FeatureTransformer',
    'create_hierarchical_model',
    'compile_hierarchical_model',
    'get_training_callbacks',
    'create_lag_features',
    'AutoregressivePredictor',
    'prepare_historical_demand_buffer',
    'SKUDemandState',
    'transform_panel',
    'build_states_from_history',
    'features_from_state',
    'update_state',
    'save_states',
    'load_states',
    'INTERMITTENT_FEATURE_NAMES',
    'CausalInferenceFeatureServer',
]
