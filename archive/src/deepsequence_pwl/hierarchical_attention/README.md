# Hierarchical Attention Architecture

This directory contains the hierarchical attention architecture for intermittent demand forecasting.

## Architecture Overview

The hierarchical attention mechanism operates at three levels with Entmax15 sparse attention applied to each component:

### Component-Specific Attention (Applied to Each Component)

Each component has its own Entmax15 attention mechanism:

- **Trend Component**: 
  - Extracts time feature → PWL calibration (if enabled) → Dense → **Entmax15 Attention on PWL output** → Dropout
  - Learns which calibrated time features matter for each SKU
  - Example: "SKU uses early growth changepoints, ignores recent trends"
  
- **Seasonal Component**: 
  - Extracts seasonal features → Dense → **Entmax15 Attention on hidden features** → Dropout
  - Learns which seasonal patterns matter (Fourier features, categorical date features)
  - Example: "SKU uses weekly patterns, ignores monthly/quarterly cycles"

- **Holiday Component**: 
  - Extracts 15 holiday distance features → **15 individual PWL calibrations** (one per holiday) → Concatenate → Dense → **Entmax15 Attention on PWL output** → Dropout
  - Learns which holidays and which distance ranges matter for each SKU
  - Example: "SKU sensitive only to Christmas & Thanksgiving, 7-14 days before"
  
- **Regressor Component**: 
  - Extracts regressor features → Dense → **Entmax15 Attention on hidden features** → Dropout
  - Learns which external regressors are important for each SKU
  - Example: "SKU responds to promotions only, ignores price changes"

### Level 1: Feature-Level Attention (Within Intermittent Handler)
Applied to component outputs within the zero probability network:
- Attention on hidden features within each component's output
- Learns which features within each component contribute to zero probability prediction

### Level 2: Component-Level Attention (Within Intermittent Handler)
- Attention across all 4 components (trend/seasonal/holiday/regressor)
- Learns the importance of each component for intermittent demand prediction
- Example: "For zero prediction: Trend=60%, Seasonal=30%, Holiday=10%, Regressor=0%"

## Key Features

- **Two-Level Hierarchy**: Feature-level (which features within component) + Component-level (which components)
- **Sparse Attention**: Uses sparsemax activation (TensorFlow-native) that can output exact zeros
- **PWL Calibration**: Piecewise linear calibration for trend and holiday components
- **SKU-Specific**: All attention is conditioned on SKU embeddings
- **Interpretable**: Can analyze attention weights at both levels to understand model decisions
- **Efficient**: Lightweight architecture with minimal parameters

## Files

- `components.py`: Component builders with hierarchical attention
  - `Entmax15`: TensorFlow sparsemax implementation
  - `TrendComponentBuilder`: Trend with PWL and attention
  - `SeasonalComponentBuilder`: Seasonal patterns
  - `HolidayComponentBuilder`: Holiday effects with PWL and attention
  - `RegressorComponentBuilder`: Additional regressors
  - `HierarchicalAttentionIntermittentHandler`: Multi-level attention for zero probability
  - `DeepSequencePWLHierarchical`: Main model class

- `model.py`: High-level model creation and training utilities
  - `create_hierarchical_model()`: Model factory function
  - `compile_hierarchical_model()`: Compilation with appropriate losses
  - `get_training_callbacks()`: Standard callbacks for training

- `__init__.py`: Package exports

## Usage Example

```python
from deepsequence_pwl.hierarchical_attention import create_hierarchical_model, compile_hierarchical_model

# Create model
main_model, trend_model, seasonal_model, holiday_model, regressor_model = create_hierarchical_model(
    num_skus=6099,
    n_features=10,
    component_hidden_units=32,
    trend_feature_indices=[0],
    seasonal_feature_indices=[2, 3, 4, 5],
    holiday_feature_index=1,
    regressor_feature_indices=[6, 7, 8, 9],
    time_min=0.0,
    time_max=365.0
)

# Compile
model = compile_hierarchical_model(main_model, learning_rate=0.005)

# Train
model.fit(
    [X_train, sku_ids_train],
    {
        'base_forecast': y_train,
        'final_forecast': y_train,
        'zero_probability': is_zero_train
    },
    validation_data=(...),
    epochs=100,
    callbacks=get_training_callbacks('best_model.h5')
)
```

## Benefits for Intermittent Demand

1. **Sparsity**: Can completely ignore irrelevant features/components
2. **Adaptability**: Different SKUs use different patterns
3. **Interpretability**: Analyze attention weights to understand predictions
4. **Efficiency**: Hierarchical structure reduces parameters while increasing capacity
5. **Robustness**: Multiple levels of attention provide redundancy

## Architecture Visualization

![Hierarchical Attention Architecture](../../../hierarchical_attention_architecture.png)

The diagram above shows the complete hierarchical attention architecture:

1. **Input Layer**: Main features (30) and SKU embeddings (8 dimensions)
   - 15 holiday distance features (days_from_NewYear, days_from_MLK, etc.)
   - 10 Fourier seasonal features (sin/cos pairs)
   - 4 categorical date features (day_of_week, month, etc.)
   - 1 time feature

2. **Component Branches**: Four parallel branches with component-specific Entmax15 attention
   - **Trend**: PWL calibration on time → Dense → Entmax15 attention
   - **Seasonal**: Dense on seasonal features → Entmax15 attention
   - **Holiday**: 15 individual PWL calibrations (one per holiday) → Concatenate → Dense → Entmax15 attention
   - **Regressor**: Dense on regressor features → Entmax15 attention (if available)
   - All components use Dense layers with Mish activation and Dropout
   - Shift-and-scale applied using SKU embeddings for personalization

3. **Intermittent Handler** (if enabled):
   - **Feature-Level Attention**: Attention within each component's output (4 layers)
   - **Component-Level Attention**: Attention across all 4 components (1 layer)
   - **Zero Probability Network**: Multi-layer Dense network for intermittent handling

4. **Outputs**: Base forecast, final forecast, and zero probability (if intermittent handling enabled)

**Key Metrics**:
- Total Parameters: 109,323
- Attention Layers: 9 total (4 component-specific + 4 feature-level + 1 component-level)
- PWL Calibrations: 15 individual calibrators for holiday distance features
- Sparse Attention: Entmax15 (sparsemax) for exact zeros at all levels
- Attention Layers: 5 (4 feature-level + 1 component-level)
- Sparse Attention: Entmax15 (sparsemax) for exact zeros

To regenerate the architecture diagram:
```bash
python visualize_hierarchical_architecture.py
```

## Testing

Run `test_hierarchical_attention.py` in the project root to verify the architecture:

```bash
python test_hierarchical_attention.py
```

This tests:
- Model building
- Forward pass
- Training step
- Attention layer verification
- PWL calibration integration
