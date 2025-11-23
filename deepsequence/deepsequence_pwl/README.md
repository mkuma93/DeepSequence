# DeepSequence PWL Implementation

Piecewise Linear (PWL) calibration-based components for time series forecasting.

## Components

### Core Components
- **trend_component.py** - Long-term trend with PWL calibration
- **seasonal_component.py** - Seasonal patterns with PWL calibration  
- **holiday_component.py** - Holiday/event effects with PWL
- **regressor_component.py** - External regressor handling

### Supporting Modules
- **intermittent_handler.py** - Zero-probability classification
- **model.py** - Complete model builder
- **config.py** - Configuration parameters
- **utils.py** - Utility functions

## Key Features

✅ PWL Calibration for flexible non-linear transformations
✅ Monotonicity constraints available
✅ Interpretable component outputs
✅ Optimized for intermittent demand

## Usage

```python
from deepsequence_pwl import build_deepsequence_pwl_model

model = build_deepsequence_pwl_model(
    num_features=10,
    embedding_dim=8,
    component_hidden_units=32,
    use_regressor=False  # 3-component optimized version
)
```

## Architecture

1. **Embedding Layer** - SKU representations
2. **Trend Component** - Dense(32, mish) → PWL(10 keypoints)
3. **Seasonal Component** - Dense(32, mish) → PWL(5) → Dense(32, mish) → PWL(5)
4. **Holiday Component** - Dense(32, mish) → PWL(5) → Dense(32, sigmoid)
5. **Transformer** - Cross-component integration (FFN: 96)
6. **Zero-Probability** - Binary classifier for intermittent demand

## Performance

Trained on 75K predictions:
- 48% improvement over LightGBM
- 90% SKU win rate
- 57% better on zero periods
- 44% better on non-zero periods
