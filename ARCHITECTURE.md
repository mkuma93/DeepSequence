# DeepSequence Architecture

## Overview

DeepSequence is a custom deep learning architecture designed for multi-horizon time series forecasting at the SKU level. Inspired by Facebook's Prophet, it decomposes the forecasting problem into seasonal and trend components while leveraging deep learning for complex pattern recognition.

## Design Philosophy

The architecture is built on three key principles:

1. **Seasonal Decomposition**: Explicitly model multiple seasonality patterns (weekly, monthly, yearly)
2. **Recurrent Learning**: Capture temporal dependencies through LSTM/GRU layers
3. **Contextual Awareness**: Incorporate exogenous variables and categorical embeddings

## Architecture Components

### 1. Seasonal Module

Captures multiple seasonality patterns similar to Prophet's Fourier series approach:

```
Input: Time features (day_of_week, month, quarter, etc.)
       ↓
Embedding Layers (for categorical time features)
       ↓
Dense Hidden Layers (configurable)
       ↓
Seasonal Components (weekly, monthly, yearly)
```

**Features**:
- Separate sub-networks for each seasonality type
- Configurable hidden layers and units
- L1 regularization for feature selection
- Dropout for regularization

### 2. Regression Module

Handles trend, exogenous variables, and contextual features:

```
Input: Historical values + Exogenous variables + Cluster info
       ↓
Embedding Layers (for StockCode, clusters)
       ↓
LSTM/GRU Layers (temporal dependencies)
       ↓
Dense Layers (feature transformation)
       ↓
Lattice Layers (constraint handling)
       ↓
Regression Output
```

**Features**:
- Recurrent layers for sequence modeling
- Multiple hidden layers with configurable activation
- Lattice layers for monotonicity constraints
- Support for business rules and constraints

### 3. Fusion Layer

Combines seasonal and regression components:

```
Seasonal Output + Regression Output → Final Forecast
```

## Model Parameters

### Seasonal Component Configuration
- `seasonal_hl`: Number of hidden layers
- `seasonal_hunit`: Units per hidden layer
- `seasonality_l1`: L1 regularization strength
- `seasonal_dropout`: Dropout rate
- `sr_hidden_act`: Hidden layer activation (e.g., 'relu', 'swish')
- `sr_output_act`: Output activation

### Regression Component Configuration
- `rm_lat_unit`: Lattice units
- `rm_lattice_size`: Lattice size (for constraints)
- `rm_hidden_unit`: Hidden layer units
- `rm_hidden_layer`: Number of hidden layers
- `rm_drop_out`: Dropout rate
- `rm_L1`: L1 regularization
- `rr_hidden_act`: Hidden activation
- `rr_output_act`: Output activation

## Training Strategy

### Data Preparation
1. Time series split for validation
2. Feature engineering (lags, rolling statistics, time features)
3. Clustering for similar SKU groups
4. Normalization and scaling

### Optimization
- **Loss Function**: Mean Absolute Percentage Error (MAPE) or custom loss
- **Optimizer**: Adam with configurable learning rate
- **Batch Size**: 512 (default, configurable)
- **Early Stopping**: Monitors validation loss
- **Model Checkpointing**: Saves best performing model

### Validation
- Walk-forward validation for time series
- 8-week horizon forecasting
- Performance metrics: MAPE, MAE, RMSE

## Advantages over Traditional Approaches

1. **Multi-Seasonality**: Handles weekly, monthly, and yearly patterns simultaneously
2. **Non-linear Patterns**: Deep learning captures complex relationships
3. **Contextual Information**: Integrates price, holidays, and cluster information
4. **Scalability**: Processes multiple SKUs efficiently
5. **Constraint Handling**: Business rules through lattice layers

## Comparison with Prophet

| Aspect | Prophet | DeepSequence |
|--------|---------|--------------|
| Seasonality | Fourier series | Neural networks |
| Trend | Piecewise linear | Recurrent layers |
| Exogenous | Limited | Full integration |
| Non-linearity | Limited | High capacity |
| Training | Fast | Slower (GPU recommended) |
| Interpretability | High | Moderate |
| Accuracy (complex patterns) | Good | Better |

## Implementation Details

### Dependencies
- TensorFlow >= 2.6.0
- TensorFlow Lattice (for constraint layers)
- NumPy, Pandas for data handling

### Custom Modules
The implementation uses custom classes from `src/deepsequence/`:
- `SeasonalComponent`: Seasonal component builder
- `RegressorComponent`: Regression component builder

## Future Enhancements

Potential improvements:
- Attention mechanisms for time series
- Transfer learning across similar SKUs
- Hierarchical forecasting capabilities
- Uncertainty quantification
- AutoML for hyperparameter tuning

## Performance

On retail SKU forecasting datasets:
- Competitive MAPE against LightGBM models
- Better performance on SKUs with strong seasonal patterns
- Model selection based on per-SKU validation performance

## Usage Example

```python
# Initialize seasonal module
sm = SC.seasonal(data, target='Quantity', id_var='StockCode', 
                horizon=8, weekly=True, yearly=True, monthly=True)
sm.seasonal_feature()
sm.seasonal_model(hidden=2, hidden_unit=64, hidden_act='swish')

# Initialize regression module
rm = RC.Regressor(ts=train_data, exog=exog_data, target='Quantity',
                 id_var='StockCode', categorical_var=['cluster'])
rm.feature_engineering(lag_features=True)
rm.build_model(latent_units=32, hidden_layers=2)

# Train combined model
history = model.fit(train_x, train_y, 
                   validation_data=(val_x, val_y),
                   epochs=500, batch_size=512,
                   callbacks=[early_stopping, model_checkpoint])

# Generate forecasts
forecasts = model.predict(future_x)
```

## References

- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Gupta, M., et al. (2016). Monotonic calibrated interpolated look-up tables. JMLR, 17(1), 3790-3836.

---

**Author**: Mritunjay Kumar  
**Year**: 2021  
**License**: [Add your license]
