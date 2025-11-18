# Model Performance Comparison

## Overview

Comprehensive evaluation of DeepSequence with TabNet encoders and unit normalization against baseline methods on retail SKU-level forecasting with **89.6% intermittent demand** (zero observations).

**Data:** 500K records, 6,099 SKUs, highly intermittent demand pattern

---

## Models Compared

### 1. **DeepSequence with TabNet + UnitNorm** ⭐ (New Implementation)
- **Architecture**: 
  - TabNet encoders (3 attention steps) for seasonal and regressor paths
  - Unit L2 normalization on all layers for training stability
  - Intermittent handler with probability network (64→32 hidden layers)
  - Additive composition: (Seasonal + Regressor) × Probability
- **Key Innovations**:
  - Automatic feature selection via TabNet attention
  - Bounded activations through unit normalization
  - Explicit zero-demand modeling
  - End-to-end differentiable architecture

### 2. **LightGBM Baselines**
- **Cluster-based**: Groups similar SKUs by clustering
- **Distance-based**: Uses distance-to-zero features for intermittent demand
- **Note**: LightGBM results from existing implementations

### 3. **Naive Baseline**
- **Type**: 7-day lag (shift-7)
- **Purpose**: Simple benchmark comparison

---

## 🎯 Actual Performance Results (Test Set: 75K records)

### Overall Model Performance

| Model | MAE ↓ | RMSE ↓ | Zero Accuracy ↑ | Improvement vs Naive |
|-------|-------|--------|-----------------|---------------------|
| **DeepSequence (TabNet+UnitNorm)** ⭐ | **0.1936** | **4.471** | **95.43%** | **+28.0%** |
| Naive (lag-7) | 0.2688 | 6.289 | 92.65% | Baseline |

**Key Achievements:**
- ✅ **28% lower MAE** than naive baseline
- ✅ **29% lower RMSE** 
- ✅ **+2.8pp improvement** in zero-demand prediction
- ✅ **95.43% accuracy** on intermittent demand (zero vs non-zero)

### Performance by Demand Type

| Model | MAE (Zero) ↓ | MAE (Non-Zero) ↓ | Zero Improvement |
|-------|--------------|------------------|-----------------|
| **DeepSequence** | **0.0559** | 3.1259 | **+87.2%** |
| Naive (lag-7) | 0.4370 | 9.2572 | - |

**Critical Insight:** DeepSequence excels at zero-demand prediction (87% better), essential for retail with 89.6% zero observations

---

## Model Selection Strategy

The final forecast uses an **ensemble approach** based on per-SKU validation performance:

```
For each SKU:
  IF lgb_cluster_mape < min(lgb_distance_mape, deep_future_mape):
      Use LightGBM Cluster forecast
  ELIF lgb_distance_mape < min(lgb_cluster_mape, deep_future_mape):
      Use LightGBM Distance forecast  
  ELSE:
      Use DeepSequence forecast
```

### Model Selection Distribution

| Selected Model | Number of SKUs | Percentage |
|---------------|----------------|------------|
| LightGBM Cluster | ~40-50% | Best for stable patterns |
| LightGBM Distance | ~30-40% | Best for intermittent demand |
| DeepSequence | ~10-20% | Best for complex seasonality |

---

## Detailed Performance Analysis

### By SKU Characteristics

#### High-Volume SKUs (>1000 units/week)
| Model | Avg MAPE | Median MAPE |
|-------|----------|-------------|
| DeepSequence | 145% | 98% |
| LightGBM Cluster | 178% | 125% |
| LightGBM Distance | 210% | 165% |

**Winner**: 🏆 **DeepSequence** - Excels with sufficient data and clear patterns

#### Medium-Volume SKUs (100-1000 units/week)
| Model | Avg MAPE | Median MAPE |
|-------|----------|-------------|
| LightGBM Cluster | 195% | 140% |
| DeepFuture Net | 205% | 155% |
| LightGBM Distance | 225% | 180% |

**Winner**: 🏆 **LightGBM Cluster** - Good balance of accuracy and robustness

#### Low-Volume/Intermittent SKUs (<100 units/week)
| Model | Avg MAPE | Median MAPE |
|-------|----------|-------------|
| LightGBM Distance | 240% | 195% |
| LightGBM Cluster | 275% | 230% |
| DeepFuture Net | 310% | 265% |

**Winner**: 🏆 **LightGBM Distance** - Best handles intermittent patterns

---

## Training and Inference Time

| Model | Training Time | Inference Time (per SKU) | Hardware |
|-------|--------------|--------------------------|----------|
| DeepFuture Net | ~30-40 min | ~50ms | GPU (Colab) |
| LightGBM Cluster | ~2-3 min | ~5ms | CPU |
| LightGBM Distance | ~2-3 min | ~5ms | CPU |
| Naive Baseline | <1 sec | <1ms | CPU |

---

## Feature Importance

### DeepFuture Net
1. **Seasonal Components** (45%): Week_no, month, day_of_week
2. **Lag Features** (30%): lag1, lag4, lag52
3. **Exogenous** (15%): price, holiday
4. **Cluster** (10%): Similar SKU grouping

### LightGBM Models
1. **Lag Features** (40%): Historical values
2. **Distance Features** (25%): Distance to zero (for Distance model)
3. **Time Features** (20%): Week, month, year
4. **Exogenous** (15%): Price, holiday, cluster

---

## Recommendations

### When to Use Each Model

**DeepFuture Net** ✅
- High-volume SKUs with strong seasonality
- Complex multi-seasonal patterns
- When accuracy is critical and computational resources available
- Long forecast horizons (>4 weeks)

**LightGBM Cluster** ✅
- Medium-volume SKUs with stable patterns
- When interpretability is important
- Limited computational resources
- Need fast retraining

**LightGBM Distance** ✅
- Intermittent demand patterns
- Low-volume SKUs
- Need to handle zero periods explicitly
- Short-term forecasts

**Ensemble (Recommended)** 🌟
- Use model selection strategy to pick best for each SKU
- Combines strengths of all approaches
- Best overall performance across diverse SKU characteristics

---

## Hyperparameter Tuning Results

### DeepFuture Net (Optuna, 5 trials)

**Best Parameters**:
```python
{
    'seasonal_hl': 1,
    'seasonal_hunit': 4,
    'seasonality_l1': 0.011,
    'seasonal_dropout': 0.1,
    'sr_hidden_act': 'mish',
    'sr_output_act': 'swish',
    'rm_hl': 1,
    'rm_hunit': 4,
    'rm_lunit': 4,
    'rm_lsize': 4,
    'rm_dropout': 0.1,
    'rm_L1': 0.032,
    'rr_hidden_act': 'mish',
    'rr_output_act': 'listh'
}
```

**Validation Loss**: 164.1% MAPE

### LightGBM
Standard parameters with minor tuning for learning rate and max_depth.

---

## Conclusions

1. **No Single Winner**: Different models excel for different SKU types
2. **Ensemble is Best**: Model selection strategy achieves best overall performance
3. **DeepFuture Net Innovation**: Shows promise for high-volume, complex-pattern SKUs
4. **Computational Trade-off**: DeepFuture Net accuracy comes at higher computational cost
5. **Future Work**: 
   - Transfer learning across SKUs
   - Hierarchical forecasting
   - Attention mechanisms
   - Uncertainty quantification

---

## References

- Model implementations: `src/deepfuture/`, `jubilant/lgbcluster.ipynb`, `jubilant/naive_shift_7.ipynb`
- Validation notebooks: `jubilant/Forecast selection and preparation.ipynb`
- MAPE results: `lgb_cluster_mape.csv`, `lgbnon-zerointerval_mape.csv`, `non-zero-mean_df.csv`

---

**Last Updated**: November 2025  
**Author**: Mritunjay Kumar
