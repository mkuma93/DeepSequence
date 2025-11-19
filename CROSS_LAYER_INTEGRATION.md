# Cross Network Integration

## Overview

Added **Cross Network layers** (Deep & Cross Network - DCN) to DeepSequence architecture for explicit feature interaction learning. Cross layers learn polynomial feature interactions efficiently while keeping computational complexity linear.

**Implementation Date:** November 18, 2025

---

## Architecture Enhancement

### Cross Layer Formula

```
x_{l+1} = x_0 ⊙ (w_l^T x_l) + b_l + x_l
```

Where:
- `x_0`: Initial input (preserved across layers)
- `x_l`: Current layer input
- `w_l`: Learned weight vector
- `b_l`: Bias vector
- `⊙`: Element-wise multiplication

This creates **bounded polynomial** feature interactions of degree up to `l+1`.

### Integration Points

1. **Seasonal Path**: `TabNet → CrossNetwork(2 layers) → UnitNorm → Dense(1)`
2. **Regressor Path**: `TabNet → CrossNetwork(2 layers) → UnitNorm → Dense(1)`  
3. **Intermittent Handler**: `Concat → CrossNetwork(2 layers) → Dense layers`

---

## Model Architecture (With Cross Layers)

```
Seasonal Input (5 features)
    ↓
TabNetEncoder (32-dim output, 3 attention steps)
    ↓
CrossNetwork (2 layers, 128 params) ← NEW!
    ↓  
UnitNorm
    ↓
Dense(1) → Seasonal Forecast

Regressor Input (8 features)
    ↓
TabNetEncoder (32-dim output, 3 attention steps)
    ↓
CrossNetwork (2 layers, 128 params) ← NEW!
    ↓
UnitNorm
    ↓
Dense(1) → Regressor Forecast

Combined: Seasonal + Regressor
    ↓
Multiply by Probability → Final Output

Intermittent Handler:
  Concat(Seasonal TabNet, Regressor TabNet) [64-dim]
    ↓
  CrossNetwork (2 layers, 256 params) ← NEW!
    ↓
  Dense(64) → ReLU → UnitNorm → Dropout
    ↓
  Dense(32) → ReLU → UnitNorm → Dropout  
    ↓
  Dense(1, sigmoid) → Probability
```

**Total Cross Layer Parameters**: 512 (very lightweight!)
- seasonal_cross: 128 params (2 × 32 features × 2 layers)
- regressor_cross: 128 params (2 × 32 features × 2 layers)
- intermittent_cross: 256 params (2 × 64 features × 2 layers)

**Total Model Parameters**: 131,870 (vs 131,358 without cross layers)

---

## Motivation

### Why Cross Layers?

1. **TabNet Limitation**: TabNet learns feature **importance** via attention, but doesn't explicitly model feature **interactions**

2. **Feature Interactions Needed**:
   - `week_no × year` → Yearly seasonal patterns
   - `lag_1 × average_distance` → Recent demand × intermittent signal
   - `seasonal_features × regressor_features` → Combined patterns for zero probability

3. **LightGBM Advantage**: Tree models naturally learn interactions through splits
   - DeepSequence needed explicit interaction modeling to compete

4. **Cross Layer Benefits**:
   - ✅ Learns high-order interactions efficiently (polynomial up to depth l+1)
   - ✅ Minimal parameters (linear in feature dimension)
   - ✅ Residual connections prevent information loss
   - ✅ Works well with TabNet's attention-selected features

---

## Implementation Details

### Files Created/Modified

**New Files:**
- `src/deepsequence/cross_layer.py` - CrossLayer and CrossNetwork implementations
- `test_cross_layer.py` - Comprehensive test suite (7 tests, all passing)

**Modified Files:**
- `src/deepsequence/model.py` - Added CrossNetwork after TabNet encoders
- `src/deepsequence/__init__.py` - Export CrossLayer, CrossNetwork  
- `quick_performance_eval.py` - Updated to use cross layers

### Key Classes

#### `CrossLayer`
- Single cross layer with residual connection
- Parameters: `units`, `use_bias`, regularizers
- Input: Single tensor or tuple `(x_0, x_l)`
- Output: `x_{l+1}` with same shape as input

#### `CrossNetwork`  
- Stacks multiple cross layers
- Parameters: `num_layers`, `units`, `use_bias`
- Automatically handles `x_0` preservation across layers

### Usage Example

```python
from src.deepsequence import CrossNetwork

# After TabNet encoder
tabnet_output = tabnet_encoder(input)  # Shape: (batch, 32)

# Add cross network for interactions
cross_output = CrossNetwork(
    num_layers=2,
    use_bias=True,
    name='feature_cross'
)(tabnet_output)  # Shape: (batch, 32)

# Continue with unit norm and dense layers
output = UnitNorm()(cross_output)
forecast = Dense(1)(output)
```

---

## Testing

### Test Suite (`test_cross_layer.py`)

All 7 tests passing ✅:

1. **test_cross_layer_output_shape** - Verifies shape preservation
2. **test_cross_layer_with_separate_inputs** - Tests tuple input handling
3. **test_cross_layer_learns_interactions** - Confirms non-trivial transformations
4. **test_cross_network_multiple_layers** - Tests layer stacking
5. **test_cross_layer_gradients** - Verifies backpropagation
6. **test_cross_layer_residual_connection** - Tests identity mapping
7. **test_cross_network_integration** - Full Keras model integration

**Run tests:**
```bash
python test_cross_layer.py
```

---

## Expected Performance Improvements

### Hypothesis

Cross layers should improve **non-zero MAPE** (current: ~96%, target: closer to LGB 75-77%) by:

1. **Better Feature Interactions**:
   - Seasonal × Regressor interactions for demand patterns
   - Lag × Distance interactions for intermittent signals
   - Time × Cluster interactions for SKU-specific seasonality

2. **Improved Zero Probability**:
   - Cross layer on concat features learns seasonal-regressor interactions
   - Better separation of zero vs non-zero demand patterns

3. **Maintained Strengths**:
   - Zero accuracy should remain ~95.4% (already excellent)
   - Overall MAE should stay competitive or improve

### Baseline (TabNet + UnitNorm, NO Cross)

From `quick_performance_results.csv`:
- MAE: 0.1936 (28% better than naive)
- RMSE: 4.471
- Zero Accuracy: 95.43%

### With Cross Layers (Expected)

- MAE: 0.18-0.19 (similar or better)
- RMSE: 4.3-4.5 (similar or better)
- Zero Accuracy: ~95% (maintained)
- **Non-zero MAPE: 85-90%** (improved from ~96%, closer to LGB 75-77%)

---

## Comparison with LightGBM

| Aspect | LightGBM | DeepSequence (Baseline) | DeepSequence + Cross |
|--------|----------|------------------------|---------------------|
| **Feature Interactions** | Tree splits (automatic) | TabNet attention only | TabNet + Cross Network ✅ |
| **Non-zero MAPE** | 75-77% ✅ | ~96% | **Target: 85-90%** 🎯 |
| **Zero Accuracy** | Unknown | 95.43% ✅ | ~95% (maintained) ✅ |
| **Overall MAE** | Unknown | 0.1936 ✅ | ~0.18-0.19 (similar/better) |
| **Parameters** | N/A | 131K | 131.8K (+512 cross params) |
| **Architecture** | Ensemble | Unified | Unified ✅ |

**Key Insight**: Cross layers add the missing piece - explicit feature interactions that tree models get "for free" through splits.

---

## Future Enhancements

### Potential Improvements

1. **Deeper Cross Networks**:
   - Try 3-4 layers for higher-order interactions
   - Trade-off: more params vs better interactions

2. **Attention-Weighted Cross**:
   - Use TabNet attention masks to weight cross interactions
   - Focus on important feature pairs

3. **Cross Layer Variants**:
   - Matrix-based cross (DCN-v2) for more expressiveness
   - Mixture of cross experts for SKU-specific interactions

4. **Hyperparameter Tuning**:
   - Optimal number of cross layers per path
   - Cross layer placement (before/after unit norm)

---

## References

1. **Deep & Cross Network (DCN)**  
   Wang et al., "Deep & Cross Network for Ad Click Predictions", 2017
   - Original paper introducing cross layers
   - Proves effectiveness for feature interaction learning

2. **DCN-v2**  
   Wang et al., "DCN V2: Improved Deep & Cross Network", 2020
   - Matrix-based cross layers for more expressiveness

3. **TabNet**  
   Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning", 2019
   - Attention-based feature selection
   - Complementary to cross layers (importance + interactions)

---

## Summary

✅ **Implemented**: Cross Network layers at 3 integration points  
✅ **Tested**: All 7 test cases passing  
✅ **Integrated**: Updated model architecture and evaluation script  
✅ **Lightweight**: Only 512 additional parameters (+0.4%)  
✅ **Ready**: For full performance evaluation on complete dataset  

**Next Steps:**
1. Run full performance evaluation with cross layers
2. Compare non-zero MAPE with baseline and LightGBM
3. Fine-tune number of cross layers if needed
4. Update PERFORMANCE_COMPARISON.md with results

**Expected Impact**: Improved non-zero MAPE while maintaining excellent zero-demand prediction, closing the gap with LightGBM on intermittent retail forecasting.
