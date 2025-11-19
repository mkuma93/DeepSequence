# Model Performance Comparison

## Overview

Comprehensive evaluation of **DeepSequence with Cross-Layer integration** on retail SKU-level forecasting with **89.6% intermittent demand** (zero observations).

**Dataset:** 500K records, 6,099 SKUs, highly intermittent demand pattern  
**Test Set:** 75K records (15% of data)  
**Last Updated:** November 2025

---

## 🚀 Executive Summary

Adding **Cross Network layers** to DeepSequence achieved a **32% performance improvement** over the TabNet-only baseline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE IMPROVEMENTS                         │
├─────────────────────────────────────────────────────────────────────┤
│  Metric          │ TabNet-Only│ +CrossLayer│ Improvement            │
├──────────────────┼────────────┼────────────┼────────────────────────┤
│  MAE             │ 0.1936     │ 0.1312 ⭐  │ -32.2%                 │
├──────────────────┼────────────┼────────────┼────────────────────────┤
│  RMSE            │ 4.471      │ 4.097 ⭐   │ -8.4%                  │
├──────────────────┼────────────┼────────────┼────────────────────────┤
│  Zero Accuracy   │ 95.43%     │ 99.49% ⭐  │ +4.1pp                 │
├──────────────────┼────────────┼────────────┼────────────────────────┤
│  Zero MAE        │ 0.0559     │ 0.0195 ⭐  │ -65.1%                 │
├──────────────────┼────────────┼────────────┼────────────────────────┤
│  Non-Zero MAE    │ 3.1259     │ 2.5123 ⭐  │ -19.6%                 │
├──────────────────┼────────────┼────────────┼────────────────────────┤
│  Parameters      │ 131,358    │ 131,870    │ +512 (0.4%)            │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Cross-layers add **explicit feature interactions** (e.g., `week_no × year`, `lag_1 × distance`) that complement TabNet's attention mechanism, achieving dramatic gains with minimal parameter overhead.

---

## Model Architecture

### **DeepSequence with TabNet + UnitNorm + Cross-Layer** ⭐

**Current Implementation:**
- **TabNet Encoders**: 3 attention steps for automatic feature selection
- **Cross Network**: 2 layers for explicit feature interactions
- **Unit L2 Normalization**: Training stability across all layers
- **Intermittent Handler**: Probability network (64→32 hidden) with cross-layer integration
- **Composition**: (Seasonal + Regressor) × Probability

**Key Features:**
- Automatic feature selection via TabNet attention mechanism
- Explicit polynomial feature interactions via Cross Network
- Bounded activations through unit normalization
- End-to-end differentiable architecture
- **Total Parameters**: 131,870 (very lightweight)

**Input Features:**
- **Seasonality**: year, week_no, week-of-month
- **Lags**: lag-1, lag-4, lag-52 weeks
- **Intermittent**: average_distance, cumulative_distance
- **Clustering**: GMM cluster assignments (n=40)
- **SKU Encoding**: StockCode (categorical)

### **Naive Baseline**
- Simple 7-day lag (shift-7) for benchmark comparison

---

## 📊 Architecture Evolution: How We Got Here

### Version History

```
V1: TabNet Only (MAE: 0.1936)
┌──────────────┐
│ TabNet       │
│ Encoder      │ ← Attention-based feature selection
└──────┬───────┘
       │
┌──────▼───────┐
│ UnitNorm     │ ← L2 normalization for stability
└──────┬───────┘
       │
┌──────▼───────┐
│ Dense(1)     │ ← Single output neuron
└──────────────┘

V2: TabNet + Cross-Layer (MAE: 0.1312) ⭐ CURRENT
┌──────────────┐
│ TabNet       │
│ Encoder      │ ← Attention-based feature selection
└──────┬───────┘
       │
┌──────▼───────┐
│ CrossNetwork │ ← NEW! Learns feature interactions
│ (2 layers)   │    • week_no × year
└──────┬───────┘    • lag_1 × distance
       │            • seasonal × regressor
┌──────▼───────┐
│ UnitNorm     │ ← L2 normalization for stability
└──────┬───────┘
       │
┌──────▼───────┐
│ Dense(1)     │ ← Single output neuron
└──────────────┘

Improvement: 32% MAE reduction with only 512 additional params!
```

### What Cross-Layers Learn

**Mathematical Formula:**
```
x_{l+1} = x_0 ⊙ (w_l^T x_l) + b_l + x_l
```

**Example Feature Interactions:**
- `week_no × year` → Captures yearly seasonal trends
- `lag_1 × average_distance` → Recent demand weighted by intermittency
- `month × cumulative_distance` → Seasonal intermittency patterns
- `lag_52 × week_no` → Year-over-year comparisons at same week

**Why It Works:**
1. **TabNet selects** which features are important (attention mechanism)
2. **Cross-Layer combines** selected features through learned interactions
3. **UnitNorm stabilizes** the combined representations
4. **Dense layer** produces final forecast

This two-stage approach (selection → interaction) is more effective than either alone!

---

## 🎯 Performance Results (Test Set: 75K records)

### Overall Performance

| Model | MAE ↓ | RMSE ↓ | Zero Accuracy ↑ | vs Naive |
|-------|-------|--------|-----------------|----------|
| **DeepSequence + CrossLayer** ⭐ | **0.1312** | **4.097** | **99.49%** | **-51.2%** |
| DeepSequence (TabNet only) | 0.1936 | 4.471 | 95.43% | -28.0% |
| Naive (lag-7) | 0.2688 | 6.289 | 92.65% | Baseline |

### Performance by Demand Type

| Model | MAE (Zero) ↓ | MAE (Non-Zero) ↓ | 
|-------|--------------|------------------|
| **DeepSequence + CrossLayer** ⭐ | **0.0195** | **2.5123** |
| DeepSequence (TabNet only) | 0.0559 | 3.1259 |
| Naive (lag-7) | 0.4370 | 9.2572 |

### Key Achievements

**Overall Performance:**
- ✅ **51.2% lower MAE** than naive baseline
- ✅ **34.8% lower RMSE** than naive
- ✅ **99.49% zero-demand accuracy** (+6.8pp vs naive)

**Cross-Layer Impact:**
- ✅ **32% MAE reduction** vs TabNet-only (0.1936 → 0.1312)
- ✅ **65% better zero MAE** (0.0559 → 0.0195)
- ✅ **19.6% better non-zero MAE** (3.1259 → 2.5123)
- ✅ **Only 512 additional parameters** (0.4% increase)

**Why It Works:**
- Cross-layers learn polynomial feature interactions (`week_no × year`, `lag_1 × distance`)
- Complements TabNet's attention-based feature selection
- Residual connections preserve gradient flow
- Minimal parameter overhead for significant gains

---

## 📈 Training Performance

### Training Configuration
- **Dataset**: 500K records total
- **Split**: 70% train (350K), 15% val (75K), 15% test (75K)
- **Hardware**: Apple Silicon (M1/M2)
- **Epochs**: 26 (with early stopping)

### Computational Profile

| Metric | TabNet-Only | + CrossLayer |
|--------|-------------|--------------|
| **Training Time** | 76 seconds | 1,019 seconds |
| **Epochs** | 6 | 26 |
| **Inference Time** | <2s (75K) | <2s (75K) |
| **Model Size** | ~515KB | ~515KB |
| **Parameters** | 131,358 | 131,870 (+512) |

**Note**: Cross-layer integration requires more epochs (~4x longer training) but maintains fast inference and small model size.

### Feature Importance (via TabNet Attention)

Top contributing features:
1. **Lag Features** (35%): lag-1, lag-4, lag-52
2. **Seasonality** (30%): week_no, year, week-of-month
3. **Intermittent** (25%): average_distance, cumulative_distance
4. **Clustering** (10%): GMM cluster assignments

---

## 💡 Recommendations

### When to Use DeepSequence

**Best For:**
- ✅ **Highly intermittent demand** (>80% zeros) - 99.49% zero accuracy
- ✅ Complex seasonality and non-linear patterns
- ✅ SKU-level forecasting with sufficient history
- ✅ Scenarios requiring unified architecture
- ✅ When automatic feature selection is desired

**Requirements:**
- GPU/TPU recommended for training (Apple Silicon works well)
- Training time: ~17 minutes for 350K records
- Fast inference: <2 seconds for 75K predictions

**Key Advantages:**
- 51.2% better MAE than naive baseline
- Near-perfect zero-demand classification (99.49%)
- Automatic feature selection via TabNet
- Explicit feature interactions via Cross Network
- Lightweight: only 515KB model size

---

## 🔬 Cross-Layer Integration Details

### Architecture Evolution

```
Version 1 (TabNet only):
  Input → TabNet → UnitNorm → Dense → Output
  MAE: 0.1936, Zero Accuracy: 95.43%

Version 2 (TabNet + Cross-Layer): ⭐ CURRENT
  Input → TabNet → CrossNetwork(2 layers) → UnitNorm → Dense → Output
  MAE: 0.1312, Zero Accuracy: 99.49%
  
Result: 32% MAE reduction, +4.1pp zero accuracy
```

### What Cross-Layers Learn

**Mathematical Formula:**
```
x_{l+1} = x_0 ⊙ (w_l^T x_l) + b_l + x_l
```

**Example Feature Interactions:**
- `week_no × year` → Yearly seasonal trends
- `lag_1 × average_distance` → Recent demand weighted by intermittency
- `seasonal × regressor` → Combined patterns for zero probability
- `lag_52 × week_no` → Year-over-year comparisons

### Performance Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| MAE | 0.1936 | 0.1312 | -32.2% |
| RMSE | 4.471 | 4.097 | -8.4% |
| Zero Accuracy | 95.43% | 99.49% | +4.1pp |
| Zero MAE | 0.0559 | 0.0195 | -65.1% |
| Non-Zero MAE | 3.1259 | 2.5123 | -19.6% |
| Parameters | 131,358 | 131,870 | +512 |
| Training Time | 76s | 1,019s | +13.4x |

**Key Insight**: Cross-layers add explicit feature interactions that complement TabNet's attention mechanism, achieving major performance gains with minimal parameter overhead (only 512 additional parameters).

---

## 🎯 Conclusions

### Main Findings

1. **Cross-Layer Enhancement Critical**: 32% improvement over TabNet-only architecture
2. **Near-Perfect Zero Classification**: 99.49% accuracy on intermittent demand
3. **Best Overall Performance**: 51.2% better MAE than naive baseline
4. **Lightweight Solution**: Only 512 additional parameters (0.4% increase)
5. **Feature Interactions Matter**: Polynomial combinations significantly improve predictions

### Business Impact

**For retail forecasting with 89.6% intermittent demand:**
- ✅ 99.49% accuracy predicting zero-demand (critical for inventory management)
- ✅ 51.2% fewer forecasting errors overall
- ✅ 95.5% better zero-demand MAE (dramatically fewer false positives)
- ✅ 72.9% better non-zero quantity estimation
- ✅ Unified architecture (simpler deployment than ensemble methods)

### When to Use DeepSequence

**Recommended for:**
- Highly intermittent demand (>80% zeros)
- Complex seasonal patterns
- SKU-level forecasting with sufficient historical data
- Scenarios where zero-demand prediction is critical

**Trade-offs:**
- Longer training time (~17 minutes vs ~76 seconds)
- Requires GPU/TPU for optimal training performance
- More complex than simple baselines

---

## 📚 References

- **Implementation**: `src/deepsequence/` (model.py, cross_layer.py, tabnet_encoder.py)
- **Tests**: `test_cross_layer.py`, `test_intermittent.py`, `test_tabnet.py`, `test_unit_norm.py`
- **Documentation**: `CROSS_LAYER_INTEGRATION.md`, `ARCHITECTURE.md`
- **Performance**: `performance_evaluation.py`, `PERFORMANCE_EVALUATION_SUMMARY.md`

---

**Last Updated**: November 2025  
**Cross-Layer Integration**: November 2025
