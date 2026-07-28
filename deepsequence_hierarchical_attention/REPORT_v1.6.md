# DeepSequence Hierarchical Attention — Report v1.6

**Date:** 2026-07-28  
**Scope:** Feature contract v1.6 and same-feature bake-off only.  
**Artifact:** `eval_results_same_features_v16_distance_holidays.json`

---

## 1. Feature contract (v1.6)

**28 columns.** Holiday **distance only** (binary `is_*` removed as redundant with `days_from_* == 0`). Same matrix for all models.

| Group | Features |
|-------|----------|
| Trend (1) | `time_index` |
| Seasonal (6) | `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `year_sin`, `year_cos` |
| Lags (3) | `lag_1`, `lag_2`, `lag_7` |
| Intermittent (3) | `days_since_last_sale`, `last_sale_quantity`, `lifetime_cumsum` |
| Holiday distance (15) | `days_from_NewYear` … `days_from_NewYearEve` |

**Not included:** binary holiday indicators; `lag_14` (no gain in prior ablation).

**Sequence models (DeepAR / TST):** lookback windows of `[Quantity] + same 28 features` (lookback = 14).

---

## 2. Experiment setup

| Item | Value |
|------|--------|
| Panel | Jubilant intermittent demand |
| SKUs | 800 (seed 42) |
| Train zero rate | ≈ 0.899 |
| Volume strata | Train-volume terciles (low / mid / high) |
| DS loss | three_term (gated) |
| LightGBM | L1 |
| DeepAR / TST | three_term |
| Metrics | Rounded all-day MAE, nonzero MAE, AUROC/AUCPR, bias |

---

## 3. Results

### Overall

| Rank | Model | MAE (all) | MAE (nonzero) | AUROC | Bias |
|------|-------|-----------|---------------|-------|------|
| 1 | **DeepSequence three_term** | **1.732** | 6.943 | 0.826 | +0.35 |
| 2 | LightGBM | 1.845 | 8.026 | 0.766 | +0.06 |
| 3 | Temporal transformer (TST) | 1.882 | **6.892** | **0.835** | +0.55 |
| 4 | DeepAR-lite | 2.018 | 7.431 | 0.806 | +0.56 |

### By train-volume tercile (all-day MAE)

| Band | Best | 2nd | 3rd | 4th |
|------|------|-----|-----|-----|
| **Low** | **DS 0.235** | DeepAR 0.250 | TST 0.259 | LGBM 1.142 |
| **Mid** | **DS 0.843** | TST 0.931 | DeepAR 1.046 | LGBM 1.422 |
| **High** | **LGBM 2.744** | DS 3.637 | TST 3.936 | DeepAR 4.202 |

### Nonzero MAE (sale days)

| Band | Best → worst |
|------|----------------|
| Overall | TST 6.89 ≈ DS 6.94 → DeepAR 7.43 → LGBM 8.03 |
| High | TST 7.33 ≈ DS 7.34 → DeepAR 8.03 → LGBM **8.98** |

### Bias (neural models)

| Band | DS | TST | DeepAR |
|------|----|-----|--------|
| Overall | **+0.35** | +0.55 | +0.56 |
| High | **+1.03** | +1.37 | +1.37 |

DS over-forecasts less than TST/DeepAR. LightGBM under-forecasts on high volume (bias ≈ −0.90), which improves high all-day MAE but worsens high nonzero MAE.

---

## 4. Interpretation

- **Overall + low + mid:** DeepSequence is best under a fair same-feature contract.
- **Sale-day size (nonzero MAE):** TST slightly ahead; DS essentially tied.
- **High all-day MAE:** LightGBM “wins” by under-forecasting quiet days, not by sizing demand better when sales occur.
- **Inventory / service level:** Prefer DS (or TST). Under-forecasting (LightGBM on high) risks stockouts; DS covers low/mid well and does not under-forecast relative to TST/DeepAR.

---

## 5. Recommendation (v1.6)

**Ship DeepSequence hierarchical + three_term + gated intermittent head** on feature contract v1.6 as the default single model.

Use LightGBM only if a deliberately lean / under-forecast policy is required. DeepAR is not competitive on this bake-off.

---

## 6. Package

- Python package: `deepsequence-hierarchical-attention` (see `pyproject.toml`)
- Feature SSOT: `feature_config.yaml` (v1.6), also shipped inside the package
- Example: `examples/v16_deepsequence_example.ipynb`
- Full metrics JSON: `eval_results_same_features_v16_distance_holidays.json`
