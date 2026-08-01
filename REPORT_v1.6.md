# DeepSequence Hierarchical Attention — Report v1.6

**Date:** 2026-07-28  
**Scope:** Feature contract v1.6 and same-feature bake-off only.  
**Artifact:** `eval_results_same_features_v16_distance_holidays.json` (aggregated metrics only)

---

## Dataset Availability

The experiments were conducted using proprietary enterprise demand data that cannot be publicly released due to confidentiality agreements. To support reproducibility, the repository includes:

* complete model implementation,
* preprocessing pipeline,
* synthetic example dataset,
* training configuration,
* evaluation methodology.

---

## 1. Feature contract (v1.6)

**28 columns.** Holiday **distance only** (binary `is_*` removed as redundant with `days_from_* == 0`). Same matrix for all models.

| Group | Features |
|-------|----------|
| Trend (1) | `time_index` |
| Seasonal (6) | `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `year_sin`, `year_cos` |
| Lags (3) | `lag_1`, `lag_2`, `lag_7` |
| Intermittent (3) | `days_since_last_sale`, `last_sale_quantity`, `lifetime_cumsum` |
| Holiday distance (15) | `days_from_*` calendar-distance features (generic public holidays) |

**Not included:** binary holiday indicators; `lag_14` (no gain in prior ablation).

**Sequence models (DeepAR / TST):** lookback windows of demand history + the same 28 features (lookback = 14).

---

## 2. Experiment setup

| Item | Value |
|------|--------|
| Panel | Proprietary enterprise demand (not released; see Dataset Availability) |
| Series count | 800 (seed 42) |
| Train zero rate | ≈ 0.899 |
| Volume strata | Train-volume terciles (low / mid / high) |
| DeepSequence | gated hierarchical (built-in loss) |
| LightGBM | L1 |
| DeepAR / TST / TFT | sequence models (same lookback windows + gated head) |
| Metrics | Intermittent suite (see §2.1): IWMAE, nonzero MAE, MASE, occurrence F1, underforecast rate, AUROC/AUCPR, bias |

No company names, product names, customer IDs, or series identifiers are published with these results.

### 2.1 Why not pure MAE?

With ≈90% zero days, all-day MAE heavily rewards near-zero forecasts and understates sale-day mistakes. Ranking primary metric is therefore **inverse-frequency weighted MAE (IWMAE)** on rounded forecasts; all-day MAE is retained only as a secondary level check.

| Metric | Role |
|--------|------|
| **IWMAE (rounded)** | Primary: weights sale days by \(1/\hat\pi\) and zero days by \(1/(1-\hat\pi)\) (\(\hat\pi\) = empirical nonzero rate) |
| **MAE (nonzero)** | Magnitude error on sale days only |
| **MASE (season-7)** | Scale-free vs train seasonal naive \(\lvert y_t - y_{t-7}\rvert\) |
| **Occurrence F1** | Timing: rounded \(\hat y>0\) vs \(y>0\) |
| **Underforecast rate (nz)** | Share of sale days with \(\hat y < y\) (stockout proxy) |
| **AUROC / AUCPR** | Quality of intermittent gate \(p\) |
| **Bias / bias_nonzero** | Systematic over/under forecast |

Tables below are from the recorded bake-off (`eval_results_same_features_v16_distance_holidays.json`). Primary sort is **IWMAE**; all-day MAE is secondary.

---

## 3. Results

### Overall (primary: IWMAE)

| Rank | Model | IWMAE | MAE (all) | MAE (nz) | MASE | Occ F1 | Under (nz) | AUROC | Bias |
|------|-------|------:|----------:|---------:|-----:|-------:|-----------:|------:|-----:|
| 1 | **DeepSequence** | **4.004** | **1.732** | 6.943 | **0.933** | **0.401** | 0.658 | 0.826 | +0.35 |
| 2 | TFT-lite | 4.026 | 1.798 | 6.910 | 0.968 | 0.394 | 0.646 | 0.825 | +0.42 |
| 3 | Temporal transformer (TST) | 4.065 | 1.882 | **6.892** | 1.014 | 0.392 | 0.619 | **0.835** | +0.55 |
| 4 | DeepAR-lite | 4.374 | 2.018 | 7.431 | 1.087 | 0.334 | **0.588** | 0.806 | +0.56 |
| 5 | LightGBM | 4.567 | 1.845 | 8.026 | 0.994 | 0.207 | 0.778 | 0.766 | +0.06 |

MASE scale (train season-7 naive): **1.856**. LightGBM’s competitive all-day MAE / bias collapses under IWMAE and occurrence F1—it rarely calls sale days and under-forecasts when they occur.

### By train-volume tercile (IWMAE)

| Band | Best | 2nd | 3rd | 4th | 5th |
|------|------|-----|-----|-----|-----|
| **Low** | TST 2.581 | TFT 2.635 | DeepAR 2.668 | DS 2.691 | LGBM 2.773 |
| **Mid** | **DS 3.399** | TST 3.407 | TFT 3.435 | DeepAR 3.524 | LGBM 3.750 |
| **High** | **DS 5.000** | TFT 5.061 | LGBM 5.074 | TST 5.182 | DeepAR 5.610 |

### By train-volume tercile (all-day MAE, secondary)

| Band | Best | 2nd | 3rd | 4th | 5th |
|------|------|-----|-----|-----|-----|
| **Low** | **DS 0.235** | DeepAR 0.250 | TFT 0.251 | TST 0.259 | LGBM 1.142 |
| **Mid** | **DS 0.843** | TFT 0.887 | TST 0.931 | DeepAR 1.046 | LGBM 1.422 |
| **High** | **LGBM 2.744** | DS 3.637 | TFT 3.758 | TST 3.936 | DeepAR 4.202 |

### Nonzero MAE (sale days)

| Band | Best → worst |
|------|----------------|
| Overall | TST 6.89 ≈ TFT 6.91 ≈ DS 6.94 → DeepAR 7.43 → LGBM 8.03 |
| High | TFT 7.30 ≈ TST 7.33 ≈ DS 7.34 → DeepAR 8.03 → LGBM **8.98** |

### Bias (neural models)

| Band | DS | TFT | TST | DeepAR |
|------|----|-----|-----|--------|
| Overall | **+0.35** | +0.42 | +0.55 | +0.56 |
| High | **+1.03** | +1.13 | +1.37 | +1.37 |

DS over-forecasts least among neural models. LightGBM under-forecasts on high volume (bias ≈ −0.90), which improves high all-day MAE but worsens high nonzero MAE / IWMAE.

---

## 4. Interpretation

- **Primary intermittent ranking (IWMAE):** DeepSequence → TFT → TST → DeepAR → LightGBM. LightGBM’s all-day MAE look is misleading.
- **Timing:** DS has the best occurrence F1 (0.40); LightGBM is weakest (0.21).
- **Sale-day size (nonzero MAE):** TST ≈ TFT ≈ DS; LightGBM worst.
- **High all-day MAE:** LightGBM “wins” by under-forecasting; on IWMAE / high-band IWMAE, DS is best.
- **Inventory / service level:** Prefer DS (or TFT). High underforecast rate + poor occ F1 on LightGBM risks stockouts.

---

## 5. Recommendation (v1.6)

**Ship DeepSequence** (hierarchical + gated intermittent head) on feature contract v1.6 as the default single model for **1-step / short-horizon** intermittent forecasting.

TFT-lite is the strongest neural runner-up on IWMAE. Use LightGBM only if a deliberately lean / under-forecast policy is required—it ranks last on IWMAE and occurrence F1. DeepAR is not competitive on this bake-off.

For **multi-horizon recursive rollout**, see §7: prefer IWMAE / nonzero / underforecast over all-day MAE when stockout cost dominates.

---

## 6. Package

- Python package: `deepsequence-hierarchical-attention` (see `pyproject.toml`)
- Feature SSOT: `feature_config.yaml` (v1.6), also shipped inside the package
- Synthetic example: `examples/v16_deepsequence_example.ipynb`
- Training configuration sample: `deepsequence_hierarchical_attention/training/training_config.sample.json`
- Evaluation harness (1-step): `python -m deepsequence_hierarchical_attention.eval.same_features_compare`
- Evaluation harness (multi-horizon): `python -m deepsequence_hierarchical_attention.eval.multihorizon_compare`
- Aggregated 1-step metrics: `eval_results_same_features_v16_distance_holidays.json`
- Aggregated multi-horizon metrics: `eval_results_multihorizon_v16.json`

---

## 7. Multi-horizon (recursive rollout)

**Protocol.** After observing day \(t\), forecast \(y_{t+1},\ldots,y_{t+H}\) (\(H=14\)). Calendar/holiday distances at \(t+h\) are known future; lags and intermittent state are updated with **predicted** demand. Same 1-step models as §3. Test origins: up to 8 per series (seed 42) → **6339** origins. Primary ranking: **IWMAE** (§2.1).

### By horizon (IWMAE)

| Horizon | Best IWMAE | 2nd | 3rd | 4th | 5th |
|--------:|------------|------|------|------|------|
| **h=1** | **DS 4.128** | TFT 4.190 | TST 4.251 | DeepAR 4.615 | LGBM 4.797 |
| **h=7** | **TST 4.133** | TFT 4.191 | DS 4.246 | LGBM 4.504 | DeepAR 4.679 |
| **h=14** | **TST 4.263** | DS 4.340 | TFT 4.396 | LGBM 4.637 | DeepAR 4.672 |
| **mean 1..14** | **TST 4.267** | TFT 4.319 | DS 4.352 | LGBM 4.672 | DeepAR 4.737 |

### By horizon (all-day MAE, secondary)

| Horizon | Best all-day MAE | 2nd | 3rd | 4th | 5th |
|--------:|------------------|------|------|------|------|
| **h=1** | **DS 1.778** | TFT 1.842 | TST 1.943 | LGBM 1.958 | DeepAR 2.049 |
| **h=7** | **LGBM 1.839** | TST 2.270 | TFT 2.295 | DS 2.487 | DeepAR 2.586 |
| **h=14** | **LGBM 1.915** | TST 2.413 | TFT 2.419 | DS 2.515 | DeepAR 2.589 |
| **mean 1..14** | **LGBM 1.910** | TST 2.266 | TFT 2.278 | DS 2.456 | DeepAR 2.530 |

### Occurrence F1 / underforecast (sale days)

| Horizon | Best occ F1 | Worst occ F1 | Highest under (nz) |
|--------:|-------------|--------------|--------------------|
| h=1 | **DS 0.425** | LGBM 0.220 | LGBM 0.777 |
| h=7 | TFT 0.338 | LGBM 0.211 | LGBM 0.803 |
| h=14 | TFT 0.337 | LGBM 0.220 | LGBM 0.813 |
| mean | TFT 0.353 | LGBM 0.217 | LGBM 0.793 |

### Nonzero MAE (sale days)

| Horizon | Best → worst |
|--------:|----------------|
| h=1 | DS 7.23 ≈ TFT 7.28 ≈ TST 7.29 → DeepAR 8.00 → LGBM 8.48 |
| h=7 | **DS 6.54** ≈ TST 6.56 ≈ TFT 6.67 → DeepAR 7.41 → LGBM 7.92 |
| h=14 | TST 6.72 ≈ DS 6.76 ≈ TFT 7.00 → DeepAR 7.43 → LGBM 8.18 |
| mean | DS 6.85 ≈ TST 6.91 ≈ TFT 7.00 → DeepAR 7.65 → LGBM 8.25 |

### Bias (mean over horizons)

| Model | Bias |
|-------|-----:|
| LightGBM | **−0.04** |
| TST | +0.84 |
| TFT | +0.85 |
| DeepAR | +0.89 |
| DeepSequence | +1.04 |

**Reading.** Under **IWMAE**, LightGBM never ranks first at any horizon despite winning long-horizon all-day MAE—it has the worst occurrence F1 and highest underforecast rate. DeepSequence wins **h=1** on IWMAE; TST/TFT lead longer-horizon IWMAE. Prefer DS/TFT/TST for inventory; treat LGBM’s MAE lead as a near-zero-forecast artifact.

---

## 8. Direct multi-horizon DeepSequence (improvement experiment)

**Motivation.** DS was trained 1-step only; long-horizon IWMAE slips from bias accumulation under recursive rollout. A direct Dense(H) gated head — same hierarchical backbone, same loss, just `[B, H]` outputs — removes compounding error and trains jointly on all H steps.

**Variants tested (H=14, 800 SKUs, seed 42).**

| Variant | Description |
|---------|-------------|
| `ds_h1_recursive` | Baseline: classic 1-step DS, evaluated recursively |
| `ds_mh_direct` | Direct Dense(14) head, horizon_decay=0.95 |
| `ds_mh_tuned_direct` | Same head, stronger BCE (α=0.35) + magnitude (w=1.25) + decay=0.95 |

**Results (IWMAE, primary).**

| Horizon | ds_h1_recursive | ds_mh_direct | ds_mh_tuned_direct |
|--------:|----------------:|-------------:|-------------------:|
| **h=1** | **4.132** | 4.341 | 4.386 |
| **h=7** | 4.254 | 4.113 | **4.082** |
| **h=14** | 4.370 | 4.256 | **4.223** |
| **mean** | 4.362 | 4.247 | **4.234** |

Bias accumulation under recursion: h1 reaches +1.27 at h=7 / +1.20 at h=14. Direct MH stays ≤+0.54 (with 0.78× bias calibration on val origins). At h=7/14, `ds_mh_tuned` also beats TST (4.133/4.263) and TFT (4.191/4.396) from §7.

**Takeaway.** Direct multi-horizon DS closes ~0.13–0.15 IWMAE at long horizons vs the 1-step recursive baseline. Trade-off: h=1 IWMAE is weaker (4.39 vs 4.13) since the head optimises H steps jointly.

**Recommendation.**
- **h=1 / next-day default:** `ds_h1` (best h=1 IWMAE overall).
- **Multi-day planning (H≥7):** `ds_mh_tuned` (best long-horizon IWMAE, outperforms TST/TFT/LGBM).
- Artifacts: `eval_results_ds_mh_improve_v16.json` · harness: `python -m deepsequence_hierarchical_attention.eval.ds_multihorizon_improve`.
