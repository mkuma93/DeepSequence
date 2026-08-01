# DeepSequence Hierarchical Attention

Prophet-inspired **multi-series intermittent demand** forecasting: hierarchical attention over trend / seasonal / holiday / regressor experts, a **context-aware** Level-2 mixer, and an occurrence × magnitude gate \(\hat{y}=p\cdot b\).

**Package version:** 1.6.0 · **Feature contract:** v1.6 (28 columns)  
**Paper:** [PAPER.md](PAPER.md) · **Report:** [REPORT_v1.6.md](REPORT_v1.6.md) · **Example notebook:** [examples/v16_deepsequence_example.ipynb](examples/v16_deepsequence_example.ipynb)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Architecture

Hierarchical attention means **feature / component reweighting** (Level-1 inside experts, Level-2 across experts)—not temporal self-attention over days. Experts use **softsign** outputs by default; DCN **cross layers are off** by default.

```
Panel (id_var, ds, Quantity) + holiday distances
              │
              ▼
   Causal feature builder (v1.6)
     • trend: time_index
     • seasonal: dow / month / year sin-cos
     • lags: lag_1, lag_2, lag_7          (history with ds < t)
     • intermittent: days_since_last_sale,
                     last_sale_quantity,
                     lifetime_cumsum
     • holiday: days_from_* only           (no binary is_*)
              │
              ▼
   Prophet-like hierarchical experts (softsign)
     ┌──────────────┬────────────────┬─────────────────┬──────────────────┐
     │ Trend        │ Seasonal       │ Holiday         │ Regressor        │
     │ softplus-mono│ Fourier +      │ mono distances  │ mono lags/state  │
     │ changepoint  │ masked-entropy │ + selection attn│ + selection attn │
     │ (no L1 attn) │ attn           │                 │                  │
     └──────┬───────┴───────┬────────┴────────┬────────┴────────┬─────────┘
            │               │                 │                 │
            └───────────────┴────────┬────────┴─────────────────┘
                                     │
              Level-2 context-aware mixer
              (SKU emb + lag/intermittent regime; not SKU-only)
                                     │
                                     ▼
                          b = softplus(mix)     magnitude
                          p ∈ (0,1)             occurrence gate
                                     │
                                     ▼
                              ŷ = p · b
```

### Design notes

| Piece | Role |
|-------|------|
| Prophet-like experts | Separate trend / seasonal / holiday / regressor trunks |
| Level-1 attention | Seasonal masked-entropy; holiday/regressor selection attn over monotone maps; trend has **no** L1 attn |
| Softsign experts | Bounded signed expert scalars (default) |
| Level-2 mixer | Context-aware component weights from SKU + demand-regime features |
| Gate \(p\cdot b\) | Occurrence probability × softplus magnitude |
| Cross layers | DCN cross **off** by default (`use_cross_layers=False`) |
| Causal regressors | Lags + intermittent state use **strictly past** Quantity |
| Holiday features (v1.6) | Distance only — `is_*` binaries removed as redundant |
| Training loss | BCE on `p` + gated MAE + nonzero magnitude MAE |

Optional **residual causal transformer** (`residual_transformer.py`) can refine magnitude while preserving DeepSequence’s gate `p`. Default product path stays plain DeepSequence; use the residual head when a panel benefits from sequence residual correction.

---

## Feature contract v1.6

28 columns (see `feature_config.yaml`, also shipped inside the package):

| Group | Count | Columns |
|-------|------:|---------|
| Trend | 1 | `time_index` |
| Seasonal | 6 | `dow_*`, `month_*`, `year_*` sin/cos |
| Lags | 3 | `lag_1`, `lag_2`, `lag_7` |
| Intermittent | 3 | `days_since_last_sale`, `last_sale_quantity`, `lifetime_cumsum` |
| Holiday distance | 15 | `days_from_*` |

---

## Installation

### From this repo (editable)

```bash
git clone https://github.com/mkuma93/DeepSequence.git
cd DeepSequence

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
# optional: tests / notebook
pip install -e ".[dev]"
```

### From a built wheel

```bash
pip install dist/deepsequence_hierarchical_attention-1.6.0-py3-none-any.whl
```

### From Git

```bash
pip install "git+https://github.com/mkuma93/DeepSequence.git"
```

Requires **Python ≥ 3.9**, **TensorFlow ≥ 2.13**, and **tensorflow-recommenders** (cross layers).

---

## Quick example

```python
import numpy as np
import tensorflow as tf
from deepsequence_hierarchical_attention import (
    __version__,
    build_hierarchical_model_lightweight,
    three_term_loss_config,
    get_feature_config_path,
)

print(__version__)                 # 1.6.0
print(get_feature_config_path())   # packaged feature_config.yaml

model = build_hierarchical_model_lightweight(
    n_temporal_features=1,
    n_fourier_features=6,
    n_holiday_features=15,
    n_lag_features=6,   # 3 lags + 3 intermittent
    n_skus=100,
    hidden_dim=48,
    use_intermittent=True,
    use_cross_layers=False,  # opt-in True for DCN ablation
    horizon=1,       # set H > 1 for direct multi-horizon outputs
    use_sku=False,   # disable ID personalization for no-SKU pooling
)

# Compile directly with the gated DeepSequence loss. For adaptive multi-term
# weighting, use training.adaptive_loss.AdaptiveWeightedModel as the sole weighting layer.
# zero_rate must come from data (or an explicit override) — there is no silent 0.9.
zero_rate = float((y_train == 0).mean())
# Optional: per-SKU rates for gate prior (panel mean fills sparse/unseen SKUs)
# from deepsequence_hierarchical_attention import estimate_zero_rate_by_sku
# sku_zr = estimate_zero_rate_by_sku(y_train, sku_train, n_skus=n_skus)
cfg = three_term_loss_config(zero_rate, alpha_bce=0.2, w_gated=1.0, w_mag=1.0)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0025),
    loss=cfg["losses"],
    loss_weights=cfg["weights"],
)
```

`create_model_from_features` requires `zero_rate` or `y_train` (raises if both missing). With `y_train` + `sku_train` it estimates per-SKU rates, wires a non-trainable SKU gate prior, and attaches `model.make_fit_sample_weights(y, sku)` for per-SKU BCE imbalance (relative to the compiled panel `pos_weight`). Pass that dict as `sample_weight` on `fit`. `AdaptiveWeightedModel(..., sku_zero_rates=rates)` does the same in-graph for the adaptive train path.

Causal panel features (lags + intermittent) via:

```python
from deepsequence_hierarchical_attention import transform_panel

feats, states = transform_panel(df, lags=[1, 2, 7], return_states=True)
```

Full feature matrix aligned to v1.6 (including holidays) — use the packaged loader:

```python
from deepsequence_hierarchical_attention.data.feature_config_loader import load_feature_config

cfg = load_feature_config()   # version 1.6, 28 columns
X, states = cfg.create_features(train_df, holiday_df, return_states=True)
```

---

## Optional residual transformer

Module: `deepsequence_hierarchical_attention.residual_transformer`  
Also exported from the package root.

Use this when DeepSequence’s structural forecast (`y_struct`) is good on timing (`p_ds`) but you want a **causal sequence head** to correct magnitude residuals. The head **keeps** DeepSequence’s gate `p_ds` (does not re-learn occurrence).

**Contract**

```
residual = y - y_struct
delta    = ResidualTransformer(lookback seq)
base     = relu(y_struct + delta)
ŷ        = base · p_ds          # same DS probability
```

Default sequence channels (last step’s `y` / residual are masked at predict time; `p_ds` is never masked):

`[y_struct, y, resid, p_ds]`

**Workflow**

1. Train / predict DeepSequence → get `y_struct` (= `base_forecast`) and `p_ds` (= `non_zero_probability`).
2. Build a panel with `y`, `y_struct`, `p_ds` (and optional `split`).
3. Window, train, predict with the residual module.

```python
import numpy as np
import pandas as pd
from deepsequence_hierarchical_attention import (
    build_residual_transformer,
    build_residual_windows,
    train_residual_transformer,
    predict_residual_transformer,
    round_forecast,
)

# panel columns: id_var, ds, y, y_struct, p_ds  (+ optional split)
# resid is computed inside build_residual_windows if missing
lookback = 14
X, y, y_struct, p_ds, sku_ids, splits = build_residual_windows(
    panel, lookback=lookback
)

# map SKU ids → dense ints for Embedding
sku_codes, uniques = pd.factorize(sku_ids)
n_skus = len(uniques)

tr = splits == "train"
va = splits == "val"

model = build_residual_transformer(
    lookback=lookback,
    n_channels=X.shape[-1],  # 4 with default channels
    n_skus=n_skus,
    d_model=32,
    n_heads=4,
    preserve_ds_gate=True,   # default: keep DeepSequence p_ds
)

wrapped = train_residual_transformer(
    model,
    X[tr], y[tr], y_struct[tr], sku_codes[tr],
    X[va], y[va], y_struct[va], sku_codes[va],
    zero_rate=float((y[tr] == 0).mean()),
    epochs=10,
)

final, p, base, delta = predict_residual_transformer(
    model, X[va], y_struct[va], sku_codes[va]
)
yhat = round_forecast(final)  # optional inventory rounding
```

**Notes**

- Default product path remains plain DeepSequence; this head is optional.
- On the v1.6 bake-off it did not beat DeepSequence alone — keep it for panels where residual correction helps.
- Set `preserve_ds_gate=False` only if you intentionally want a new sigmoid gate (legacy).

---

## Example notebook

Run the end-to-end synthetic demo:

```bash
jupyter notebook examples/v16_deepsequence_example.ipynb
```

Or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute examples/v16_deepsequence_example.ipynb
```

The notebook builds v1.6 features, trains gated **DeepSequence** for a few epochs, and prints val MAE / nonzero MAE / bias.

---

## v1.6 bake-off (summary)

Same 28 features for DeepSequence, LightGBM, TST, DeepAR, and TFT-lite (800 series, seed 42) on the confidential evaluation panel described under **Dataset Availability**. Full write-up: **[REPORT_v1.6.md](REPORT_v1.6.md)**.

| Rank | Model | IWMAE | All-day MAE | Nonzero MAE | Occ F1 |
|------|-------|------:|------------:|------------:|-------:|
| 1 | **DeepSequence** | **4.00** | **1.73** | 6.94 | **0.40** |
| 2 | TFT-lite | 4.03 | 1.80 | 6.91 | 0.39 |
| 3 | Temporal transformer | 4.07 | 1.88 | **6.89** | 0.39 |
| 4 | DeepAR-lite | 4.37 | 2.02 | 7.43 | 0.33 |
| 5 | LightGBM | 4.57 | 1.85 | 8.03 | 0.21 |

Primary intermittent ranking is **IWMAE** (all-day MAE alone favors near-zero forecasts under ~90% zeros). Also recorded: MASE (s=7), underforecast rate on sale days, AUROC/AUCPR (see REPORT §2.1 / §3).

- **Low / mid / high IWMAE:** DS best on mid + high; TST edges low  
- **Inventory / service level:** prefer **DS** (best IWMAE + occurrence F1; TFT is the strongest neural runner-up)

Aggregated metrics only (no series identifiers): `eval_results_same_features_v16_distance_holidays.json`

**Multi-horizon (recursive, H=14):** under **IWMAE**, DS wins h=1; TST/TFT lead longer horizons; LightGBM never wins IWMAE despite all-day MAE at h≥7 (worst occ F1 / underforecast). Full tables: [REPORT_v1.6.md](REPORT_v1.6.md) §7 · `eval_results_multihorizon_v16.json`.


---

## Dataset Availability

The experiments were conducted using proprietary enterprise demand data that cannot be publicly released due to confidentiality agreements. To support reproducibility, the repository includes:

* complete model implementation,
* preprocessing pipeline,
* synthetic example dataset,
* training configuration,
* evaluation methodology.

| Included | Location |
|----------|----------|
| Model | `deepsequence_hierarchical_attention/` |
| Preprocessing | `feature_config.yaml`, `deepsequence_hierarchical_attention.data.feature_config_loader`, `intermittent_features.py` |
| Synthetic example | `examples/v16_deepsequence_example.ipynb` |
| Training configuration | `deepsequence_hierarchical_attention/training/training_config.sample.json` |
| Evaluation methodology | `python -m deepsequence_hierarchical_attention.eval.same_features_compare`, `REPORT_v1.6.md` |

This repository does not include company names, product names, customer/series identifiers, internal dashboards, or employer-specific metric names.

---

## Tests

```bash
pytest tests/test_intermittent_features.py -q
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Archive

Legacy packaging and superseded experiment code live under [`archive/`](archive/) (not part of the current model path).
