# DeepSequence Hierarchical Attention

Intermittent demand forecasting with a **lightweight hierarchical DeepSequence** backbone, **causal intermittent features**, and a **gated** occurrence × magnitude head.

**Package version:** 1.6.0 · **Feature contract:** v1.6 (28 columns)  
**Report:** [REPORT_v1.6.md](REPORT_v1.6.md) · **Example notebook:** [examples/v16_deepsequence_example.ipynb](examples/v16_deepsequence_example.ipynb)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Architecture

Hierarchical attention here means **feature / component reweighting**, not temporal self-attention over days.

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
   Lightweight hierarchical DeepSequence
     ┌─────────┬───────────┬──────────┬────────────┐
     │  Trend  │ Seasonal  │ Holiday  │ Regressor  │
     │ (PWL /  │ (Fourier  │ (distance│ (lags +    │
     │ change- │  + attn)  │  + attn) │ intermittent)│
     └────┬────┴─────┬─────┴────┬─────┴──────┬─────┘
          │          │          │            │
          └──────────┴────┬─────┴────────────┘
                          │
              SKU embedding → soft component weights
              (static per SKU; cross layers optional)
                          │
                          ▼
                   base_forecast (softplus)
                          │
              Intermittent gate p ∈ (0,1)
                          │
                          ▼
              ŷ = p · base_forecast
```

### Design notes

| Piece | Role |
|-------|------|
| Hierarchical components | Separate trend / seasonal / holiday / regressor experts |
| Component attention | Masked entropy attention + SKU shift/scale |
| SKU weights | Soft mixture over components (not day-level self-attention) |
| Gate `p` | Occurrence probability; final demand is gated magnitude |
| Causal regressors | Lags + intermittent state use **strictly past** Quantity |
| Holiday features (v1.6) | Distance only — `is_*` binaries removed as redundant |
| Training loss | BCE on `p` + gated MAE + nonzero magnitude MAE |

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
cd DeepSequence/deepsequence_hierarchical_attention

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

### From Git (subdirectory)

```bash
pip install "git+https://github.com/mkuma93/DeepSequence.git#subdirectory=deepsequence_hierarchical_attention"
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
    use_cross_layers=True,
)

# Compile with the gated DeepSequence loss (or use examples/AdaptiveWeightedModel)
zero_rate = 0.9
cfg = three_term_loss_config(zero_rate, alpha_bce=0.2, w_gated=1.0, w_mag=1.0)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0025),
    loss=cfg["losses"],
    loss_weights=cfg["weights"],
)
```

Causal panel features (lags + intermittent) via:

```python
from deepsequence_hierarchical_attention import transform_panel

feats, states = transform_panel(df, lags=[1, 2, 7], return_states=True)
```

Full feature matrix aligned to v1.6 (including holidays) — use `examples/feature_config_loader.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("examples").resolve()))
from feature_config_loader import load_feature_config

cfg = load_feature_config()   # version 1.6, 28 columns
X, states = cfg.create_features(train_df, holiday_df, return_states=True)
```

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

Same 28 features for DeepSequence, LightGBM, TST, and DeepAR (800 SKUs, seed 42). Full write-up: **[REPORT_v1.6.md](REPORT_v1.6.md)**.

| Rank | Model | All-day MAE | Nonzero MAE |
|------|-------|------------:|------------:|
| 1 | **DeepSequence** | **1.73** | 6.94 |
| 2 | LightGBM | 1.85 | 8.03 |
| 3 | Temporal transformer | 1.88 | **6.89** |
| 4 | DeepAR-lite | 2.02 | 7.43 |

- **Low / mid volume:** DS best  
- **High nonzero (sale days):** TST ≈ DS; LightGBM worst (under-forecasts)  
- **Inventory / service level:** prefer **DS** (does not under-forecast like LightGBM on high volume)

Metrics JSON: `eval_results_same_features_v16_distance_holidays.json`

---

## Tests

```bash
pytest tests/test_intermittent_features.py -q
```

---

## License

MIT — see [LICENSE](LICENSE).
