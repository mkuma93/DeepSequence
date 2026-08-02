# Weekly / daily Direct-MH artifacts (ISO Monday weekly; locked 800)

- Panel prepare: `python -m deepsequence_hierarchical_attention.data.prepare_weekly_panel`
- Feature config (weekly): `feature_config_weekly.yaml` (lags `{1,2,4}`, `gap_unit: weeks`)
- Bake-off: `python -m deepsequence_hierarchical_attention.eval.weekly_mh`
  - Weekly: DS **direct MH** + TSB recursive + LightGBM multi-output
  - Daily like-for-like: same runner with `--dataset daily_direct_mh` + `feature_config.yaml`

Week rule: `Timestamp.to_period("W-SUN").start_time` → **Monday**. Sum `Quantity` by `(id_var, week Monday)`.

Does **not** change the locked **recursive** daily bake-off (`PAPER.md` Table 1).

## Protocol (direct↔direct)

Grain comparisons in `PAPER.md` §5.3b hold MH protocol fixed:

| Grain | DeepSequence | LightGBM | TSB | Artifact |
|-------|--------------|----------|-----|----------|
| Weekly | direct MH | multi-output | recursive classical | `weekly_mh8_locked800_s42.json` |
| Daily (comparator) | direct MH | multi-output | recursive classical | `daily_direct_mh60_locked800_s42.json` |

Matched leads: weekly \(h=1/4/8\) ≈ daily \(h=7/28/56\). Absolute IWMAE is **not** cross-grain comparable (week-sums vs daily units). Recursive daily remains primary for the portfolio story.

Ideal remaining follow-up: **recursive weekly** (protocol matched to Table 1).

## Locked 800 zero-rate

Artifact: `zero_rate_daily_vs_weekly_locked800.json`

| Grain  | Zero rate | Mean demand |
|--------|----------:|------------:|
| Daily  | 0.896     | 1.04        |
| Weekly | 0.650     | 7.27        |

## Stratified evaluation (train zones; seed 42)

SKU bands from **train** only (no test leakage).

| Protocol | Primary zones | Secondary | Artifact |
|----------|---------------|-----------|----------|
| Daily recursive (Table 1) | Train **volume-sum** terciles → low/mid/high (locked bake-off) | — | Nested in `../reclaim/daily_mh_1_60_cummae_s42.json`; summary `../reclaim/strata_volume_s42.json` |
| Weekly Direct-MH | Train **mean-demand** terciles → low/mid/high | Train **zero-rate** terciles → high_zero / mid / low_zero | `weekly_mh8_locked800_s42.json` + `strata_weekly_direct_s42.json` |

Extract / rebuild summary:

```bash
.venv-test/bin/python ab_runs/reclaim/extract_strata_tables.py \
  --daily_json ab_runs/reclaim/daily_mh_1_60_cummae_s42.json \
  --weekly_json ab_runs/weekly/weekly_mh8_locked800_s42.json \
  --out_json ab_runs/reclaim/strata_volume_s42.json
```

**Weekly mean-demand IWMAE (headline):** DS wins mid/high at \(h=1/4/8\) and low at \(h\ge4\); TSB edges \(h=1\) low. See `PAPER.md` Tables W-S1 / W-S2.

## Weekly Direct-MH (seed 42, H=8)

793 origins with ≥8 test weeks.

| Model        | h=1 IWMAE | h=4 | h=8 | h=1 CumMAE | h=4 | h=8 |
|--------------|----------:|----:|----:|-----------:|----:|----:|
| DeepSequence | **9.95**  | **8.83** | **7.78** | **6.41** | **18.93** | **42.78** |
| TSB          | 10.21     | 10.01 | 10.37 | 7.41 | 20.75 | 43.28 |
| LightGBM     | 10.78     | 10.02 | 13.47 | 8.50 | 21.77 | 49.84 |

## Daily Direct-MH (seed 42, H=60)

696 origins with ≥60 test days. Artifact: `daily_direct_mh60_locked800_s42.json`.

| Model        | h=1 | h=7 | h=14 | h=28 | h=56 | h=60 | h=7 CumMAE | h=28 | h=56 |
|--------------|----:|----:|-----:|-----:|-----:|-----:|-----------:|-----:|-----:|
| DeepSequence | 5.60 | **3.26** | **3.71** | **3.87** | **9.09** | **2.51** | **10.86** | **38.34** | **76.86** |
| TSB          | **5.32** | 4.49 | 5.30 | 5.36 | 10.83 | 4.24 | 17.42 | 99.62 | 210.89 |
| LightGBM     | 5.85 | 4.37 | 5.33 | 6.22 | 10.29 | 3.71 | 15.92 | 75.53 | 142.50 |

## Rebuild commands

```bash
export DEEPSEQUENCE_DATA_DIR=/path/to/jubilant/data
TF_USE_LEGACY_KERAS=1 .venv-test/bin/python -m deepsequence_hierarchical_attention.data.prepare_weekly_panel \
  --data_dir "$DEEPSEQUENCE_DATA_DIR" \
  --sku_list ab_runs/recompare/sku_list_daily_data42.json \
  --out_dir ab_runs/weekly/panel_locked800

TF_USE_LEGACY_KERAS=1 .venv-test/bin/python -m deepsequence_hierarchical_attention.eval.weekly_mh \
  --data_dir ab_runs/weekly/panel_locked800 \
  --feature_config feature_config_weekly.yaml \
  --sku_list ab_runs/recompare/sku_list_daily_data42.json \
  --max_skus 800 --horizon 8 --report_horizons 1,4,8 \
  --models deepsequence,tsb,lightgbm --epochs 15 --seed 42 \
  --out_json ab_runs/weekly/weekly_mh8_locked800_s42.json

TF_USE_LEGACY_KERAS=1 .venv-test/bin/python -m deepsequence_hierarchical_attention.eval.weekly_mh \
  --data_dir "$DEEPSEQUENCE_DATA_DIR" \
  --feature_config feature_config.yaml --dataset daily_direct_mh \
  --sku_list ab_runs/recompare/sku_list_daily_data42.json \
  --max_skus 800 --horizon 60 --report_horizons 1,7,14,28,56,60 \
  --mase_season 7 --models deepsequence,tsb,lightgbm --epochs 15 --seed 42 \
  --out_json ab_runs/weekly/daily_direct_mh60_locked800_s42.json
```

Figures: `paper_figures/make_weekly_daily_direct_compare.py` → `fig_zero_rate_daily_vs_weekly`, `fig_weekly_daily_direct_iwmae`, `fig_weekly_daily_direct_cummae`.

Qualitative per-SKU forecasts (Direct-MH, seed 42):

```bash
TF_USE_LEGACY_KERAS=1 .venv-test/bin/python paper_figures/make_forecast_weekly_plots.py \
  --epochs 15 --max_skus 800 --seed 42
```

→ `fig_forecast_weekly_onestep` / `fig_forecast_weekly_direct` (+ pdf/json).
