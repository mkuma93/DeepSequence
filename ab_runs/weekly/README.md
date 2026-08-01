# Weekly aggregation artifacts (ISO Monday-start)

- Prepare script: `examples/prepare_weekly_panel.py`
- Feature config: `feature_config_weekly.yaml` (lags `{1,2,4}`, `gap_unit: weeks`)
- Eval: `examples/eval_weekly_mh.py` (DS direct MH + TSB + LightGBM; CumMAE hooked)
- Default smoke out: `ab_runs/weekly/panel*`

Week rule: `Timestamp.to_period("W-SUN").start_time` → **Monday**. Sum `Quantity` by `(id_var, week Monday)`. Country from SKU prefix.

Does **not** change the locked daily bake-off.

## Locked 800 zero-rate (same SKU list as daily)

Artifact: `zero_rate_daily_vs_weekly_locked800.json`

| Grain  | Zero rate | Mean demand |
|--------|----------:|------------:|
| Daily  | 0.896     | 1.04        |
| Weekly | 0.650     | 7.27        |

## Weekly MH bake-off (seed 42, H=8, report 1/4/8)

Artifact: `weekly_mh8_locked800_s42.json` (793 origins with ≥8 test weeks).

| Model        | h=1 IWMAE | h=4 | h=8 | h=1 CumMAE | h=4 | h=8 |
|--------------|----------:|----:|----:|-----------:|----:|----:|
| DeepSequence | **9.95**  | **8.83** | **7.78** | **6.41** | **18.93** | **42.78** |
| TSB          | 10.21     | 10.01 | 10.37 | 7.41 | 20.75 | 43.28 |
| LightGBM     | 10.78     | 10.02 | 13.47 | 8.50 | 21.77 | 49.84 |

Rebuild panel::

```bash
TF_USE_LEGACY_KERAS=1 .venv-test/bin/python examples/prepare_weekly_panel.py \
  --data_dir "$DEEPSEQUENCE_DATA_DIR" \
  --sku_list ab_runs/recompare/sku_list_daily_data42.json \
  --out_dir ab_runs/weekly/panel_locked800
```

Re-run bake-off::

```bash
TF_USE_LEGACY_KERAS=1 .venv-test/bin/python examples/eval_weekly_mh.py \
  --data_dir ab_runs/weekly/panel_locked800 \
  --feature_config feature_config_weekly.yaml \
  --sku_list ab_runs/recompare/sku_list_daily_data42.json \
  --max_skus 800 --horizon 8 --report_horizons 1,4,8 \
  --models deepsequence,tsb,lightgbm --epochs 15 --seed 42 \
  --out_json ab_runs/weekly/weekly_mh8_locked800_s42.json
```
