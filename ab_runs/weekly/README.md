# Weekly aggregation artifacts (ISO Monday-start)

- Prepare script: `examples/prepare_weekly_panel.py`
- Feature config: `feature_config_weekly.yaml` (lags `{1,2,4}`, `gap_unit: weeks`)
- Default smoke out: `ab_runs/weekly/panel*`

Week rule: `Timestamp.to_period("W-SUN").start_time` → **Monday**. Sum `Quantity` by `(id_var, week Monday)`. Country from SKU prefix.

Does **not** change the locked daily bake-off.
