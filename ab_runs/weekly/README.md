# Weekly aggregation artifacts (ISO Monday-start)

- Prepare script: `python -m deepsequence_hierarchical_attention.data.prepare_weekly_panel`
- Feature config: `feature_config_weekly.yaml` (lags `{1,2,4}`, `gap_unit: weeks`)
- Eval: `python -m deepsequence_hierarchical_attention.eval.weekly_mh` (DS **direct MH** + TSB recursive classical + LightGBM multi-output; CumMAE hooked)
- Default smoke out: `ab_runs/weekly/panel*`

Week rule: `Timestamp.to_period("W-SUN").start_time` → **Monday**. Sum `Quantity` by `(id_var, week Monday)`. Country from SKU prefix.

Does **not** change the locked daily bake-off.

## Protocol caveat (read before citing)

Relative to the locked **daily recursive** MH bake-off (`PAPER.md` Table 1), this weekly experiment **jointly** changes:

1. **Temporal grain / zero rate** (daily ≈90% zeros → weekly ≈65% zeros), and
2. **Forecasting protocol** (daily DeepSequence = recursive one-step rollout; weekly DeepSequence = **direct** multi-horizon head).

**Do not** attribute all weekly IWMAE / flatness improvement solely to aggregation or the drop in zero rate. Within-table weekly rankings (DS vs TSB vs LightGBM) are valid under the weekly protocol; cross-grain comparison to daily Table 1 is not like-for-like.

Ideal follow-up (not run here): **recursive weekly** and/or **direct daily** with protocol held fixed. See `PAPER.md` §3.10, §5.3b, §7.

## Locked 800 zero-rate (same SKU list as daily)

Artifact: `zero_rate_daily_vs_weekly_locked800.json`

| Grain  | Zero rate | Mean demand |
|--------|----------:|------------:|
| Daily  | 0.896     | 1.04        |
| Weekly | 0.650     | 7.27        |

## Weekly MH bake-off (seed 42, H=8, report 1/4/8)

Artifact: `weekly_mh8_locked800_s42.json` (793 origins with ≥8 test weeks). Protocol: DS direct MH (not recursive).

| Model        | h=1 IWMAE | h=4 | h=8 | h=1 CumMAE | h=4 | h=8 |
|--------------|----------:|----:|----:|-----------:|----:|----:|
| DeepSequence | **9.95**  | **8.83** | **7.78** | **6.41** | **18.93** | **42.78** |
| TSB          | 10.21     | 10.01 | 10.37 | 7.41 | 20.75 | 43.28 |
| LightGBM     | 10.78     | 10.02 | 13.47 | 8.50 | 21.77 | 49.84 |

Rebuild panel::

```bash
TF_USE_LEGACY_KERAS=1 .venv-test/bin/python -m deepsequence_hierarchical_attention.data.prepare_weekly_panel \
  --data_dir "$DEEPSEQUENCE_DATA_DIR" \
  --sku_list ab_runs/recompare/sku_list_daily_data42.json \
  --out_dir ab_runs/weekly/panel_locked800
```

Re-run bake-off::

```bash
TF_USE_LEGACY_KERAS=1 .venv-test/bin/python -m deepsequence_hierarchical_attention.eval.weekly_mh \
  --data_dir ab_runs/weekly/panel_locked800 \
  --feature_config feature_config_weekly.yaml \
  --sku_list ab_runs/recompare/sku_list_daily_data42.json \
  --max_skus 800 --horizon 8 --report_horizons 1,4,8 \
  --models deepsequence,tsb,lightgbm --epochs 15 --seed 42 \
  --out_json ab_runs/weekly/weekly_mh8_locked800_s42.json
```
