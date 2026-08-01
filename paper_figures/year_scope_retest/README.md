# Year-scope holiday retest (post-4e2ec63)

Artifacts from retesting holiday qualitative / probe runs after
`distance_scope='year'` became the default for rebuilt calendars.

## Qualitative (committed figs under `paper_figures/`)
- Daily country+binary: `fig_forecast_daily_country_hol_*.{png,pdf,json}`
- Monthly months_from+month_has: `fig_forecast_carparts_country_hol_*.{png,pdf,json}`
- Prior (superseded) JSON snapshots: `prior_*.json`

## H=1 DS probe
- `daily_h1_ds_year_vs_nearest_n150_summary.json` — paired 150-SKU note
- Regenerated US year-scope CSVs under `daily_holiday_year_scope_us/` are
  **not** committed (large); rebuild via `../run_year_scope_h1_ds_probe.py`.

Locked jubilant `holiday_features_*.csv` already match year-scope rebuild
(max abs diff 0) on this panel.
