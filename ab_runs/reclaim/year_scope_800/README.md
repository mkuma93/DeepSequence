# Year-scope locked 800 retest (seed 42)

## Holiday rebuild (A — US, apples-to-apples)
- Source splits: Jubilant data dir (outside repo; see `holiday_verify.json`)
- Regenerated dir: `holiday_features_year/` (gitignored `*.csv`; rebuild via `run_year_scope_800.py`)
- `distance_scope='year'`, calendar=`US`, keys=`HOLIDAY_KEYS` (15 locked keys)
- Locked jubilant CSVs already year-scoped: **true** (max abs vs locked = 0 on all splits)
- Explicit `nearest` differs (sample max abs ≈ 365–385) — bug class is real, but locked assets were already on the year path
- Optional (B) country+year Jubilant rebuild **not** run (would change protocol vs published US lock)

## Eval outputs (locked stack: softsign + mono + L1 + mixer on + cross off)
- `daily_h1_s42.json` — one-step DS / TST / LightGBM
- `daily_mh60_s42.json` — recursive report horizons 1/7/14/28/60 (primary vs Table 1)
- `daily_loyalty_s42.json` — mid-margin π with \(C_{\mathrm{loyalty}}=0.25\)
- `comparison_vs_prior.json` — deltas vs reclaim softsign H1 + multiseed s42 MH
- `holiday_verify.json` — checksums + max-abs audit

## Seed-42 recursive IWMAE (year-scope retest)
| h | DS | TST | LGBM | Δ DS vs prior | Δ TST vs prior |
|--:|---:|----:|-----:|--------------:|---------------:|
| 1 | 4.035 | 3.865 | 4.451 | 0 | +0.005 |
| 7 | 4.381 | 4.347 | 4.688 | 0 | +0.024 |
| 14 | 4.211 | 4.229 | 4.615 | 0 | +0.053 |
| 28 | 6.417 | 6.930 | 6.866 | 0 | +0.053 |
| 60 | 3.891 | 4.560 | 4.375 | 0 | +0.065 |

DS / LightGBM match prior multiseed s42 exactly; TST within TF train noise. Rankings unchanged (TST short; DS long).

## Monthly / Car Parts
Locked monthly holidays are OFF (`feature_config_monthly.yaml`); no day-distance CSV path — monthly bake-off unchanged.

## Pending
- Seeds 43–46
- Optional country+year Jubilant features (protocol change)
