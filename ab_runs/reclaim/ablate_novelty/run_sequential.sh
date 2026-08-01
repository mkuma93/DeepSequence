#!/usr/bin/bash
set -euo pipefail
cd "/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/mkuma93-profile/DeepSequence"
export TF_USE_LEGACY_KERAS=1 MPLCONFIGDIR=/tmp/mpl CMDSTAN_VERBOSE=FALSE PYTHONUNBUFFERED=1
PY=".venv-test/bin/python"
DATA_DAILY="/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data"
LOG="ab_runs/reclaim/ablate_novelty/sequential_runner.log"
exec >>"$LOG" 2>&1

echo "===== START $(date) ====="

echo "===== 1) Prophet carparts monthly (n_jobs=1) ====="
$PY ab_runs/reclaim/eval_prophet_baseline.py \
  --dataset carparts --max_skus 800 --seed 42 --horizons 1,2,6 --n_jobs 1 \
  --out_json ab_runs/reclaim/prophet_carparts/carparts_mh_1_2_6.json

echo "===== 2) Daily H=1 novelty ablations ====="
$PY ab_runs/reclaim/run_novelty_ablations.py \
  --data_dir "$DATA_DAILY" --epochs 25 --seed 42 --skip_existing 1

echo "===== 3) Daily Prophet subset 150 ====="
$PY ab_runs/reclaim/eval_prophet_baseline.py \
  --dataset daily --data_dir "$DATA_DAILY" \
  --max_skus 150 --seed 42 --horizons 1,28,60 --max_origins_per_sku 4 --n_jobs 1 \
  --out_json ab_runs/reclaim/prophet_daily/daily_subset150_h1_28_60.json

echo "===== 4) Daily MH60 DS-only novelty ====="
$PY ab_runs/reclaim/run_novelty_ablations_mh.py \
  --data_dir "$DATA_DAILY" --horizon 60 --report_horizons 1,28,60 \
  --epochs 25 --seed 42 --skip_existing 1

echo "===== DONE $(date) ====="
