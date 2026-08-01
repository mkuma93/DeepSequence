#!/usr/bin/bash
# Watchdog: restart Prophet until final JSON exists, then H1 + daily Prophet + MH.
set -u
cd "/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/mkuma93-profile/DeepSequence"
export TF_USE_LEGACY_KERAS=1 MPLCONFIGDIR=/tmp/mpl CMDSTAN_VERBOSE=FALSE PYTHONUNBUFFERED=1
PY=".venv-test/bin/python"
DATA_DAILY="/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data"
LOG="ab_runs/reclaim/ablate_novelty/watchdog.log"
OUT_P="ab_runs/reclaim/prophet_carparts/carparts_mh_1_2_6.json"

exec >>"$LOG" 2>&1
echo "===== WATCHDOG START $(date) ====="

attempt=0
while [[ ! -f "$OUT_P" ]]; do
  attempt=$((attempt+1))
  echo "----- Prophet attempt $attempt $(date) -----"
  $PY ab_runs/reclaim/eval_prophet_baseline.py \
    --dataset carparts --max_skus 800 --seed 42 --horizons 1,2,6 --n_jobs 1 \
    --out_json "$OUT_P" || echo "Prophet exited rc=$?"
  if [[ -f "$OUT_P" ]]; then
    echo "Prophet complete"
    break
  fi
  # brief pause before resume
  sleep 5
  if [[ $attempt -ge 40 ]]; then
    echo "Giving up Prophet after $attempt attempts"
    break
  fi
done

echo "===== H1 novelty $(date) ====="
$PY ab_runs/reclaim/run_novelty_ablations.py \
  --data_dir "$DATA_DAILY" --epochs 25 --seed 42 --skip_existing 1 || true

echo "===== Daily Prophet subset $(date) ====="
$PY ab_runs/reclaim/eval_prophet_baseline.py \
  --dataset daily --data_dir "$DATA_DAILY" \
  --max_skus 150 --seed 42 --horizons 1,28,60 --max_origins_per_sku 4 --n_jobs 1 \
  --out_json ab_runs/reclaim/prophet_daily/daily_subset150_h1_28_60.json || true

# Wait for multiseed to release GPU/CPU if still running
while pgrep -f "eval_multihorizon_compare.py.*daily_s4" >/dev/null 2>&1; do
  echo "Waiting for multiseed MH to finish... $(date)"
  sleep 60
done

echo "===== MH novelty $(date) ====="
$PY ab_runs/reclaim/run_novelty_ablations_mh.py \
  --data_dir "$DATA_DAILY" --horizon 60 --report_horizons 1,28,60 \
  --epochs 25 --seed 42 --skip_existing 1 || true

echo "===== WATCHDOG DONE $(date) ====="
