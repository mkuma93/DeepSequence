"""Paired summary of the baseline vs fixed Car Parts A/B runs."""

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(42, 57))
METRICS = ("iwmae_rounded", "mae_all_rounded", "occ_f1", "underforecast_rate_nonzero", "aucroc", "bias")


def load(arm, horizon, seed):
    path = ROOT / arm / f"h{horizon}_s{seed}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload["models"]["deepsequence"]["overall"]


def paired(horizon, metric):
    rows = []
    for seed in SEEDS:
        b, f = load("baseline", horizon, seed), load("fixed", horizon, seed)
        if b is None or f is None or b.get(metric) is None or f.get(metric) is None:
            continue
        rows.append((seed, b[metric], f[metric]))
    return rows


def report(horizon):
    print(f"\n{'=' * 72}\nhorizon={horizon}  (lower iwmae/mae/under is better)\n{'=' * 72}")
    for metric in METRICS:
        rows = paired(horizon, metric)
        if not rows:
            continue
        base = [r[1] for r in rows]
        fix = [r[2] for r in rows]
        deltas = [f - b for _, b, f in rows]
        mean_d = st.mean(deltas)
        se = st.stdev(deltas) / len(deltas) ** 0.5 if len(deltas) > 1 else float("nan")
        wins = sum(1 for d in deltas if d < 0)
        print(
            f"  {metric:28s} base={st.mean(base):.4f} fixed={st.mean(fix):.4f} "
            f"delta={mean_d:+.4f} +/-{1.96 * se:.4f} (95% CI)  fixed_lower={wins}/{len(deltas)}"
        )
    rows = paired(horizon, "iwmae_rounded")
    print("  per-seed iwmae (seed: base -> fixed):")
    for seed, b, f in rows:
        flag = "better" if f < b else ("worse" if f > b else "tie")
        print(f"    {seed}: {b:.3f} -> {f:.3f}  ({f - b:+.3f}, {flag})")


for horizon in (1, 6):
    report(horizon)
