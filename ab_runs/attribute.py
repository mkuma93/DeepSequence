"""Attribute the horizon-6 changes to the gate fix alone vs all four fixes."""

import json
import math
import statistics as st
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(42, 57))
HORIZON = 6
METRICS = ("aucroc", "occ_f1", "iwmae_rounded")
HIGHER_IS_BETTER = {"aucroc", "occ_f1"}


def load(arm, seed):
    path = ROOT / arm / f"h{HORIZON}_s{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["models"]["deepsequence"]["overall"]


def sign_test_p(better, total):
    """Two-sided exact binomial p-value under p=0.5 (ties excluded)."""
    if total == 0:
        return float("nan")
    def tail(k):
        return sum(math.comb(total, i) for i in range(0, k + 1)) / 2 ** total
    low = min(better, total - better)
    return min(1.0, 2 * tail(low))


def compare(arm, label):
    print(f"\n{'=' * 74}\n{label}  (horizon {HORIZON}, vs baseline, {len(SEEDS)} paired seeds)\n{'=' * 74}")
    for metric in METRICS:
        rows = []
        for seed in SEEDS:
            b, a = load("baseline", seed), load(arm, seed)
            if b is None or a is None:
                continue
            rows.append((seed, b[metric], a[metric]))
        deltas = [a - b for _, b, a in rows]
        mean_d = st.mean(deltas)
        ci = 1.96 * st.stdev(deltas) / len(deltas) ** 0.5
        if metric in HIGHER_IS_BETTER:
            better = sum(1 for d in deltas if d > 0)
        else:
            better = sum(1 for d in deltas if d < 0)
        ties = sum(1 for d in deltas if d == 0)
        p = sign_test_p(better, len(deltas) - ties)
        direction = "higher=better" if metric in HIGHER_IS_BETTER else "lower=better"
        print(
            f"  {metric:16s} ({direction:13s}) delta={mean_d:+.4f} +/-{ci:.4f}  "
            f"better={better}/{len(deltas)}  sign-test p={p:.3f}"
        )


compare("mh_only", "GATE FIX ALONE")
compare("fixed", "ALL FOUR FIXES")

print(f"\n{'=' * 74}\nAUC-ROC per seed: baseline -> gate-only -> all-four\n{'=' * 74}")
for seed in SEEDS:
    b, m, f = load("baseline", seed), load("mh_only", seed), load("fixed", seed)
    if not all((b, m, f)):
        continue
    print(
        f"  {seed}: {b['aucroc']:.4f} -> {m['aucroc']:.4f} -> {f['aucroc']:.4f}"
    )
