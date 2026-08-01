#!/usr/bin/env python3
"""DS-only multi-horizon novelty ablations (locked daily panel).

Runs eval_multihorizon_compare.py with --models deepsequence under Level-1
stack defaults, varying one novelty factor. Outputs under
ab_runs/reclaim/ablate_novelty/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "examples/eval_multihorizon_compare.py"
OUT_DIR = ROOT / "ab_runs/reclaim/ablate_novelty"

ARMS = [
    ("full", {}),
    ("minus_mixer", {"context_aware_component_mixer": 0}),
    ("minus_level1_attn", {"level1_selection_attention": 0}),
    ("minus_mono", {
        "trend_monotonic": 0,
        "holiday_monotonic": 0,
        "regressor_monotonic": 0,
    }),
    ("plus_cross", {"use_cross_layers": 1}),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default=os.environ.get(
            "DEEPSEQUENCE_DATA_DIR",
            "/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/"
            "My Drive/jubilant/data",
        ),
    )
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--report_horizons", default="1,28,60")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", default=",".join(a for a, _ in ARMS))
    p.add_argument("--python", default=str(ROOT / ".venv-test/bin/python"))
    p.add_argument("--skip_existing", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
    arm_map = dict(ARMS)
    summary = {
        "protocol": "recursive MH DS-only novelty ablations",
        "horizon": args.horizon,
        "report_horizons": args.report_horizons,
        "seed": args.seed,
        "arms": {},
    }
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"

    for name in [a for a, _ in ARMS if a in wanted]:
        overrides = arm_map[name]
        out_json = OUT_DIR / f"daily_mh60_{name}.json"
        log_path = OUT_DIR / f"daily_mh60_{name}.log"
        if args.skip_existing and out_json.exists():
            print(f"SKIP existing {out_json}")
            d = json.loads(out_json.read_text())
            bh = d["models"]["deepsequence"]["by_horizon"]
            summary["arms"][name] = {
                "iwmae_by_h": {
                    h: bh[h]["overall"]["iwmae"] for h in bh if "overall" in bh[h]
                },
                "path": str(out_json),
                "skipped": True,
            }
            continue
        cmd = [
            args.python,
            str(EVAL),
            "--data_dir",
            str(args.data_dir),
            "--horizon",
            str(args.horizon),
            "--report_horizons",
            args.report_horizons,
            "--models",
            "deepsequence",
            "--sku_list",
            "ab_runs/recompare/sku_list_daily_data42.json",
            "--data_seed",
            str(args.seed),
            "--seed",
            str(args.seed),
            "--train_seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--use_cross_layers",
            "0",
            "--context_aware_component_mixer",
            "1",
            "--level1_selection_attention",
            "1",
            "--trend_monotonic",
            "1",
            "--holiday_monotonic",
            "1",
            "--regressor_monotonic",
            "1",
            "--out_json",
            str(out_json),
        ]
        for k, v in overrides.items():
            try:
                idx = cmd.index(f"--{k}")
                cmd[idx + 1] = str(v)
            except ValueError:
                cmd.extend([f"--{k}", str(v)])
        print(f"\n=== MH {name} ===")
        print(" ".join(cmd))
        t0 = time.time()
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT
            )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"FAILED {name} rc={proc.returncode} log={log_path}")
            summary["arms"][name] = {"error": proc.returncode, "log": str(log_path)}
            continue
        d = json.loads(out_json.read_text())
        bh = d["models"]["deepsequence"]["by_horizon"]
        row = {
            h: bh[h]["overall"]["iwmae"] for h in bh if "overall" in bh[h]
        }
        summary["arms"][name] = {
            "iwmae_by_h": row,
            "path": str(out_json),
            "seconds": elapsed,
        }
        print(f"OK {name}: {row} in {elapsed:.1f}s")

    summary_path = OUT_DIR / "daily_mh60_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
