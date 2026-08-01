#!/usr/bin/env python3
"""Run novelty ablations (daily H=1 DS-only) under Level-1 stack defaults.

Writes under ab_runs/reclaim/ablate_novelty/. Single-seed (42) by default —
do not duplicate multi-seed orchestrator work.
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
ABLATOR = ROOT / "ab_runs/reclaim/ablate_ds_mono_mixer.py"
OUT_DIR = ROOT / "ab_runs/reclaim/ablate_novelty"

# Base = preferred Level-1 stack (softsign + mono + mixer + gate + L1 attn; cross off)
ARMS = [
    ("full", {}),
    ("minus_mixer", {"context_aware_component_mixer": 0}),
    ("minus_level1_attn", {"level1_selection_attention": 0}),
    ("minus_mono", {
        "trend_monotonic": 0,
        "holiday_monotonic": 0,
        "regressor_monotonic": 0,
    }),
    ("minus_gate", {"use_intermittent": 0}),
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
    summary = {"arms": {}, "seed": args.seed, "epochs": args.epochs}

    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"

    for name in [a for a, _ in ARMS if a in wanted]:
        overrides = arm_map[name]
        out_json = OUT_DIR / f"daily_h1_{name}.json"
        log_path = OUT_DIR / f"daily_h1_{name}.log"
        if args.skip_existing and out_json.exists():
            print(f"SKIP existing {out_json}")
            d = json.loads(out_json.read_text())
            summary["arms"][name] = {
                "iwmae": d["models"]["deepsequence"]["overall"]["iwmae"],
                "path": str(out_json),
                "skipped": True,
            }
            continue
        cmd = [
            args.python,
            str(ABLATOR),
            "--data_dir",
            str(args.data_dir),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--label",
            name,
            "--out_json",
            str(out_json),
            # explicit preferred defaults
            "--output_activation",
            "softsign",
            "--trend_monotonic",
            "1",
            "--holiday_monotonic",
            "1",
            "--regressor_monotonic",
            "1",
            "--context_aware_component_mixer",
            "1",
            "--level1_selection_attention",
            "1",
            "--use_intermittent",
            "1",
            "--use_cross_layers",
            "0",
        ]
        for k, v in overrides.items():
            # replace matching flag value
            try:
                idx = cmd.index(f"--{k}")
                cmd[idx + 1] = str(v)
            except ValueError:
                cmd.extend([f"--{k}", str(v)])

        print(f"\n=== {name} ===")
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
        iw = d["models"]["deepsequence"]["overall"]["iwmae"]
        summary["arms"][name] = {
            "iwmae": iw,
            "path": str(out_json),
            "seconds": elapsed,
        }
        print(f"OK {name}: IWMAE={iw:.4f} in {elapsed:.1f}s")

    summary_path = OUT_DIR / "daily_h1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary → {summary_path}")
    for name, row in summary["arms"].items():
        if "iwmae" in row:
            print(f"  {name:20s} IWMAE={row['iwmae']:.4f}")


if __name__ == "__main__":
    main()
