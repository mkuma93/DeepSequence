#!/usr/bin/env python3
"""DS-only H=1 probe: year-scoped US holiday CSVs vs locked nearest CSVs.

Rebuilds days_from_* with distance_scope='year' (US calendar, matching locked
feature_config.yaml — not country calendars) into a sibling data dir, then runs
DeepSequence-only on a locked-list subset for a paired IWMAE note.

Does NOT claim a full multi-seed bake-off.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from deepsequence_hierarchical_attention.holidays.calendar import HOLIDAY_KEYS, days_from_holiday_features  # noqa: E402

DATA = Path(
    os.environ.get(
        "DEEPSEQUENCE_DATA_DIR",
        "/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
    )
)
OUT = ROOT / "paper_figures" / "year_scope_retest"
YEAR_DIR = OUT / "daily_holiday_year_scope_us"
SKU_LIST = ROOT / "ab_runs" / "recompare" / "sku_list_daily_data42.json"
N_SKUS = int(os.environ.get("YEAR_SCOPE_N_SKUS", "150"))
EPOCHS = int(os.environ.get("YEAR_SCOPE_EPOCHS", "12"))


def _write_year_scope_holidays():
    YEAR_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "train_split.csv",
        "val_split.csv",
        "test_split.csv",
        "split_metadata.csv",
    ):
        src = DATA / name
        dst = YEAR_DIR / name
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)

    for split, split_csv, out_name in (
        ("train", "train_split.csv", "holiday_features_train.csv"),
        ("val", "val_split.csv", "holiday_features_val.csv"),
        ("test", "test_split.csv", "holiday_features_test.csv"),
    ):
        df = pd.read_csv(DATA / split_csv, parse_dates=["ds"])
        hol = days_from_holiday_features(
            df["ds"], holiday_keys=HOLIDAY_KEYS, country="US", distance_scope="year"
        )
        hol.to_csv(YEAR_DIR / out_name, index=False)
        # quick sanity vs locked nearest
        locked = pd.read_csv(DATA / out_name)
        n_diff = int((hol.to_numpy() != locked.to_numpy()).any(axis=1).sum())
        print(
            f"wrote {out_name}: rows={len(hol)} rows_differing_from_nearest={n_diff} "
            f"Jan-ish Christmas example year={hol.loc[0, 'days_from_Christmas']} "
            f"nearest={locked.loc[0, 'days_from_Christmas']}"
        )


def _run_ds(data_dir: Path, tag: str) -> Path:
    out_json = OUT / f"daily_h1_ds_only_{tag}_n{N_SKUS}_s42.json"
    log = OUT / f"daily_h1_ds_only_{tag}_n{N_SKUS}_s42.log"
    cmd = [
        str(ROOT / ".venv-test" / "bin" / "python"),
        "-u",
        "-m",
        "deepsequence_hierarchical_attention.eval.same_features_compare",
        "--data_dir",
        str(data_dir),
        "--models",
        "deepsequence",
        "--max_skus",
        str(N_SKUS),
        "--seed",
        "42",
        "--epochs",
        str(EPOCHS),
        "--sku_list",
        str(SKU_LIST),
        "--out_json",
        str(out_json),
    ]
    print("running:", " ".join(cmd))
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env["XDG_CACHE_HOME"] = "/tmp/xdgcache"
    with open(log, "w") as fh:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT
        )
    if proc.returncode != 0:
        raise SystemExit(f"eval failed ({tag}); see {log}")
    return out_json


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== rebuild year-scope US holiday CSVs ===")
    _write_year_scope_holidays()

    print("=== DS-only nearest (locked CSVs) ===")
    nearest_json = _run_ds(DATA, "nearest")
    print("=== DS-only year-scope ===")
    year_json = _run_ds(YEAR_DIR, "year")

    near = json.loads(nearest_json.read_text())
    year = json.loads(year_json.read_text())
    ni = near["models"]["deepsequence"]["overall"]["iwmae_rounded"]
    yi = year["models"]["deepsequence"]["overall"]["iwmae_rounded"]
    prior800 = None
    prior_path = ROOT / "ab_runs" / "recompare" / "daily_h1_s42.json"
    if prior_path.exists():
        prior800 = json.loads(prior_path.read_text())["models"]["deepsequence"][
            "overall"
        ]["iwmae_rounded"]
    summary = {
        "n_skus": N_SKUS,
        "epochs": EPOCHS,
        "seed": 42,
        "sku_list": str(SKU_LIST),
        "holiday_rebuild": "US calendar days_from_* with distance_scope=year",
        "year_scope_data_dir": str(YEAR_DIR),
        "iwmae_rounded_nearest_subset": ni,
        "iwmae_rounded_year_subset": yi,
        "delta_year_minus_nearest": yi - ni,
        "prior_artifact_800_ds_iwmae_rounded": prior800,
        "note": (
            "Paired subset probe only; not a full multi-seed locked bake-off. "
            "Year CSVs are US-only (locked contract), not country calendars."
        ),
    }
    out = OUT / f"daily_h1_ds_year_vs_nearest_n{N_SKUS}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
