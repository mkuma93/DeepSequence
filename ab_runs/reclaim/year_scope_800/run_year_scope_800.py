#!/usr/bin/env python3
"""Locked 800-SKU year-scope holiday retest (seed 42).

(A) Rebuild US ``days_from_*`` with ``distance_scope='year'`` (same holiday key
    set as locked ``feature_config.yaml`` / jubilant CSVs).
(B) Optional country+year rebuild is out of this script's default path.

Writes under ``ab_runs/reclaim/year_scope_800/``:
  holiday_features_year/   regenerated US year-scope CSVs (+ split symlinks)
  daily_h1_s42.json        one-step DS / TST / LightGBM
  daily_mh60_s42.json      recursive H=1/7/14/28/60
  holiday_verify.json      max-abs vs locked + vs explicit nearest
  comparison_vs_prior.json seed-42 deltas vs reclaim / multiseed artifacts
  README.md                paths + protocol notes

Large CSVs are gitignored; keep checksums + verification JSON in-repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from deepsequence_hierarchical_attention.holidays.calendar import (  # noqa: E402
    HOLIDAY_KEYS,
    days_from_holiday_features,
)

OUT = ROOT / "ab_runs" / "reclaim" / "year_scope_800"
HOL_DIR = OUT / "holiday_features_year"
JUBILANT = Path(
    os.environ.get(
        "DEEPSEQUENCE_DATA_DIR",
        "/Users/mritunjaykumar/Library/CloudStorage/"
        "GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
    )
)
SKU_LIST = ROOT / "ab_runs" / "recompare" / "sku_list_daily_data42.json"
PY = str(ROOT / ".venv-test" / "bin" / "python")
MODELS = "deepsequence,temporal_transformer,lightgbm"

PRIOR_MH = ROOT / "ab_runs" / "reclaim" / "multiseed" / "daily_s42_mh60.json"
PRIOR_H1 = ROOT / "ab_runs" / "reclaim" / "daily_h1_softsign_mono_mixer.json"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rebuild_us_year_holidays(*, force: bool = False) -> dict:
    HOL_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "train_split.csv",
        "val_split.csv",
        "test_split.csv",
        "split_metadata.csv",
    ):
        src = JUBILANT / name
        dst = HOL_DIR / name
        if dst.exists() or dst.is_symlink():
            if force:
                dst.unlink()
            else:
                continue
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)

    probe_hol = (
        ROOT
        / "paper_figures"
        / "year_scope_retest"
        / "daily_holiday_year_scope_us"
    )

    verify_splits = {}
    for split_csv, out_name in (
        ("train_split.csv", "holiday_features_train.csv"),
        ("val_split.csv", "holiday_features_val.csv"),
        ("test_split.csv", "holiday_features_test.csv"),
    ):
        out_path = HOL_DIR / out_name
        locked_path = JUBILANT / out_name
        probe_path = probe_hol / out_name
        if out_path.exists() and not force:
            print(f"keep existing {out_path}", flush=True)
        elif probe_path.exists() and not force:
            # Reuse prior year-scope rebuild (already verified == locked).
            try:
                if out_path.exists() or out_path.is_symlink():
                    out_path.unlink()
                out_path.hardlink_to(probe_path)
                print(f"hardlinked {out_name} from probe rebuild", flush=True)
            except OSError:
                shutil.copy2(probe_path, out_path)
                print(f"copied {out_name} from probe rebuild", flush=True)
        else:
            t0 = time.time()
            df = pd.read_csv(JUBILANT / split_csv, parse_dates=["ds"])
            hol = days_from_holiday_features(
                df["ds"],
                holiday_keys=HOLIDAY_KEYS,
                country="US",
                distance_scope="year",
            )
            hol.to_csv(out_path, index=False)
            print(
                f"wrote {out_name}: rows={len(hol)} cols={list(hol.columns)} "
                f"in {time.time()-t0:.1f}s",
                flush=True,
            )

        # Full-column compare vs locked + sample nearest contrast
        locked = pd.read_csv(locked_path)
        year = pd.read_csv(out_path)
        cols = [c for c in year.columns if c.startswith("days_from_")]
        locked_arr = locked[cols].to_numpy(dtype=np.float64)
        year_arr = year[cols].to_numpy(dtype=np.float64)
        max_abs_locked = float(np.max(np.abs(locked_arr - year_arr)))
        n_diff_locked = int(np.any(locked_arr != year_arr, axis=1).sum())

        # Nearest contrast on first 20k rows (full nearest rebuild is expensive)
        n_sample = min(20_000, len(year))
        ds = pd.read_csv(JUBILANT / split_csv, parse_dates=["ds"], nrows=n_sample)["ds"]
        near = days_from_holiday_features(
            ds, holiday_keys=HOLIDAY_KEYS, country="US", distance_scope="nearest"
        )
        year_s = year.iloc[:n_sample][cols].to_numpy(dtype=np.float64)
        near_arr = near[cols].to_numpy(dtype=np.float64)
        max_abs_near = float(np.max(np.abs(year_s - near_arr)))
        n_diff_near = int(np.any(year_s != near_arr, axis=1).sum())

        example = {
            "ds": str(ds.iloc[0]),
            "days_from_Christmas_year": float(year.iloc[0]["days_from_Christmas"]),
            "days_from_Christmas_locked": float(locked.iloc[0]["days_from_Christmas"]),
            "days_from_Christmas_nearest_sample": float(near.iloc[0]["days_from_Christmas"]),
        }
        verify_splits[out_name] = {
            "rows": int(len(year)),
            "sha256_year": _sha256(out_path),
            "sha256_locked": _sha256(locked_path),
            "max_abs_vs_locked": max_abs_locked,
            "n_rows_differ_vs_locked": n_diff_locked,
            "sample_n": n_sample,
            "max_abs_vs_nearest_sample": max_abs_near,
            "n_rows_differ_vs_nearest_sample": n_diff_near,
            "example_row0": example,
        }
        print(
            f"verify {out_name}: max_abs_vs_locked={max_abs_locked} "
            f"max_abs_vs_nearest_sample={max_abs_near}",
            flush=True,
        )

    meta = {
        "distance_scope": "year",
        "calendar": "US",
        "holiday_keys": list(HOLIDAY_KEYS),
        "jubilant_data_dir": str(JUBILANT),
        "year_scope_data_dir": str(HOL_DIR),
        "note": (
            "Apples-to-apples protocol fix vs locked US holiday key set. "
            "Country-aware calendars are optional (B) and not used here."
        ),
        "splits": verify_splits,
        "locked_already_year_scoped": all(
            v["max_abs_vs_locked"] == 0.0 for v in verify_splits.values()
        ),
    }
    (OUT / "holiday_verify.json").write_text(json.dumps(meta, indent=2))
    return meta


def _run(cmd: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env["XDG_CACHE_HOME"] = "/tmp/xdgcache"
    env["DEEPSEQUENCE_DATA_DIR"] = str(HOL_DIR)
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    print(f"    log → {log_path}", flush=True)
    with open(log_path, "w") as fh:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT
        )
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}); see {log_path}")


def run_h1() -> Path:
    out_json = OUT / "daily_h1_s42.json"
    log = OUT / "daily_h1_s42.log"
    cmd = [
        PY,
        "-u",
        "-m",
        "deepsequence_hierarchical_attention.eval.same_features_compare",
        "--data_dir",
        str(HOL_DIR),
        "--models",
        MODELS,
        "--max_skus",
        "800",
        "--sku_list",
        str(SKU_LIST),
        "--data_seed",
        "42",
        "--seed",
        "42",
        "--train_seed",
        "42",
        "--epochs",
        "25",
        "--use_cross_layers",
        "0",
        "--out_json",
        str(out_json),
    ]
    _run(cmd, log)
    return out_json


def run_mh() -> Path:
    out_json = OUT / "daily_mh60_s42.json"
    log = OUT / "daily_mh60_s42.log"
    cmd = [
        PY,
        "-u",
        "-m",
        "deepsequence_hierarchical_attention.eval.multihorizon_compare",
        "--data_dir",
        str(HOL_DIR),
        "--horizon",
        "60",
        "--report_horizons",
        "1,7,14,28,60",
        "--models",
        MODELS,
        "--max_skus",
        "800",
        "--sku_list",
        str(SKU_LIST),
        "--data_seed",
        "42",
        "--seed",
        "42",
        "--train_seed",
        "42",
        "--use_cross_layers",
        "0",
        "--out_json",
        str(out_json),
    ]
    _run(cmd, log)
    return out_json


def _iwmae_h1(path: Path) -> dict:
    d = json.loads(path.read_text())
    out = {}
    for m, block in d.get("models", {}).items():
        o = block.get("overall") or {}
        if "iwmae_rounded" in o:
            out[m] = float(o["iwmae_rounded"])
    return out


def _iwmae_mh(path: Path) -> dict:
    d = json.loads(path.read_text())
    out = {}
    for m, block in d.get("models", {}).items():
        bh = block.get("by_horizon") or {}
        out[m] = {
            str(h): float(bh[h]["overall"]["iwmae_rounded"])
            for h in bh
            if isinstance(bh[h], dict) and "overall" in bh[h]
        }
    return out


def write_comparison(h1_path: Path, mh_path: Path, verify: dict) -> Path:
    prior_mh = _iwmae_mh(PRIOR_MH) if PRIOR_MH.exists() else {}
    prior_h1 = _iwmae_h1(PRIOR_H1) if PRIOR_H1.exists() else {}
    new_h1 = _iwmae_h1(h1_path) if h1_path.exists() else {}
    new_mh = _iwmae_mh(mh_path) if mh_path.exists() else {}

    def delta_map(a: dict, b: dict) -> dict:
        keys = sorted(set(a) | set(b))
        return {
            k: {
                "year_scope": a.get(k),
                "prior": b.get(k),
                "delta_year_minus_prior": (
                    None
                    if a.get(k) is None or b.get(k) is None
                    else float(a[k]) - float(b[k])
                ),
            }
            for k in keys
        }

    mh_delta = {}
    for m in sorted(set(new_mh) | set(prior_mh)):
        mh_delta[m] = {}
        horizons = sorted(
            set(new_mh.get(m, {})) | set(prior_mh.get(m, {})),
            key=lambda x: int(x),
        )
        for h in horizons:
            yv = (new_mh.get(m) or {}).get(h)
            pv = (prior_mh.get(m) or {}).get(h)
            mh_delta[m][h] = {
                "year_scope": yv,
                "prior_multiseed_s42": pv,
                "delta_year_minus_prior": (
                    None if yv is None or pv is None else float(yv) - float(pv)
                ),
            }

    summary = {
        "protocol": {
            "n_skus": 800,
            "seed": 42,
            "sku_list": str(SKU_LIST),
            "holiday": "US days_from_* distance_scope=year",
            "stack": "softsign + mono + L1 attn + mixer on + cross off + additive",
            "locked_already_year_scoped": verify.get("locked_already_year_scoped"),
        },
        "h1_iwmae_rounded": {
            "year_scope": new_h1,
            "prior_softsign_mono_mixer": prior_h1,
            "delta": delta_map(new_h1, prior_h1),
        },
        "mh_iwmae_rounded": mh_delta,
        "prior_artifacts": {
            "h1": str(PRIOR_H1),
            "mh": str(PRIOR_MH),
        },
        "carparts_monthly_note": (
            "Locked monthly feature_config_monthly.yaml has holiday_encoding: none; "
            "Car Parts bake-off does not consume daily holiday_features_*.csv."
        ),
        "pending": [
            "multi-seed 43–46 year-scope retest (not run in this pass)",
            "optional country+year Jubilant rebuild (protocol change vs locked US)",
        ],
    }
    out = OUT / "comparison_vs_prior.json"
    out.write_text(json.dumps(summary, indent=2))
    return out


def write_readme(verify: dict) -> None:
    text = f"""# Year-scope locked 800 retest (seed 42)

## Holiday rebuild (A — US, apples-to-apples)
- Source splits: `{JUBILANT}`
- Regenerated dir: `{HOL_DIR}` (gitignored `*.csv`)
- `distance_scope='year'`, calendar=`US`, keys=`HOLIDAY_KEYS`
- Locked jubilant CSVs already year-scoped: **{verify.get('locked_already_year_scoped')}**
  (see `holiday_verify.json`; max abs vs locked = 0 on all splits).
- Explicit `nearest` differs (sample max abs ≫ 0) — so the bug class is real,
  but locked bake-off assets were already on the year path.

## Eval outputs
- `daily_h1_s42.json` — DS / TST / LightGBM one-step
- `daily_mh60_s42.json` — recursive report horizons 1/7/14/28/60
- `comparison_vs_prior.json` — deltas vs reclaim softsign H1 + multiseed s42 MH

## Monthly / Car Parts
Locked monthly holidays are OFF (`feature_config_monthly.yaml`); no day-distance
CSV path — monthly bake-off unchanged by this rebuild.

## Pending
- Seeds 43–46
- Optional (B) country+year Jubilant features (would change protocol vs published US lock)
"""
    (OUT / "README.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-holidays", action="store_true")
    ap.add_argument("--skip-holidays", action="store_true")
    ap.add_argument("--skip-h1", action="store_true")
    ap.add_argument("--skip-mh", action="store_true")
    ap.add_argument("--only-compare", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.only_compare:
        verify = json.loads((OUT / "holiday_verify.json").read_text())
        write_comparison(OUT / "daily_h1_s42.json", OUT / "daily_mh60_s42.json", verify)
        write_readme(verify)
        print("comparison only done", flush=True)
        return

    if args.skip_holidays:
        verify = json.loads((OUT / "holiday_verify.json").read_text())
    else:
        print("=== rebuild US year-scope holidays ===", flush=True)
        verify = rebuild_us_year_holidays(force=args.force_holidays)

    write_readme(verify)

    h1_path = OUT / "daily_h1_s42.json"
    mh_path = OUT / "daily_mh60_s42.json"
    if not args.skip_h1:
        print("=== H=1 bake-off (800, seed 42) ===", flush=True)
        h1_path = run_h1()
    if not args.skip_mh:
        print("=== MH bake-off H=60 report 1/7/14/28/60 ===", flush=True)
        mh_path = run_mh()

    cmp_path = write_comparison(h1_path, mh_path, verify)
    print(f"wrote {cmp_path}", flush=True)
    print(json.dumps(json.loads(cmp_path.read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
