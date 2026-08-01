#!/usr/bin/env python3
"""Aggregate Jubilant-style daily panels to weekly (ISO Monday-start).

Week rule
---------
- ``ds`` = Monday of the ISO week that contains each daily observation
  (``Timestamp.to_period('W-SUN').start_time`` ≡ Monday-start ISO week).
- Demand: ``Quantity`` summed over days in the week, **same** ``id_var`` (SKU).
- Country: parsed from ``id_var`` prefix via ``holiday_calendar.country_from_sku_id``
  and written to ``meta.json`` / optional ``country`` column (grouping helper).

Outputs (under ``--out_dir``):
  train_split.csv, val_split.csv, test_split.csv
  holiday_features_{train,val,test}.csv  (days_from_* at week Monday)
  meta.json

Does **not** touch the locked daily bake-off. Smoke artifacts also land under
``ab_runs/weekly/`` when using the default smoke path.

Example (locked 800 SKUs)::

  TF_USE_LEGACY_KERAS=1 .venv-test/bin/python examples/prepare_weekly_panel.py \\
    --data_dir "$DEEPSEQUENCE_DATA_DIR" \\
    --sku_list ab_runs/recompare/sku_list_daily_data42.json \\
    --out_dir ab_runs/weekly/panel_locked800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from holiday_calendar import (  # noqa: E402
    HOLIDAY_KEYS,
    days_from_holiday_features,
    days_from_holiday_features_by_country,
    country_from_sku_id,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_dir",
        default=None,
        help="Daily panel dir with train/val/test_split.csv (+ optional holidays).",
    )
    p.add_argument(
        "--out_dir",
        default=str(ROOT / "ab_runs/weekly/panel"),
        help="Weekly panel output directory.",
    )
    p.add_argument(
        "--sku_list",
        default=None,
        help="Optional JSON array / line list of id_var to keep (e.g. locked 800).",
    )
    p.add_argument(
        "--max_skus",
        type=int,
        default=None,
        help="Optional cap after sku_list / universe (deterministic first N).",
    )
    p.add_argument(
        "--holiday_calendar",
        default="from_sku",
        choices=("from_sku", "US"),
        help="Country calendar for days_from_*; from_sku uses id_var prefix.",
    )
    p.add_argument(
        "--val_weeks",
        type=int,
        default=8,
        help="Trailing weeks of (train∪val history) held out as val before test.",
    )
    p.add_argument(
        "--test_weeks",
        type=int,
        default=12,
        help="Trailing weeks held out as test.",
    )
    return p.parse_args()


def _load_sku_list(path: str | None) -> list[str] | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return [str(x) for x in json.loads(text)]
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _read_daily_splits(data_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}_split.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        df = pd.read_csv(path)
        df["split_src"] = split
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["ds"] = pd.to_datetime(panel["ds"])
    panel["id_var"] = panel["id_var"].astype(str)
    panel["Quantity"] = panel["Quantity"].astype(float)
    return panel


def daily_to_weekly(panel: pd.DataFrame) -> pd.DataFrame:
    """Sum Quantity by (id_var, ISO Monday week)."""
    df = panel.copy()
    # W-SUN period: week labeled by Sunday end → start_time is Monday.
    week_start = df["ds"].dt.to_period("W-SUN").dt.start_time
    df["ds"] = pd.to_datetime(week_start).dt.normalize()
    weekly = (
        df.groupby(["id_var", "ds"], as_index=False, sort=True)["Quantity"]
        .sum()
        .sort_values(["id_var", "ds"], kind="mergesort")
        .reset_index(drop=True)
    )
    weekly["country"] = weekly["id_var"].map(
        lambda s: country_from_sku_id(s, default="US")
    )
    return weekly


def _time_split(
    weekly: pd.DataFrame, *, val_weeks: int, test_weeks: int
) -> dict[str, pd.DataFrame]:
    """Global calendar split on unique week Mondays (same cut for all SKUs)."""
    weeks = np.sort(weekly["ds"].unique())
    if len(weeks) < val_weeks + test_weeks + 4:
        raise SystemExit(
            f"Too few weeks ({len(weeks)}) for val={val_weeks} test={test_weeks}"
        )
    test_cut = weeks[-(test_weeks)]
    val_cut = weeks[-(test_weeks + val_weeks)]
    train = weekly[weekly["ds"] < val_cut].copy()
    val = weekly[(weekly["ds"] >= val_cut) & (weekly["ds"] < test_cut)].copy()
    test = weekly[weekly["ds"] >= test_cut].copy()
    return {"train": train, "val": val, "test": test}


def _holiday_for_split(
    split_df: pd.DataFrame, *, calendar_mode: str
) -> pd.DataFrame:
    dates = pd.to_datetime(split_df["ds"])
    if calendar_mode == "US":
        hol = days_from_holiday_features(
            dates, holiday_keys=HOLIDAY_KEYS, country="US", distance_scope="year"
        )
    else:
        countries = [country_from_sku_id(s, default="US") for s in split_df["id_var"]]
        hol = days_from_holiday_features_by_country(
            dates,
            countries,
            holiday_keys=HOLIDAY_KEYS,
            default_country="US",
            distance_scope="year",
        )
    out = hol.copy()
    out.insert(0, "id_var", split_df["id_var"].to_numpy())
    out.insert(1, "ds", pd.to_datetime(split_df["ds"]).to_numpy())
    return out


def main():
    args = parse_args()
    data_dir = Path(
        args.data_dir
        or Path(
            "/Users/mritunjaykumar/Library/CloudStorage/"
            "GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data"
        )
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = _read_daily_splits(data_dir)
    skus = _load_sku_list(args.sku_list)
    if skus is not None:
        keep = set(skus)
        panel = panel[panel["id_var"].isin(keep)].copy()
    if args.max_skus is not None:
        universe = sorted(panel["id_var"].unique())
        keep = set(universe[: int(args.max_skus)])
        panel = panel[panel["id_var"].isin(keep)].copy()

    weekly = daily_to_weekly(panel)
    splits = _time_split(
        weekly, val_weeks=int(args.val_weeks), test_weeks=int(args.test_weeks)
    )

    meta = {
        "week_rule": "ISO",
        "week_start": "monday",
        "period_alias": "W-SUN → start_time (Monday)",
        "aggregation": "sum Quantity by (id_var, week Monday)",
        "country_from": "id_var prefix via holiday_calendar.country_from_sku_id",
        "holiday_calendar": args.holiday_calendar,
        "n_skus": int(weekly["id_var"].nunique()),
        "n_weeks_total": int(weekly["ds"].nunique()),
        "country_counts": weekly.groupby("country")["id_var"]
        .nunique()
        .astype(int)
        .to_dict(),
        "split_weeks": {
            k: {
                "n_rows": int(len(v)),
                "n_skus": int(v["id_var"].nunique()),
                "ds_min": str(v["ds"].min().date()) if len(v) else None,
                "ds_max": str(v["ds"].max().date()) if len(v) else None,
            }
            for k, v in splits.items()
        },
        "source_daily_dir": str(data_dir),
        "sku_list": args.sku_list,
        "feature_config": "feature_config_weekly.yaml",
    }

    for name, df in splits.items():
        cols = ["id_var", "ds", "Quantity"]
        df[cols].to_csv(out_dir / f"{name}_split.csv", index=False)
        hol = _holiday_for_split(df, calendar_mode=args.holiday_calendar)
        hol.to_csv(out_dir / f"holiday_features_{name}.csv", index=False)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote weekly panel under {out_dir}")


if __name__ == "__main__":
    main()
