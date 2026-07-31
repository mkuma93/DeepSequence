#!/usr/bin/env python3
"""
Prepare Monash Car Parts (intermittent monthly) as a DeepSequence panel.

Source: https://zenodo.org/records/4656021 (zeros for missing).
Outputs train/val/test_split.csv + holiday_features_*.csv under --out_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TSF = ROOT / "public_data/car_parts/car_parts_dataset_without_missing_values.tsf"
DEFAULT_OUT = ROOT / "public_data/car_parts/panel"

HOLIDAY_NAMES = [
    "days_from_NewYear",
    "days_from_MLK",
    "days_from_Presidents",
    "days_from_Valentine",
    "days_from_Easter",
    "days_from_Mothers",
    "days_from_Memorial",
    "days_from_Fathers",
    "days_from_July4",
    "days_from_Labor",
    "days_from_Halloween",
    "days_from_Thanksgiving",
    "days_from_BlackFriday",
    "days_from_Christmas",
    "days_from_NewYearEve",
]


def parse_tsf(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            # T1:1998-01-01 00-00-00:0,0,...
            name, rest = line.split(":", 1)
            stamp, vals = rest.split(":", 1)
            start = pd.Timestamp(stamp.replace(" 00-00-00", ""))
            y = np.asarray([float(x) for x in vals.split(",")], dtype=np.float32)
            dates = pd.date_range(start, periods=len(y), freq="MS")
            for d, q in zip(dates, y):
                rows.append({"id_var": name, "ds": d, "Quantity": float(q)})
    return pd.DataFrame(rows)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """weekday: Mon=0 .. Sun=6; n: 1=first, -1=last."""
    if n > 0:
        d = pd.Timestamp(year=year, month=month, day=1)
        shift = (weekday - d.dayofweek) % 7
        d = d + pd.Timedelta(days=shift + 7 * (n - 1))
        return d
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    shift = (d.dayofweek - weekday) % 7
    return d - pd.Timedelta(days=shift)


def _easter(year: int) -> pd.Timestamp:
    # Anonymous Gregorian algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def holiday_calendar(years) -> dict:
    cal = {name: [] for name in HOLIDAY_NAMES}
    for y in years:
        cal["days_from_NewYear"].append(pd.Timestamp(y, 1, 1))
        cal["days_from_MLK"].append(_nth_weekday(y, 1, 0, 3))
        cal["days_from_Presidents"].append(_nth_weekday(y, 2, 0, 3))
        cal["days_from_Valentine"].append(pd.Timestamp(y, 2, 14))
        cal["days_from_Easter"].append(_easter(y))
        cal["days_from_Mothers"].append(_nth_weekday(y, 5, 6, 2))
        cal["days_from_Memorial"].append(_nth_weekday(y, 5, 0, -1))
        cal["days_from_Fathers"].append(_nth_weekday(y, 6, 6, 3))
        cal["days_from_July4"].append(pd.Timestamp(y, 7, 4))
        cal["days_from_Labor"].append(_nth_weekday(y, 9, 0, 1))
        cal["days_from_Halloween"].append(pd.Timestamp(y, 10, 31))
        thanks = _nth_weekday(y, 11, 3, 4)
        cal["days_from_Thanksgiving"].append(thanks)
        cal["days_from_BlackFriday"].append(thanks + pd.Timedelta(days=1))
        cal["days_from_Christmas"].append(pd.Timestamp(y, 12, 25))
        cal["days_from_NewYearEve"].append(pd.Timestamp(y, 12, 31))
    return {k: np.asarray(v, dtype="datetime64[ns]") for k, v in cal.items()}


def holiday_distances(dates: pd.Series, cal: dict) -> pd.DataFrame:
    out = {}
    d = dates.to_numpy(dtype="datetime64[ns]")
    for name, events in cal.items():
        # signed days to nearest event
        delta = (d[:, None] - events[None, :]).astype("timedelta64[D]").astype(np.int32)
        idx = np.argmin(np.abs(delta), axis=1)
        out[name] = delta[np.arange(len(d)), idx].astype(np.float32)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsf", type=Path, default=DEFAULT_TSF)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--val_months", type=int, default=6)
    p.add_argument("--test_months", type=int, default=6)
    p.add_argument("--min_train_nonzero", type=int, default=2)
    args = p.parse_args()

    print(f"Parsing {args.tsf} ...")
    panel = parse_tsf(args.tsf)
    panel["ds"] = pd.to_datetime(panel["ds"])
    n_series = panel["id_var"].nunique()
    n_steps = panel.groupby("id_var").size().iloc[0]
    print(f"series={n_series} steps={n_steps} zero_rate={ (panel.Quantity==0).mean():.3f}")

    # Chronological split on absolute dates (all series aligned)
    dates = np.sort(panel["ds"].unique())
    test_cut = dates[-args.test_months]
    val_cut = dates[-(args.test_months + args.val_months)]
    train = panel[panel.ds < val_cut].copy()
    val = panel[(panel.ds >= val_cut) & (panel.ds < test_cut)].copy()
    test = panel[panel.ds >= test_cut].copy()

    # Drop cold / all-zero train series
    nz = train.groupby("id_var")["Quantity"].apply(lambda s: (s > 0).sum())
    keep = set(nz[nz >= args.min_train_nonzero].index.astype(str))
    train = train[train.id_var.astype(str).isin(keep)]
    val = val[val.id_var.astype(str).isin(keep)]
    test = test[test.id_var.astype(str).isin(keep)]
    print(f"kept_series={len(keep)} train/val/test rows={len(train)}/{len(val)}/{len(test)}")

    years = range(int(panel.ds.dt.year.min()) - 1, int(panel.ds.dt.year.max()) + 2)
    cal = holiday_calendar(years)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("train", train), ("val", val), ("test", test)):
        df = df.sort_values(["id_var", "ds"], kind="mergesort").reset_index(drop=True)
        hol = holiday_distances(df["ds"], cal)
        df.to_csv(args.out_dir / f"{name}_split.csv", index=False)
        hol.to_csv(args.out_dir / f"holiday_features_{name}.csv", index=False)
        print(f"wrote {name}: {len(df)} rows")

    import json as _json

    meta = {
        "source": "Monash Car Parts (Zenodo 4656021)",
        "frequency": "monthly",
        "n_series_kept": len(keep),
        "val_months": args.val_months,
        "test_months": args.test_months,
        "zero_rate_all": float((panel.Quantity == 0).mean()),
    }
    (args.out_dir / "meta.json").write_text(_json.dumps(meta, indent=2))
    print("done", args.out_dir)


if __name__ == "__main__":
    main()
