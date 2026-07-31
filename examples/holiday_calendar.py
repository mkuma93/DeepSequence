"""US public-holiday calendar helpers (shared by daily distance + monthly month-has)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

# Keys used in feature names: month_has_{Key} / days_from_{Key}
HOLIDAY_KEYS: Sequence[str] = (
    "NewYear",
    "MLK",
    "Presidents",
    "Valentine",
    "Easter",
    "Mothers",
    "Memorial",
    "Fathers",
    "July4",
    "Labor",
    "Halloween",
    "Thanksgiving",
    "BlackFriday",
    "Christmas",
    "NewYearEve",
)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """weekday: Mon=0 .. Sun=6; n: 1=first, -1=last."""
    if n > 0:
        d = pd.Timestamp(year=year, month=month, day=1)
        shift = (weekday - d.dayofweek) % 7
        return d + pd.Timedelta(days=shift + 7 * (n - 1))
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    shift = (d.dayofweek - weekday) % 7
    return d - pd.Timedelta(days=shift)


def easter(year: int) -> pd.Timestamp:
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


def holiday_dates_for_year(year: int) -> Dict[str, pd.Timestamp]:
    thanks = nth_weekday(year, 11, 3, 4)
    return {
        "NewYear": pd.Timestamp(year, 1, 1),
        "MLK": nth_weekday(year, 1, 0, 3),
        "Presidents": nth_weekday(year, 2, 0, 3),
        "Valentine": pd.Timestamp(year, 2, 14),
        "Easter": easter(year),
        "Mothers": nth_weekday(year, 5, 6, 2),
        "Memorial": nth_weekday(year, 5, 0, -1),
        "Fathers": nth_weekday(year, 6, 6, 3),
        "July4": pd.Timestamp(year, 7, 4),
        "Labor": nth_weekday(year, 9, 0, 1),
        "Halloween": pd.Timestamp(year, 10, 31),
        "Thanksgiving": thanks,
        "BlackFriday": thanks + pd.Timedelta(days=1),
        "Christmas": pd.Timestamp(year, 12, 25),
        "NewYearEve": pd.Timestamp(year, 12, 31),
    }


def holiday_calendar(years: Iterable[int]) -> Dict[str, np.ndarray]:
    """Map holiday key → array of datetime64 event dates across years."""
    cal = {k: [] for k in HOLIDAY_KEYS}
    for y in years:
        for k, d in holiday_dates_for_year(int(y)).items():
            cal[k].append(d)
    return {k: np.asarray(v, dtype="datetime64[ns]") for k, v in cal.items()}


def month_has_holiday_features(
    dates: pd.Series,
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
) -> pd.DataFrame:
    """
    For each calendar month in ``dates``, set month_has_{Key}=1 if that holiday
    falls in the same year-month (else 0). Monthly encoding of which holidays
    belong to the observation month.
    """
    ds = pd.to_datetime(dates)
    years = range(int(ds.dt.year.min()) - 1, int(ds.dt.year.max()) + 2)
    # Build set of (year, month) per holiday
    ym_sets = {k: set() for k in holiday_keys}
    for y in years:
        for k, d in holiday_dates_for_year(y).items():
            if k in ym_sets:
                ym_sets[k].add((d.year, d.month))

    y = ds.dt.year.to_numpy()
    m = ds.dt.month.to_numpy()
    out = {}
    for k in holiday_keys:
        flags = np.fromiter(
            ((int(yy), int(mm)) in ym_sets[k] for yy, mm in zip(y, m)),
            dtype=np.float32,
            count=len(ds),
        )
        out[f"month_has_{k}"] = flags
    return pd.DataFrame(out)


def _month_index_arr(years: np.ndarray, months: np.ndarray) -> np.ndarray:
    return years.astype(np.int32) * 12 + months.astype(np.int32)


def months_from_holiday_features(
    dates: pd.Series,
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
) -> pd.DataFrame:
    """
    Signed months from the observation month to the nearest holiday month.

    Analogous to days_from_* on daily panels, but distance is in calendar
    months (holiday month membership), not days.
    """
    ds = pd.to_datetime(dates)
    years = range(int(ds.dt.year.min()) - 1, int(ds.dt.year.max()) + 2)
    # Holiday events as month indices
    event_mi = {k: [] for k in holiday_keys}
    for y in years:
        for k, d in holiday_dates_for_year(y).items():
            if k in event_mi:
                event_mi[k].append(d.year * 12 + d.month)
    for k in event_mi:
        event_mi[k] = np.asarray(event_mi[k], dtype=np.int32)

    obs = _month_index_arr(ds.dt.year.to_numpy(), ds.dt.month.to_numpy())
    out = {}
    for k in holiday_keys:
        ev = event_mi[k]
        delta = obs[:, None] - ev[None, :]
        idx = np.argmin(np.abs(delta), axis=1)
        out[f"months_from_{k}"] = delta[np.arange(len(obs)), idx].astype(np.float32)
    return pd.DataFrame(out)


def days_from_holiday_features(
    dates: pd.Series,
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
) -> pd.DataFrame:
    """Signed days to nearest occurrence of each holiday (daily panels)."""
    ds = pd.to_datetime(dates)
    years = range(int(ds.dt.year.min()) - 1, int(ds.dt.year.max()) + 2)
    cal = holiday_calendar(years)
    d = ds.to_numpy(dtype="datetime64[ns]")
    out = {}
    for k in holiday_keys:
        events = cal[k]
        delta = (d[:, None] - events[None, :]).astype("timedelta64[D]").astype(np.int32)
        idx = np.argmin(np.abs(delta), axis=1)
        out[f"days_from_{k}"] = delta[np.arange(len(d)), idx].astype(np.float32)
    return pd.DataFrame(out)
