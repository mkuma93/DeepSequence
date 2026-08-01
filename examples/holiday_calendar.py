"""Multi-country holiday calendar helpers (shared by daily distance + monthly month-has).

Feature schema uses a **shared** key set (``HOLIDAY_KEYS``). Each country fills
those slots with local public / retail dates where they apply; keys with no
local analogue get a large sentinel distance so ``is_*`` binaries stay off.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Keys used in feature names: month_has_{Key} / days_from_{Key} / is_{Key}
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

# Distance when a shared key has no local holiday (keeps is_* off).
NA_DISTANCE_DAYS: float = 9999.0

# Canonical codes used internally.
SUPPORTED_COUNTRY_CODES: Sequence[str] = (
    "US",
    "UK",
    "AU",
    "IE",
    "FR",
    "DE",
    "NL",
    "EU",  # generic Western/EU retail fallback
)

# Jubilant / Online Retail id_var prefixes → canonical code.
COUNTRY_ALIASES: Dict[str, str] = {
    "US": "US",
    "USA": "US",
    "United States": "US",
    "United States of America": "US",
    "UK": "UK",
    "United Kingdom": "UK",
    "GB": "UK",
    "Great Britain": "UK",
    "England": "UK",
    "AU": "AU",
    "Australia": "AU",
    "IE": "IE",
    "EIRE": "IE",
    "Ireland": "IE",
    "FR": "FR",
    "France": "FR",
    "DE": "DE",
    "Germany": "DE",
    "NL": "NL",
    "Netherlands": "NL",
    # Smaller locked-panel countries → EU retail fallback
    "Belgium": "EU",
    "Spain": "EU",
    "Switzerland": "EU",
    "Portugal": "EU",
    "Sweden": "EU",
    "Denmark": "EU",
    "Finland": "EU",
    "Poland": "EU",
    "Italy": "EU",
    "Austria": "EU",
    "Norway": "EU",
    "Cyprus": "EU",
    "Channel Islands": "UK",
}


def normalize_country(country: Optional[str], default: str = "US") -> str:
    """Map free-text / SKU prefix to a supported calendar code."""
    if country is None or (isinstance(country, float) and np.isnan(country)):
        return default
    raw = str(country).strip()
    if not raw:
        return default
    if raw in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[raw]
    # case-insensitive alias lookup
    lower_map = {k.lower(): v for k, v in COUNTRY_ALIASES.items()}
    hit = lower_map.get(raw.lower())
    if hit:
        return hit
    return default


def country_from_sku_id(sku_id: Union[str, object], default: str = "US") -> str:
    """Parse country from ``{Country}_{stock_code}`` id_var prefix."""
    s = str(sku_id)
    prefix = s.split("_", 1)[0] if "_" in s else s
    return normalize_country(prefix, default=default)


def countries_from_sku_ids(
    sku_ids: Iterable,
    default: str = "US",
) -> np.ndarray:
    return np.asarray(
        [country_from_sku_id(s, default=default) for s in sku_ids],
        dtype=object,
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


def _observed_weekend(d: pd.Timestamp) -> pd.Timestamp:
    """UK/AU-style: Saturday→Mon, Sunday→Mon."""
    if d.dayofweek == 5:
        return d + pd.Timedelta(days=2)
    if d.dayofweek == 6:
        return d + pd.Timedelta(days=1)
    return d


def _mothering_sunday(year: int) -> pd.Timestamp:
    """UK/IE Mothering Sunday = fourth Sunday of Lent = Easter − 3 weeks."""
    return easter(year) - pd.Timedelta(days=21)


def _us_thanksgiving(year: int) -> pd.Timestamp:
    return nth_weekday(year, 11, 3, 4)


def _black_friday_retail(year: int) -> pd.Timestamp:
    """US Black Friday; also used as shared retail event elsewhere."""
    return _us_thanksgiving(year) + pd.Timedelta(days=1)


def _kings_day_nl(year: int) -> pd.Timestamp:
    """Netherlands King's Day: 27 Apr, or 26 Apr if Sunday."""
    d = pd.Timestamp(year, 4, 27)
    if d.dayofweek == 6:
        return pd.Timestamp(year, 4, 26)
    return d


def holiday_dates_for_year(
    year: int,
    country: str = "US",
) -> Dict[str, Optional[pd.Timestamp]]:
    """
    Return shared-key → date for ``year`` under ``country`` calendar.

    Missing local analogues map to ``None`` (caller uses ``NA_DISTANCE_DAYS``).
    """
    code = normalize_country(country)
    year = int(year)
    eas = easter(year)
    bf = _black_friday_retail(year)

    # Shared retail / Christian anchors (most Western markets).
    shared = {
        "Valentine": pd.Timestamp(year, 2, 14),
        "Easter": eas,
        "Halloween": pd.Timestamp(year, 10, 31),
        "Christmas": pd.Timestamp(year, 12, 25),
        "NewYearEve": pd.Timestamp(year, 12, 31),
        "BlackFriday": bf,
    }

    if code == "US":
        thanks = _us_thanksgiving(year)
        return {
            "NewYear": pd.Timestamp(year, 1, 1),
            "MLK": nth_weekday(year, 1, 0, 3),
            "Presidents": nth_weekday(year, 2, 0, 3),
            "Valentine": shared["Valentine"],
            "Easter": eas,
            "Mothers": nth_weekday(year, 5, 6, 2),
            "Memorial": nth_weekday(year, 5, 0, -1),
            "Fathers": nth_weekday(year, 6, 6, 3),
            "July4": pd.Timestamp(year, 7, 4),
            "Labor": nth_weekday(year, 9, 0, 1),
            "Halloween": shared["Halloween"],
            "Thanksgiving": thanks,
            "BlackFriday": thanks + pd.Timedelta(days=1),
            "Christmas": shared["Christmas"],
            "NewYearEve": shared["NewYearEve"],
        }

    if code == "UK":
        # England & Wales bank holidays + shared retail.
        return {
            "NewYear": _observed_weekend(pd.Timestamp(year, 1, 1)),
            "MLK": nth_weekday(year, 8, 0, -1),  # Summer bank holiday (last Mon Aug)
            "Presidents": None,
            "Valentine": shared["Valentine"],
            "Easter": eas,
            "Mothers": _mothering_sunday(year),
            "Memorial": nth_weekday(year, 5, 0, -1),  # Spring BH (last Mon May)
            "Fathers": nth_weekday(year, 6, 6, 3),
            "July4": None,
            "Labor": nth_weekday(year, 5, 0, 1),  # Early May BH
            "Halloween": shared["Halloween"],
            "Thanksgiving": None,
            "BlackFriday": bf,
            "Christmas": shared["Christmas"],
            "NewYearEve": shared["NewYearEve"],
        }

    if code == "AU":
        return {
            "NewYear": _observed_weekend(pd.Timestamp(year, 1, 1)),
            "MLK": _observed_weekend(pd.Timestamp(year, 1, 26)),  # Australia Day
            "Presidents": None,
            "Valentine": shared["Valentine"],
            "Easter": eas,
            "Mothers": nth_weekday(year, 5, 6, 2),
            "Memorial": _observed_weekend(pd.Timestamp(year, 4, 25)),  # ANZAC
            "Fathers": nth_weekday(year, 9, 6, 1),  # 1st Sun Sep
            "July4": nth_weekday(year, 6, 0, 2),  # King's Birthday (most states)
            "Labor": nth_weekday(year, 10, 0, 1),  # Labour Day NSW/ACT/SA/Vic
            "Halloween": shared["Halloween"],
            "Thanksgiving": None,
            "BlackFriday": bf,
            "Christmas": shared["Christmas"],
            "NewYearEve": shared["NewYearEve"],
        }

    if code == "IE":
        return {
            "NewYear": _observed_weekend(pd.Timestamp(year, 1, 1)),
            "MLK": _observed_weekend(pd.Timestamp(year, 3, 17)),  # St Patrick's
            "Presidents": nth_weekday(year, 8, 0, 1),  # August BH
            "Valentine": shared["Valentine"],
            "Easter": eas,
            "Mothers": _mothering_sunday(year),
            "Memorial": nth_weekday(year, 6, 0, 1),  # June BH
            "Fathers": nth_weekday(year, 6, 6, 3),
            "July4": None,
            "Labor": nth_weekday(year, 5, 0, 1),  # May BH
            "Halloween": shared["Halloween"],
            "Thanksgiving": None,
            "BlackFriday": bf,
            "Christmas": shared["Christmas"],
            "NewYearEve": shared["NewYearEve"],
        }

    if code == "FR":
        return {
            "NewYear": pd.Timestamp(year, 1, 1),
            "MLK": None,
            "Presidents": None,
            "Valentine": shared["Valentine"],
            "Easter": eas,
            "Mothers": nth_weekday(year, 5, 6, -1),  # last Sun May
            "Memorial": eas + pd.Timedelta(days=39),  # Ascension
            "Fathers": nth_weekday(year, 6, 6, 3),
            "July4": pd.Timestamp(year, 7, 14),  # Bastille
            "Labor": pd.Timestamp(year, 5, 1),
            "Halloween": shared["Halloween"],
            "Thanksgiving": None,
            "BlackFriday": bf,
            "Christmas": shared["Christmas"],
            "NewYearEve": shared["NewYearEve"],
        }

    if code == "DE":
        return {
            "NewYear": pd.Timestamp(year, 1, 1),
            "MLK": None,
            "Presidents": None,
            "Valentine": shared["Valentine"],
            "Easter": eas,
            "Mothers": nth_weekday(year, 5, 6, 2),
            "Memorial": eas + pd.Timedelta(days=39),  # Christi Himmelfahrt / Vatertag eve
            "Fathers": eas + pd.Timedelta(days=39),  # Vatertag = Ascension
            "July4": pd.Timestamp(year, 10, 3),  # German Unity Day
            "Labor": pd.Timestamp(year, 5, 1),
            "Halloween": shared["Halloween"],
            "Thanksgiving": None,
            "BlackFriday": bf,
            "Christmas": shared["Christmas"],
            "NewYearEve": shared["NewYearEve"],
        }

    if code == "NL":
        return {
            "NewYear": pd.Timestamp(year, 1, 1),
            "MLK": None,
            "Presidents": None,
            "Valentine": shared["Valentine"],
            "Easter": eas,
            "Mothers": nth_weekday(year, 5, 6, 2),
            "Memorial": pd.Timestamp(year, 5, 5),  # Liberation Day
            "Fathers": nth_weekday(year, 6, 6, 3),
            "July4": _kings_day_nl(year),
            "Labor": pd.Timestamp(year, 5, 1),
            "Halloween": shared["Halloween"],
            "Thanksgiving": None,
            "BlackFriday": bf,
            "Christmas": shared["Christmas"],
            "NewYearEve": shared["NewYearEve"],
        }

    # EU / generic Western retail fallback
    return {
        "NewYear": pd.Timestamp(year, 1, 1),
        "MLK": None,
        "Presidents": None,
        "Valentine": shared["Valentine"],
        "Easter": eas,
        "Mothers": nth_weekday(year, 5, 6, 2),
        "Memorial": eas + pd.Timedelta(days=39),
        "Fathers": nth_weekday(year, 6, 6, 3),
        "July4": None,
        "Labor": pd.Timestamp(year, 5, 1),
        "Halloween": shared["Halloween"],
        "Thanksgiving": None,
        "BlackFriday": bf,
        "Christmas": shared["Christmas"],
        "NewYearEve": shared["NewYearEve"],
    }


def holiday_calendar(
    years: Iterable[int],
    country: str = "US",
) -> Dict[str, np.ndarray]:
    """Map holiday key → array of datetime64 event dates across years."""
    cal: Dict[str, List[pd.Timestamp]] = {k: [] for k in HOLIDAY_KEYS}
    for y in years:
        for k, d in holiday_dates_for_year(int(y), country=country).items():
            if d is not None and k in cal:
                cal[k].append(d)
    return {k: np.asarray(v, dtype="datetime64[ns]") for k, v in cal.items()}


def month_has_holiday_features(
    dates: pd.Series,
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
    country: str = "US",
) -> pd.DataFrame:
    """
    For each calendar month in ``dates``, set month_has_{Key}=1 if that holiday
    falls in the same year-month (else 0). Monthly encoding of which holidays
    belong to the observation month.
    """
    ds = pd.to_datetime(dates)
    years = range(int(ds.dt.year.min()) - 1, int(ds.dt.year.max()) + 2)
    ym_sets = {k: set() for k in holiday_keys}
    for y in years:
        for k, d in holiday_dates_for_year(y, country=country).items():
            if k in ym_sets and d is not None:
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
    country: str = "US",
) -> pd.DataFrame:
    """
    Signed months from the observation month to the nearest holiday month.

    Analogous to days_from_* on daily panels, but distance is in calendar
    months (holiday month membership), not days.
    """
    ds = pd.to_datetime(dates)
    years = range(int(ds.dt.year.min()) - 1, int(ds.dt.year.max()) + 2)
    event_mi = {k: [] for k in holiday_keys}
    for y in years:
        for k, d in holiday_dates_for_year(y, country=country).items():
            if k in event_mi and d is not None:
                event_mi[k].append(d.year * 12 + d.month)
    for k in event_mi:
        event_mi[k] = np.asarray(event_mi[k], dtype=np.int32)

    obs = _month_index_arr(ds.dt.year.to_numpy(), ds.dt.month.to_numpy())
    out = {}
    for k in holiday_keys:
        ev = event_mi[k]
        if len(ev) == 0:
            out[f"months_from_{k}"] = np.full(len(obs), NA_DISTANCE_DAYS, dtype=np.float32)
            continue
        delta = obs[:, None] - ev[None, :]
        idx = np.argmin(np.abs(delta), axis=1)
        out[f"months_from_{k}"] = delta[np.arange(len(obs)), idx].astype(np.float32)
    return pd.DataFrame(out)


def month_has_holiday_features_by_country(
    dates: pd.Series,
    countries: Sequence[str],
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
    default_country: str = "US",
) -> pd.DataFrame:
    """
    Row-wise country calendars with a unified ``month_has_*`` schema.

    N/A local holidays stay 0 for that country/month.
    """
    ds = pd.to_datetime(dates).reset_index(drop=True)
    if len(ds) != len(countries):
        raise ValueError(
            f"dates length {len(ds)} != countries length {len(countries)}"
        )
    codes = np.asarray(
        [normalize_country(c, default=default_country) for c in countries],
        dtype=object,
    )
    out = {
        f"month_has_{k}": np.zeros(len(ds), dtype=np.float32) for k in holiday_keys
    }
    for code in sorted(set(codes.tolist())):
        mask = codes == code
        if not mask.any():
            continue
        built = month_has_holiday_features(
            ds.loc[mask],
            holiday_keys=holiday_keys,
            country=code,
        )
        for k in holiday_keys:
            col = f"month_has_{k}"
            out[col][mask] = built[col].to_numpy(dtype=np.float32)
    return pd.DataFrame(out)


def months_from_holiday_features_by_country(
    dates: pd.Series,
    countries: Sequence[str],
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
    default_country: str = "US",
) -> pd.DataFrame:
    """Row-wise country calendars with a unified ``months_from_*`` schema."""
    ds = pd.to_datetime(dates).reset_index(drop=True)
    if len(ds) != len(countries):
        raise ValueError(
            f"dates length {len(ds)} != countries length {len(countries)}"
        )
    codes = np.asarray(
        [normalize_country(c, default=default_country) for c in countries],
        dtype=object,
    )
    out = {
        f"months_from_{k}": np.full(len(ds), NA_DISTANCE_DAYS, dtype=np.float32)
        for k in holiday_keys
    }
    for code in sorted(set(codes.tolist())):
        mask = codes == code
        if not mask.any():
            continue
        built = months_from_holiday_features(
            ds.loc[mask],
            holiday_keys=holiday_keys,
            country=code,
        )
        for k in holiday_keys:
            col = f"months_from_{k}"
            out[col][mask] = built[col].to_numpy(dtype=np.float32)
    return pd.DataFrame(out)


def build_country_month_holiday_features(
    df: pd.DataFrame,
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
    encoding: str = "month_has",
    sku_col: str = "id_var",
    date_col: str = "ds",
    country_col: Optional[str] = None,
    default_country: str = "US",
) -> pd.DataFrame:
    """
    Build aligned monthly holiday frame for a panel.

    ``encoding`` is ``month_has`` or ``months_from``. Country is taken from
    ``country_col`` if present, else parsed from ``{Country}_{code}`` in
    ``sku_col`` (falls back to ``default_country`` when the prefix is not a
    known calendar — e.g. Monash Car Parts ``T####`` ids).
    """
    enc = str(encoding).lower()
    if country_col is not None and country_col in df.columns:
        countries = df[country_col].tolist()
    else:
        countries = countries_from_sku_ids(df[sku_col], default=default_country)
    if enc == "months_from":
        return months_from_holiday_features_by_country(
            df[date_col],
            countries,
            holiday_keys=holiday_keys,
            default_country=default_country,
        )
    if enc == "month_has":
        return month_has_holiday_features_by_country(
            df[date_col],
            countries,
            holiday_keys=holiday_keys,
            default_country=default_country,
        )
    raise ValueError(f"Unsupported monthly holiday encoding: {encoding}")


def days_from_holiday_features(
    dates: pd.Series,
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
    country: str = "US",
) -> pd.DataFrame:
    """Signed days to nearest occurrence of each holiday (single-country)."""
    ds = pd.to_datetime(dates)
    years = range(int(ds.dt.year.min()) - 1, int(ds.dt.year.max()) + 2)
    cal = holiday_calendar(years, country=country)
    d = ds.to_numpy(dtype="datetime64[ns]")
    out = {}
    for k in holiday_keys:
        events = cal[k]
        if len(events) == 0:
            out[f"days_from_{k}"] = np.full(len(d), NA_DISTANCE_DAYS, dtype=np.float32)
            continue
        delta = (d[:, None] - events[None, :]).astype("timedelta64[D]").astype(np.int32)
        idx = np.argmin(np.abs(delta), axis=1)
        out[f"days_from_{k}"] = delta[np.arange(len(d)), idx].astype(np.float32)
    return pd.DataFrame(out)


def days_from_holiday_features_by_country(
    dates: pd.Series,
    countries: Sequence[str],
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
    default_country: str = "US",
) -> pd.DataFrame:
    """
    Row-wise country calendars with a **unified** ``days_from_*`` schema.

    Rows share the same column names; N/A keys for a country get
    ``NA_DISTANCE_DAYS``.
    """
    ds = pd.to_datetime(dates).reset_index(drop=True)
    if len(ds) != len(countries):
        raise ValueError(
            f"dates length {len(ds)} != countries length {len(countries)}"
        )
    codes = np.asarray(
        [normalize_country(c, default=default_country) for c in countries],
        dtype=object,
    )
    out = {
        f"days_from_{k}": np.full(len(ds), NA_DISTANCE_DAYS, dtype=np.float32)
        for k in holiday_keys
    }
    for code in sorted(set(codes.tolist())):
        mask = codes == code
        if not mask.any():
            continue
        built = days_from_holiday_features(
            ds.loc[mask],
            holiday_keys=holiday_keys,
            country=code,
        )
        for k in holiday_keys:
            col = f"days_from_{k}"
            out[col][mask] = built[col].to_numpy(dtype=np.float32)
    return pd.DataFrame(out)


def build_country_holiday_distances(
    df: pd.DataFrame,
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
    sku_col: str = "id_var",
    date_col: str = "ds",
    country_col: Optional[str] = None,
    default_country: str = "US",
) -> pd.DataFrame:
    """
    Build aligned ``days_from_*`` frame for a panel.

    Country is taken from ``country_col`` if present, else parsed from
    ``{Country}_{code}`` in ``sku_col``.
    """
    if country_col is not None and country_col in df.columns:
        countries = df[country_col].tolist()
    else:
        countries = countries_from_sku_ids(df[sku_col], default=default_country)
    return days_from_holiday_features_by_country(
        df[date_col],
        countries,
        holiday_keys=holiday_keys,
        default_country=default_country,
    )


# Retail-sensitive events often spill ±1 day around the calendar date.
RETAIL_WINDOW_KEYS: Sequence[str] = (
    "Valentine",
    "Easter",
    "Halloween",
    "Thanksgiving",
    "BlackFriday",
    "Christmas",
)


def binary_holiday_features(
    days_from_df: pd.DataFrame,
    holiday_keys: Sequence[str] = HOLIDAY_KEYS,
    window_days: int = 0,
    window_keys: Optional[Sequence[str]] = None,
    prefix: str = "is_",
    include_any: bool = False,
) -> pd.DataFrame:
    """
    On-day (or short-window) binary indicators from signed ``days_from_*``.

    For each key ``K``, ``{prefix}{K}=1`` when ``|days_from_K| <= w(K)``,
    where ``w(K)=window_days`` if ``K`` is in ``window_keys`` (default: all
    keys when ``window_keys`` is None), else ``w(K)=0`` (exact on-day).

    When ``include_any=True``, also emits ``{prefix}any_holiday`` = max over
    per-holiday binaries. Sentinel distances (``NA_DISTANCE_DAYS``) never fire.
    """
    keys = list(holiday_keys)
    win_set = set(window_keys) if window_keys is not None else set(keys)
    out: Dict[str, np.ndarray] = {}
    flags = []
    for k in keys:
        col = f"days_from_{k}"
        if col not in days_from_df.columns:
            raise ValueError(f"Missing distance column {col} for binary holiday {k}")
        w = int(window_days) if k in win_set else 0
        dist = days_from_df[col].to_numpy(dtype=np.float32)
        # Treat N/A sentinel as off even if window somehow matched.
        flag = (
            (np.abs(dist) <= w) & (np.abs(dist) < NA_DISTANCE_DAYS * 0.5)
        ).astype(np.float32)
        out[f"{prefix}{k}"] = flag
        flags.append(flag)
    if include_any and flags:
        out[f"{prefix}any_holiday"] = np.maximum.reduce(flags).astype(np.float32)
    return pd.DataFrame(out)
