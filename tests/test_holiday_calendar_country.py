"""Unit tests for multi-country holiday date rules and sku→country mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from holiday_calendar import (  # noqa: E402
    NA_DISTANCE_DAYS,
    binary_holiday_features,
    build_country_holiday_distances,
    build_country_month_holiday_features,
    country_from_sku_id,
    days_from_holiday_features,
    holiday_dates_for_year,
    month_has_holiday_features,
    months_from_holiday_features,
    normalize_country,
)


@pytest.mark.parametrize(
    "sku,code",
    [
        ("United Kingdom_22155", "UK"),
        ("Australia_10002", "AU"),
        ("USA_123", "US"),
        ("EIRE_22423", "IE"),
        ("France_22720", "FR"),
        ("Germany_22326", "DE"),
        ("Netherlands_22629", "NL"),
        ("Belgium_1", "EU"),
        ("Spain_9", "EU"),
    ],
)
def test_country_from_sku_id(sku, code):
    assert country_from_sku_id(sku) == code


def test_normalize_country_aliases():
    assert normalize_country("United Kingdom") == "UK"
    assert normalize_country("EIRE") == "IE"
    assert normalize_country("unknown_xyz", default="US") == "US"


def test_us_known_dates_2024():
    d = holiday_dates_for_year(2024, "US")
    assert d["NewYear"] == pd.Timestamp(2024, 1, 1)
    assert d["MLK"] == pd.Timestamp(2024, 1, 15)  # 3rd Mon Jan
    assert d["Presidents"] == pd.Timestamp(2024, 2, 19)
    assert d["Memorial"] == pd.Timestamp(2024, 5, 27)
    assert d["July4"] == pd.Timestamp(2024, 7, 4)
    assert d["Labor"] == pd.Timestamp(2024, 9, 2)
    assert d["Thanksgiving"] == pd.Timestamp(2024, 11, 28)
    assert d["BlackFriday"] == pd.Timestamp(2024, 11, 29)
    assert d["Christmas"] == pd.Timestamp(2024, 12, 25)


def test_uk_known_dates_2011_and_2024():
    # 2011 England & Wales bank holidays
    d2011 = holiday_dates_for_year(2011, "UK")
    assert d2011["Labor"] == pd.Timestamp(2011, 5, 2)  # Early May BH
    assert d2011["Memorial"] == pd.Timestamp(2011, 5, 30)  # Spring BH
    assert d2011["MLK"] == pd.Timestamp(2011, 8, 29)  # Summer BH (slot reuse)
    assert d2011["Mothers"] == pd.Timestamp(2011, 4, 3)  # Mothering Sunday
    assert d2011["July4"] is None
    assert d2011["Thanksgiving"] is None
    assert d2011["BlackFriday"] == pd.Timestamp(2011, 11, 25)

    d2024 = holiday_dates_for_year(2024, "UK")
    assert d2024["Labor"] == pd.Timestamp(2024, 5, 6)  # Early May BH
    assert d2024["Memorial"] == pd.Timestamp(2024, 5, 27)  # Spring BH
    assert d2024["MLK"] == pd.Timestamp(2024, 8, 26)  # Summer BH
    assert d2024["NewYear"] == pd.Timestamp(2024, 1, 1)


def test_au_known_dates_2024():
    d = holiday_dates_for_year(2024, "AU")
    assert d["MLK"] == pd.Timestamp(2024, 1, 26)  # Australia Day
    assert d["Memorial"] == pd.Timestamp(2024, 4, 25)  # ANZAC
    assert d["Labor"] == pd.Timestamp(2024, 10, 7)  # Labour Day (1st Mon Oct)
    assert d["Fathers"] == pd.Timestamp(2024, 9, 1)  # 1st Sun Sep
    assert d["Thanksgiving"] is None


def test_ie_fr_de_nl_smoke_dates():
    ie = holiday_dates_for_year(2024, "IE")
    assert ie["MLK"] == pd.Timestamp(2024, 3, 18)  # St Patrick's observed Mon
    assert ie["Labor"] == pd.Timestamp(2024, 5, 6)

    fr = holiday_dates_for_year(2024, "FR")
    assert fr["July4"] == pd.Timestamp(2024, 7, 14)  # Bastille
    assert fr["Labor"] == pd.Timestamp(2024, 5, 1)

    de = holiday_dates_for_year(2024, "DE")
    assert de["July4"] == pd.Timestamp(2024, 10, 3)
    assert de["Labor"] == pd.Timestamp(2024, 5, 1)

    nl = holiday_dates_for_year(2024, "NL")
    assert nl["July4"] == pd.Timestamp(2024, 4, 27)  # King's Day
    assert nl["Memorial"] == pd.Timestamp(2024, 5, 5)


def test_uk_days_from_zero_on_early_may_bank_holiday():
    dates = pd.Series([pd.Timestamp(2011, 5, 2)])
    dist = days_from_holiday_features(dates, country="UK")
    assert dist.loc[0, "days_from_Labor"] == 0.0
    # US July4 should be N/A sentinel for UK
    assert dist.loc[0, "days_from_July4"] == NA_DISTANCE_DAYS
    assert dist.loc[0, "days_from_Thanksgiving"] == NA_DISTANCE_DAYS


def test_panel_mixed_countries_unified_schema():
    df = pd.DataFrame(
        {
            "id_var": [
                "United Kingdom_1",
                "Australia_2",
                "USA_3",
            ],
            "ds": [
                pd.Timestamp(2011, 5, 2),
                pd.Timestamp(2024, 1, 26),
                pd.Timestamp(2024, 7, 4),
            ],
            "Quantity": [1.0, 1.0, 1.0],
        }
    )
    hol = build_country_holiday_distances(df)
    assert list(hol.columns) == [f"days_from_{k}" for k in [
        "NewYear", "MLK", "Presidents", "Valentine", "Easter", "Mothers",
        "Memorial", "Fathers", "July4", "Labor", "Halloween", "Thanksgiving",
        "BlackFriday", "Christmas", "NewYearEve",
    ]]
    assert hol.loc[0, "days_from_Labor"] == 0.0  # UK Early May
    assert hol.loc[1, "days_from_MLK"] == 0.0  # AU Australia Day
    assert hol.loc[2, "days_from_July4"] == 0.0  # US
    # UK row: July4 N/A
    assert hol.loc[0, "days_from_July4"] == NA_DISTANCE_DAYS

    binaries = binary_holiday_features(hol, include_any=True)
    assert binaries.loc[0, "is_Labor"] == 1.0
    assert binaries.loc[0, "is_July4"] == 0.0
    assert binaries.loc[1, "is_MLK"] == 1.0
    assert binaries.loc[2, "is_July4"] == 1.0


def test_us_calendar_matches_legacy_defaults():
    """US single-country path still matches historical 2024 anchors."""
    dates = pd.Series(pd.date_range("2024-01-01", periods=366, freq="D"))
    dist = days_from_holiday_features(dates, country="US")
    # On Thanksgiving 2024-11-28
    idx = dates[dates == "2024-11-28"].index[0]
    assert dist.loc[idx, "days_from_Thanksgiving"] == 0.0
    assert dist.loc[idx, "days_from_BlackFriday"] == -1.0


def test_month_has_us_november_thanksgiving():
    dates = pd.Series([pd.Timestamp(2024, 11, 1), pd.Timestamp(2024, 7, 1)])
    mh = month_has_holiday_features(dates, country="US")
    assert mh.loc[0, "month_has_Thanksgiving"] == 1.0
    assert mh.loc[0, "month_has_BlackFriday"] == 1.0
    assert mh.loc[0, "month_has_July4"] == 0.0
    assert mh.loc[1, "month_has_July4"] == 1.0


def test_month_has_uk_july4_off_and_labor_on():
    # UK Early May bank holiday is Labor slot; July4 N/A → month_has 0
    dates = pd.Series([pd.Timestamp(2011, 5, 1), pd.Timestamp(2011, 7, 1)])
    mh = month_has_holiday_features(dates, country="UK")
    assert mh.loc[0, "month_has_Labor"] == 1.0
    assert mh.loc[1, "month_has_July4"] == 0.0


def test_build_country_month_has_carparts_defaults_us():
    """Monash Car Parts T#### ids have no country prefix → default US."""
    df = pd.DataFrame(
        {
            "id_var": ["T1851", "United Kingdom_1"],
            "ds": [pd.Timestamp(2024, 7, 1), pd.Timestamp(2011, 5, 1)],
            "Quantity": [1.0, 1.0],
        }
    )
    hol = build_country_month_holiday_features(
        df, encoding="month_has", default_country="US"
    )
    assert hol.loc[0, "month_has_July4"] == 1.0  # US default for T1851
    assert hol.loc[1, "month_has_Labor"] == 1.0  # UK Early May
    assert hol.loc[1, "month_has_July4"] == 0.0


def test_months_from_country_sentinel_for_na():
    df = pd.DataFrame(
        {
            "id_var": ["United Kingdom_1"],
            "ds": [pd.Timestamp(2011, 7, 1)],
            "Quantity": [1.0],
        }
    )
    hol = build_country_month_holiday_features(
        df, encoding="months_from", default_country="US"
    )
    assert hol.loc[0, "months_from_July4"] == NA_DISTANCE_DAYS


def test_year_scope_christmas_resets_at_year_boundary():
    """
    Year-scoped convention: days_from = obs - holiday_date(Y).

    Early in the year, Christmas is still ahead → large **negative** distance
    to this year's Dec 25 (not a small positive to last year's Christmas).
    After Christmas in the same year → small positive. At the next Jan 1,
    Christmas of the new year applies immediately (again large negative).

    Contrast with distance_scope='nearest' (legacy / locked CSV style).
    """
    dates = pd.Series(
        [
            pd.Timestamp(2011, 1, 7),
            pd.Timestamp(2011, 12, 1),
            pd.Timestamp(2011, 12, 26),
            pd.Timestamp(2012, 1, 1),
        ]
    )
    year = days_from_holiday_features(dates, country="US", distance_scope="year")
    nearest = days_from_holiday_features(
        dates, country="US", distance_scope="nearest"
    )

    # 2011-01-07: year → −352 to 2011-12-25; nearest → +13 after 2010-12-25
    assert year.loc[0, "days_from_Christmas"] == -352.0
    assert nearest.loc[0, "days_from_Christmas"] == 13.0

    # Mid-December: both scopes agree (nearest event is this year's Christmas)
    assert year.loc[1, "days_from_Christmas"] == -24.0
    assert nearest.loc[1, "days_from_Christmas"] == -24.0

    # Day after Christmas: +1 under both
    assert year.loc[2, "days_from_Christmas"] == 1.0
    assert nearest.loc[2, "days_from_Christmas"] == 1.0

    # New Year's Day 2012: year resets to 2012-12-25 (−359), nearest still
    # counts days after 2011-12-25 (+7)
    assert year.loc[3, "days_from_Christmas"] == -359.0
    assert nearest.loc[3, "days_from_Christmas"] == 7.0


def test_year_scope_newyear_and_newyeareve_anchors():
    """NewYear = Jan 1 of Y; NewYearEve = Dec 31 of Y (not next year's Jan 1)."""
    dates = pd.Series(
        [pd.Timestamp(2011, 1, 1), pd.Timestamp(2011, 1, 7), pd.Timestamp(2011, 12, 31)]
    )
    dist = days_from_holiday_features(dates, country="US", distance_scope="year")
    assert dist.loc[0, "days_from_NewYear"] == 0.0
    assert dist.loc[0, "days_from_NewYearEve"] == -364.0
    assert dist.loc[1, "days_from_NewYear"] == 6.0
    assert dist.loc[2, "days_from_NewYearEve"] == 0.0
    # Dec 31 is far from this year's Jan 1 NewYear (not −1 to next NewYear)
    assert dist.loc[2, "days_from_NewYear"] == 364.0


def test_year_scope_months_from_christmas():
    """Jan vs Christmas is −11 months (this year), not +1 to prior December."""
    dates = pd.Series(
        [pd.Timestamp(2011, 1, 1), pd.Timestamp(2011, 6, 1), pd.Timestamp(2011, 12, 1)]
    )
    year = months_from_holiday_features(dates, country="US", distance_scope="year")
    nearest = months_from_holiday_features(
        dates, country="US", distance_scope="nearest"
    )
    assert year.loc[0, "months_from_Christmas"] == -11.0
    assert nearest.loc[0, "months_from_Christmas"] == 1.0
    assert year.loc[1, "months_from_Christmas"] == -6.0
    assert year.loc[2, "months_from_Christmas"] == 0.0


def test_default_distance_scope_is_year():
    dates = pd.Series([pd.Timestamp(2011, 1, 7)])
    dist = days_from_holiday_features(dates, country="US")
    assert dist.loc[0, "days_from_Christmas"] == -352.0


def test_year_scope_binaries_still_on_day():
    """is_* from |days_from| still fire on this year's holiday date."""
    dates = pd.Series([pd.Timestamp(2011, 12, 25), pd.Timestamp(2011, 1, 7)])
    dist = days_from_holiday_features(dates, country="UK", distance_scope="year")
    binaries = binary_holiday_features(dist, window_days=0)
    assert binaries.loc[0, "is_Christmas"] == 1.0
    assert binaries.loc[1, "is_Christmas"] == 0.0
