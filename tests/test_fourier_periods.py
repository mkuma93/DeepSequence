"""Calendar correctness of Fourier periods across sampling frequencies.

Periods are in time steps, so an annual cycle is 12 at monthly grain and
365.25 (not 365) at daily grain.
"""

import pytest

pytest.importorskip("tensorflow")

from deepsequence_hierarchical_attention.components_lightweight import (  # noqa: E402
    DAYS_PER_MONTH,
    DAYS_PER_QUARTER,
    DAYS_PER_YEAR,
    DEFAULT_FOURIER_PERIODS,
    FOURIER_MIN_PERIOD,
    FOURIER_PERIODS_BY_FREQUENCY,
    LearnableFourierFeatures,
    SeasonalComponentLightweight,
    fourier_periods_for_frequency,
    pad_fourier_periods,
)


def test_daily_periods_use_mean_gregorian_calendar():
    """30/91/365 ignore 28-31 day months and leap years."""
    assert DAYS_PER_YEAR == 365.25
    assert DAYS_PER_MONTH == pytest.approx(30.4375)
    assert DAYS_PER_QUARTER == pytest.approx(91.3125)

    daily = FOURIER_PERIODS_BY_FREQUENCY["daily"]
    assert daily == DEFAULT_FOURIER_PERIODS
    assert daily[:2] == (7.0, 14.0)
    assert daily[2:] == pytest.approx((30.4375, 91.3125, 365.25))
    # The old hard-coded integers are gone
    assert 30.0 not in daily and 91.0 not in daily and 365.0 not in daily


def test_monthly_periods_are_exact_integers():
    monthly = fourier_periods_for_frequency("monthly")
    assert monthly == [3.0, 6.0, 12.0]


def test_weekly_periods_are_year_relative():
    weekly = fourier_periods_for_frequency("weekly")
    # quarter, half-year and year in weeks
    assert weekly[1] == pytest.approx(13.0446, abs=1e-3)
    assert weekly[2] == pytest.approx(26.0893, abs=1e-3)
    assert weekly[3] == pytest.approx(52.1786, abs=1e-3)


def test_frequency_aliases_and_truncation():
    assert fourier_periods_for_frequency("M") == fourier_periods_for_frequency("monthly")
    assert fourier_periods_for_frequency("Days") == fourier_periods_for_frequency("daily")
    assert fourier_periods_for_frequency("monthly", n_frequencies=2) == [3.0, 6.0]


def test_unknown_frequency_is_rejected():
    with pytest.raises(ValueError, match="Unknown frequency"):
        fourier_periods_for_frequency("hourly")


def test_pad_fourier_periods_fills_requested_count():
    """Monthly has three natural periods; the layer needs one per frequency."""
    padded = pad_fourier_periods([3.0, 6.0, 12.0], 5)
    assert len(padded) == 5
    assert padded[:3] == [3.0, 6.0, 12.0]
    assert all(FOURIER_MIN_PERIOD < p <= 12.0 for p in padded[3:])
    assert pad_fourier_periods([3.0, 6.0, 12.0], 2) == [3.0, 6.0]


def test_max_period_no_longer_clips_the_annual_cycle():
    """A hard 365 cap pinned the yearly daily frequency at initialization."""
    layer = LearnableFourierFeatures(
        n_frequencies=5, initial_periods=list(DEFAULT_FOURIER_PERIODS)
    )
    layer.build((None, 1))
    assert layer.resolved_max_period(DEFAULT_FOURIER_PERIODS) > DAYS_PER_YEAR
    assert layer.get_learned_periods()[-1] == pytest.approx(365.25, rel=1e-4)


def test_monthly_seasonal_component_keeps_monthly_periods():
    seasonal = SeasonalComponentLightweight(
        use_learnable_fourier=True,
        n_learnable_frequencies=3,
        fourier_periods=fourier_periods_for_frequency("monthly"),
    )
    seasonal.build((None, 1))
    seasonal.fourier_layer.build((None, 1))
    learned = seasonal.fourier_layer.get_learned_periods()
    assert learned == pytest.approx([3.0, 6.0, 12.0], rel=1e-4)


def test_builder_resolves_periods_from_frequency():
    from deepsequence_hierarchical_attention import (
        build_hierarchical_model_lightweight,
    )

    model = build_hierarchical_model_lightweight(
        n_temporal_features=1,
        n_fourier_features=1,
        n_holiday_features=0,
        n_lag_features=3,
        n_skus=5,
        use_learnable_fourier=True,
        n_learnable_frequencies=3,
        fourier_frequency="monthly",
    )
    fourier_layer = model.get_layer("seasonal").fourier_layer
    assert fourier_layer.get_learned_periods() == pytest.approx(
        [3.0, 6.0, 12.0], rel=1e-4
    )
