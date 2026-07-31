"""Frequency-aware lag (and Fourier) defaults.

Lags are in *time steps of the series*, same convention as Fourier periods:
an annual lag is 7 at daily grain, 4 at weekly, 12 at monthly, 4 at quarterly.

Override policy (feature_config_loader):
  - Explicit ``lag_features`` / ``metadata.lags: [..]`` win.
  - ``metadata.lags: auto`` (or missing lag_features + known frequency) fills
    from :func:`default_lags_for_frequency`.

Fourier periods live in ``components_lightweight.fourier_periods_for_frequency``;
:func:`default_fourier_periods_for_frequency` is a thin alias for discoverability.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

# Canonical keys used by lag + Fourier presets.
FREQUENCY_ALIASES = {
    "d": "daily",
    "day": "daily",
    "days": "daily",
    "daily": "daily",
    "w": "weekly",
    "week": "weekly",
    "weeks": "weekly",
    "weekly": "weekly",
    "m": "monthly",
    "month": "monthly",
    "months": "monthly",
    "monthly": "monthly",
    "q": "quarterly",
    "quarter": "quarterly",
    "quarters": "quarterly",
    "quarterly": "quarterly",
}

# Presets: short + seasonal anchor (~week / ~month / ~year depending on grain).
LAGS_BY_FREQUENCY = {
    "daily": (1, 2, 7),
    "weekly": (1, 2, 4),
    "monthly": (1, 2, 12),
    "quarterly": (1, 2, 4),
}

DEFAULT_LAGS = LAGS_BY_FREQUENCY["daily"]


def normalize_frequency(frequency: Union[str, None]) -> str:
    """Map D/W/M/Q (and aliases) to canonical daily/weekly/monthly/quarterly."""
    if frequency is None:
        raise ValueError("frequency is required")
    key = str(frequency).strip().lower()
    if key not in FREQUENCY_ALIASES:
        raise ValueError(
            f"Unknown frequency {frequency!r}. "
            f"Expected one of {sorted(set(FREQUENCY_ALIASES.values()))} "
            f"(aliases: D/W/M/Q, day(s), week(s), month(s), quarter(s))."
        )
    return FREQUENCY_ALIASES[key]


def default_lags_for_frequency(frequency: Union[str, None]) -> List[int]:
    """Default causal lag offsets (in steps) for a sampling frequency."""
    key = normalize_frequency(frequency)
    return list(LAGS_BY_FREQUENCY[key])


def default_fourier_periods_for_frequency(
    frequency: Union[str, None],
    n_frequencies: Optional[int] = None,
) -> List[float]:
    """Alias of ``fourier_periods_for_frequency`` (lazy import; avoids TF at import)."""
    from .components_lightweight import fourier_periods_for_frequency

    return fourier_periods_for_frequency(frequency, n_frequencies=n_frequencies)


def is_auto_lags_spec(spec) -> bool:
    """True when YAML/metadata requests frequency-based lag fill-in."""
    if spec is None:
        return False
    if isinstance(spec, str) and spec.strip().lower() in ("auto", "default", "freq"):
        return True
    return False


def coerce_lag_list(lags: Sequence) -> List[int]:
    return [int(x) for x in lags]
