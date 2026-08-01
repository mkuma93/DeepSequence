"""
Feature Configuration Loader and Validator
Ensures all models use the exact same feature specification.

Regressor lag + intermittent features are built causally via
``intermittent_features.transform_panel`` (history with ds < t only).

Lag / Fourier frequency defaults
--------------------------------
Explicit ``lag_features`` (or ``metadata.lags: [..]``) always win.
If ``metadata.lags`` is ``auto`` (or lag_features is empty) and
``metadata.frequency`` is set (D/W/M/Q), lags fill from
``default_lags_for_frequency``. Daily locked configs keep ``{1,2,7}``;
monthly YAML keeps ``{1,2,12}`` (== ``default_lags_for_frequency("M")``).
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

try:
    from deepsequence_hierarchical_attention.frequency_presets import (
        coerce_lag_list,
        default_fourier_periods_for_frequency,
        default_lags_for_frequency,
        is_auto_lags_spec,
        normalize_frequency,
    )
    from deepsequence_hierarchical_attention.intermittent_features import (
        INTERMITTENT_FEATURE_NAMES,
        SKUDemandState,
        StateMap,
        transform_panel,
        build_states_from_history,
        save_states,
        load_states,
        features_from_state,
        update_state,
    )
except ImportError:  # running from examples/ without package install
    import sys

    _pkg = Path(__file__).resolve().parents[1] / "deepsequence_hierarchical_attention"
    if str(_pkg.parent) not in sys.path:
        sys.path.insert(0, str(_pkg.parent))
    from deepsequence_hierarchical_attention.frequency_presets import (
        coerce_lag_list,
        default_fourier_periods_for_frequency,
        default_lags_for_frequency,
        is_auto_lags_spec,
        normalize_frequency,
    )
    from deepsequence_hierarchical_attention.intermittent_features import (
        INTERMITTENT_FEATURE_NAMES,
        SKUDemandState,
        StateMap,
        transform_panel,
        build_states_from_history,
        save_states,
        load_states,
        features_from_state,
        update_state,
    )


class FeatureConfig:
    """Load and validate feature configuration from YAML.

    Frequency-aware lags: see module docstring. After load,
    ``lag_periods`` reflects either explicit YAML lags or frequency presets.
    """

    def __init__(self, config_path=None):
        candidates = []
        if config_path is not None:
            candidates.append(Path(config_path))
        # Repo root (examples/../feature_config.yaml)
        candidates.append(Path(__file__).resolve().parent.parent / "feature_config.yaml")
        # Packaged copy inside installed module
        try:
            import deepsequence_hierarchical_attention as _pkg

            candidates.append(
                Path(_pkg.__file__).resolve().parent / "feature_config.yaml"
            )
        except Exception:
            pass

        resolved = None
        for candidate in candidates:
            if candidate is not None and candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(
                "feature_config.yaml not found. Tried:\n  "
                + "\n  ".join(str(c) for c in candidates)
            )

        with open(resolved, "r") as f:
            self.config = yaml.safe_load(f)

        self._resolve_frequency_defaults()
        self._validate_config()
        self.config_path = str(resolved)

    def _metadata(self) -> dict:
        return self.config.setdefault("metadata", {})

    @property
    def frequency(self) -> Optional[str]:
        """Canonical frequency if ``metadata.frequency`` / ``freq`` is set."""
        meta = self.config.get("metadata", {}) or {}
        raw = meta.get("frequency", meta.get("freq"))
        if raw is None:
            return None
        return normalize_frequency(raw)

    def _resolve_frequency_defaults(self) -> None:
        """Fill lags from frequency presets when YAML requests auto / omits them."""
        meta = self._metadata()
        lags_spec = meta.get("lags", meta.get("lag_periods"))
        lag_features = self.config.get("lag_features")
        freq_raw = meta.get("frequency", meta.get("freq"))

        explicit_list = isinstance(lags_spec, (list, tuple))
        auto = is_auto_lags_spec(lags_spec)
        missing_features = lag_features is None or lag_features == []

        if explicit_list:
            lags = coerce_lag_list(lags_spec)
            self._materialize_lag_features(lags)
            meta["lags_resolved_from"] = "metadata.lags"
            return

        if auto or missing_features:
            if freq_raw is None:
                if auto:
                    raise ValueError(
                        "metadata.lags is 'auto' but metadata.frequency is unset"
                    )
                return
            lags = default_lags_for_frequency(freq_raw)
            self._materialize_lag_features(lags)
            meta["lags"] = "auto" if auto else meta.get("lags", "auto")
            meta["lags_resolved_from"] = f"default_lags_for_frequency({freq_raw!r})"
            return

        # Explicit lag_features: leave as-is (daily {1,2,7}, monthly {1,2,12}, …)
        meta.setdefault("lags_resolved_from", "lag_features")

    def _materialize_lag_features(self, lags: List[int]) -> None:
        """Replace lag_features names/lags; keep starting index when possible."""
        existing = self.config.get("lag_features") or []
        start_index = int(existing[0]["index"]) if existing else None
        if start_index is None:
            # After trend + cyclical
            n_before = len(self.config.get("trend_features", [])) + len(
                self.config.get("cyclical_features", [])
            )
            start_index = n_before

        old_names = [f["name"] for f in existing]
        new_features = []
        for i, lag in enumerate(lags):
            new_features.append(
                {
                    "name": f"lag_{lag}",
                    "description": f"Demand {lag} step(s) ago (causal shift)",
                    "lag": int(lag),
                    "index": start_index + i,
                }
            )
        self.config["lag_features"] = new_features

        new_names = [f["name"] for f in new_features]
        if old_names and "feature_order" in self.config:
            order = list(self.config["feature_order"])
            # Replace old lag names in-place when counts match; else splice
            if len(old_names) == len(new_names):
                name_map = dict(zip(old_names, new_names))
                self.config["feature_order"] = [name_map.get(n, n) for n in order]
            else:
                # Drop old lag names, insert new ones at first lag position
                first_pos = min(
                    (order.index(n) for n in old_names if n in order),
                    default=start_index,
                )
                order = [n for n in order if n not in old_names]
                for j, name in enumerate(new_names):
                    order.insert(first_pos + j, name)
                self.config["feature_order"] = order
                self._metadata()["total_features"] = len(order)

        arch = self.config.get("model_architecture", {})
        reg = arch.get("regressor_component")
        if reg is not None and old_names:
            names = list(reg.get("feature_names", []))
            if len(old_names) == len(new_names):
                name_map = dict(zip(old_names, new_names))
                reg["feature_names"] = [name_map.get(n, n) for n in names]
            else:
                # Keep intermittent tail; replace lag head
                rest = [n for n in names if n not in old_names]
                reg["feature_names"] = new_names + rest
                # Indices: lag block then intermittent (assumes contiguous)
                lag_idx = [start_index + i for i in range(len(new_names))]
                inter_start = start_index + len(new_names)
                inter_idx = list(
                    range(inter_start, inter_start + len(rest))
                )
                reg["feature_indices"] = lag_idx + inter_idx

    def resolved_fourier_periods(self) -> Optional[List[float]]:
        """Fourier periods from metadata or frequency presets (None if unknown)."""
        meta = self.config.get("metadata", {}) or {}
        for key in (
            "fourier_periods",
            "fourier_periods_months",
            "fourier_periods_days",
        ):
            if key in meta and meta[key] is not None:
                spec = meta[key]
                if is_auto_lags_spec(spec):
                    break
                return [float(x) for x in spec]
        freq = self.frequency
        if freq is None:
            return None
        return default_fourier_periods_for_frequency(freq)

    def _validate_config(self):
        """Validate configuration is complete and consistent."""
        required_sections = [
            "cyclical_features",
            "lag_features",
            "holiday_features",
            "model_architecture",
            "feature_order",
            "metadata",
        ]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")

        expected_total = self.config["metadata"]["total_features"]
        actual_total = len(self.config["feature_order"])
        if expected_total != actual_total:
            raise ValueError(
                f"Feature count mismatch: expected {expected_total}, got {actual_total}"
            )

        all_features = []
        if "trend_features" in self.config:
            all_features.extend(self.config["trend_features"])
        all_features.extend(self.config["cyclical_features"])
        all_features.extend(self.config["lag_features"])
        if "intermittent_features" in self.config:
            all_features.extend(self.config["intermittent_features"])
        all_features.extend(self.config["holiday_features"])
        if "binary_holiday_features" in self.config:
            all_features.extend(self.config["binary_holiday_features"])

        indices = [f["index"] for f in all_features]
        expected_indices = list(range(len(all_features)))
        if indices != expected_indices:
            raise ValueError("Feature indices are not sequential")

    @property
    def total_features(self):
        return self.config["metadata"]["total_features"]

    @property
    def cyclical_names(self):
        return [f["name"] for f in self.config["cyclical_features"]]

    @property
    def lag_names(self):
        return [f["name"] for f in self.config["lag_features"]]

    @property
    def intermittent_names(self):
        if "intermittent_features" in self.config:
            return [f["name"] for f in self.config["intermittent_features"]]
        return list(INTERMITTENT_FEATURE_NAMES)

    @property
    def holiday_names(self):
        return [f["name"] for f in self.config["holiday_features"]]

    @property
    def binary_holiday_names(self):
        if "binary_holiday_features" in self.config:
            return [f["name"] for f in self.config["binary_holiday_features"]]
        return []

    @property
    def holiday_block_names(self):
        """Distance (+ optional binary) columns appended after regressors."""
        return self.holiday_names + self.binary_holiday_names

    @property
    def feature_names(self):
        return self.config["feature_order"]

    @property
    def lag_periods(self):
        return [int(f["lag"]) for f in self.config["lag_features"]]

    @property
    def trend_indices(self):
        if "trend_component" in self.config["model_architecture"]:
            return self.config["model_architecture"]["trend_component"]["feature_indices"]
        return []

    @property
    def seasonal_indices(self):
        return self.config["model_architecture"]["seasonal_component"]["feature_indices"]

    @property
    def regressor_indices(self):
        return self.config["model_architecture"]["regressor_component"]["feature_indices"]

    @property
    def holiday_indices(self):
        return self.config["model_architecture"]["holiday_component"]["feature_indices"]

    def create_features(
        self,
        df: pd.DataFrame,
        holiday_features_df: Optional[pd.DataFrame] = None,
        prior_states: Optional[Mapping[str, SKUDemandState]] = None,
        return_states: bool = False,
        days_since_sentinel: float = -1.0,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, StateMap]]:
        """
        Create features according to config specification.

        Causal regressor features (lags + intermittent) use only Quantity
        history with ds < t. Pass ``prior_states`` from a previous split
        (e.g. end of train) when transforming val/test so early rows are
        not falsely cold-started.

        Args:
            df: DataFrame with columns ['ds', 'id_var', 'Quantity']
            holiday_features_df: Holiday distance features aligned to ``df``
                rows **before** sorting (same length / order as input df).
                Optional when the config has no holiday features.
            prior_states: Optional warm-start SKU states
            return_states: If True, also return end-of-panel states
            days_since_sentinel: Value when SKU never sold before t

        Returns:
            features_df, or (features_df, states) if return_states
        """
        if holiday_features_df is None:
            holiday_features_df = pd.DataFrame(index=range(len(df)))
        if len(holiday_features_df) != len(df):
            raise ValueError(
                f"holiday_features_df length {len(holiday_features_df)} "
                f"!= df length {len(df)}"
            )

        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        # Keep holiday rows locked to original df order via positional index
        holiday_features_df = holiday_features_df.reset_index(drop=True)
        df = df.reset_index(drop=True)

        sort_order = df.sort_values(["id_var", "ds"], kind="mergesort").index.to_numpy()
        df_sorted = df.loc[sort_order].reset_index(drop=True)
        holiday_sorted = holiday_features_df.loc[sort_order].reset_index(drop=True)

        features: Dict[str, np.ndarray] = {}

        for trend_feature in self.config["trend_features"]:
            name = trend_feature["name"]
            source_col = trend_feature["source_column"]
            transformation = trend_feature["transformation"]
            if transformation == "days_since_epoch":
                epoch = pd.Timestamp("1970-01-01")
                features[name] = (df_sorted[source_col] - epoch).dt.days.values.astype(float)
            elif transformation == "months_since_epoch":
                ds = pd.to_datetime(df_sorted[source_col])
                features[name] = (ds.dt.year * 12 + ds.dt.month).astype(float).to_numpy()
            elif transformation == "weeks_since_epoch":
                epoch = pd.Timestamp("1970-01-01")
                days = (df_sorted[source_col] - epoch).dt.days.values.astype(float)
                features[name] = days / 7.0
            else:
                raise ValueError(f"Unknown transformation: {transformation}")

        ds = pd.to_datetime(df_sorted["ds"])
        day_of_week = ds.dt.dayofweek.values
        month = ds.dt.month.values
        day_of_year = ds.dt.dayofyear.values
        # ISO week-of-year (1–53); used by weekly panels (Monday-start ds).
        week_of_year = ds.dt.isocalendar().week.astype(float).to_numpy()
        month_index = (ds.dt.year * 12 + ds.dt.month).astype(float).to_numpy()

        for feat_config in self.config["cyclical_features"]:
            name = feat_config["name"]
            # Explicit source/period (monthly/weekly profile) or legacy name heuristics (daily)
            if "source" in feat_config and "period" in feat_config:
                source = feat_config["source"]
                period = float(feat_config["period"])
                if source == "month_index":
                    value = month_index
                elif source == "month_of_year":
                    value = month.astype(float)
                elif source == "day_of_week":
                    value = day_of_week.astype(float)
                elif source == "day_of_year":
                    value = day_of_year.astype(float)
                elif source in ("week_of_year", "iso_week"):
                    value = week_of_year
                else:
                    raise ValueError(f"Unknown cyclical source: {source}")
            elif "dow" in name:
                period, value = 7.0, day_of_week.astype(float)
            elif "woy" in name or "weekofyear" in name.replace("_", ""):
                period, value = 365.25 / 7.0, week_of_year
            elif "month" in name:
                period, value = 12.0, month.astype(float)
            elif "year" in name:
                period, value = 365.25, day_of_year.astype(float)
            else:
                raise ValueError(f"Unknown cyclical feature: {name}")
            if "sin" in name:
                features[name] = np.sin(2 * np.pi * value / period)
            else:
                features[name] = np.cos(2 * np.pi * value / period)

        # Causal lags + intermittent regressor features
        meta = self.config.get("metadata", {}).get("intermittent_features", {})
        sentinel = float(meta.get("cold_start_days_since_sentinel", days_since_sentinel))
        gap_unit = str(meta.get("gap_unit", "days"))
        rate_window = int(
            meta.get(
                "rate_window",
                self.config.get("metadata", {}).get("rate_window", 12),
            )
        )
        causal_df, end_states = transform_panel(
            df_sorted,
            id_col="id_var",
            date_col="ds",
            quantity_col="Quantity",
            lags=self.lag_periods,
            prior_states=prior_states,
            days_since_sentinel=sentinel,
            gap_unit=gap_unit,
            intermittent_names=self.intermittent_names,
            rate_window=rate_window,
            return_states=True,
        )
        for col in causal_df.columns:
            features[col] = causal_df[col].to_numpy(dtype=float)

        # Explicit column order: trend/cyclical already inserted; add regressor in config order
        regressor_order = self.lag_names + self.intermittent_names
        ordered = {}
        for name in self.config["trend_features"]:
            ordered[name["name"]] = features[name["name"]]
        for name in self.cyclical_names:
            ordered[name] = features[name]
        for name in regressor_order:
            ordered[name] = features[name]
        features_df = pd.DataFrame(ordered)

        expected_holidays = self.holiday_names
        meta = self.config.get("metadata", {}) or {}
        holiday_encoding = str(meta.get("holiday_encoding", "days_from"))
        holiday_calendar_mode = str(meta.get("holiday_calendar", "static")).lower()
        # year = within-year reset (default for rebuilt paths); nearest = legacy
        # cross-year. Locked bake-off CSVs are nearest-style until regenerated.
        holiday_distance_scope = str(
            meta.get("holiday_distance_scope", meta.get("distance_scope", "year"))
        ).lower()
        if expected_holidays:
            enc_norm = (
                holiday_encoding.lower().replace("+", "_").replace("-", "_")
            )
            if enc_norm in (
                "months_from_and_month_has",
                "month_has_and_months_from",
            ):
                enc_norm = "months_from_month_has"
            if enc_norm in ("month_has", "months_from", "months_from_month_has"):
                try:
                    from holiday_calendar import (
                        month_has_holiday_features,
                        months_from_holiday_features,
                    )
                except ImportError:
                    from examples.holiday_calendar import (  # type: ignore
                        month_has_holiday_features,
                        months_from_holiday_features,
                    )
                mf_keys: List[str] = []
                mh_keys: List[str] = []
                for name in expected_holidays:
                    if name.startswith("months_from_"):
                        mf_keys.append(name.replace("months_from_", "", 1))
                    elif name.startswith("month_has_"):
                        mh_keys.append(name.replace("month_has_", "", 1))
                    else:
                        raise ValueError(
                            f"{holiday_encoding} encoding expects months_from_* "
                            f"and/or month_has_* names, got {name}"
                        )
                if enc_norm == "months_from" and mh_keys:
                    raise ValueError(
                        "holiday_encoding=months_from but found month_has_* names"
                    )
                if enc_norm == "month_has" and mf_keys:
                    raise ValueError(
                        "holiday_encoding=month_has but found months_from_* names"
                    )
                if enc_norm == "months_from_month_has" and not (mf_keys and mh_keys):
                    raise ValueError(
                        "holiday_encoding=months_from_month_has expects both "
                        "months_from_* and month_has_* feature names"
                    )
                # Prefer months_from key order when both present; month_has keys
                # may repeat the same holiday labels.
                keys = list(dict.fromkeys(mf_keys + mh_keys))
                build_enc = enc_norm
                if holiday_calendar_mode in ("country", "per_country", "country_aware"):
                    try:
                        from holiday_calendar import (
                            build_country_month_holiday_features,
                        )
                    except ImportError:
                        from examples.holiday_calendar import (  # type: ignore
                            build_country_month_holiday_features,
                        )
                    country_col = meta.get("holiday_country_column")
                    built = build_country_month_holiday_features(
                        df_sorted,
                        holiday_keys=keys,
                        encoding=build_enc,
                        sku_col="id_var",
                        date_col="ds",
                        country_col=country_col if country_col in df_sorted.columns else None,
                        default_country=str(
                            meta.get(
                                "holiday_country_default",
                                meta.get("holiday_country", "US"),
                            )
                        ),
                        distance_scope=holiday_distance_scope,
                    )
                else:
                    country = str(meta.get("holiday_country", "US"))
                    parts = []
                    if mf_keys or build_enc in ("months_from", "months_from_month_has"):
                        use_keys = mf_keys or keys
                        parts.append(
                            months_from_holiday_features(
                                df_sorted["ds"],
                                holiday_keys=use_keys,
                                country=country,
                                distance_scope=holiday_distance_scope,
                            )
                        )
                    if mh_keys or build_enc in ("month_has", "months_from_month_has"):
                        use_keys = mh_keys or keys
                        parts.append(
                            month_has_holiday_features(
                                df_sorted["ds"], holiday_keys=use_keys, country=country
                            )
                        )
                    built = (
                        pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
                        if len(parts) > 1
                        else parts[0]
                    )
                holiday_subset = built[expected_holidays].reset_index(drop=True)
                features_df = pd.concat([features_df, holiday_subset], axis=1)
            else:
                # days_from_*: either use precomputed frame, or rebuild from
                # per-country calendars (sku_id prefix / country column).
                if holiday_calendar_mode in ("country", "per_country", "country_aware"):
                    try:
                        from holiday_calendar import build_country_holiday_distances
                    except ImportError:
                        from examples.holiday_calendar import (  # type: ignore
                            build_country_holiday_distances,
                        )
                    keys = [
                        n.replace("days_from_", "", 1)
                        if n.startswith("days_from_")
                        else n
                        for n in expected_holidays
                    ]
                    country_col = meta.get("holiday_country_column")
                    built = build_country_holiday_distances(
                        df_sorted,
                        holiday_keys=keys,
                        sku_col="id_var",
                        date_col="ds",
                        country_col=country_col if country_col in df_sorted.columns else None,
                        default_country=str(meta.get("holiday_country_default", "US")),
                        distance_scope=holiday_distance_scope,
                    )
                    holiday_subset = built[expected_holidays].reset_index(drop=True)
                    # Keep holiday_sorted in sync for binary derivation below.
                    holiday_sorted = built.reset_index(drop=True)
                else:
                    actual_holidays = [
                        c for c in holiday_sorted.columns if c.startswith("days_from_")
                    ]
                    missing = set(expected_holidays) - set(actual_holidays)
                    if missing:
                        raise ValueError(f"Missing holiday features: {missing}")
                    holiday_subset = holiday_sorted[expected_holidays].reset_index(drop=True)
                features_df = pd.concat([features_df, holiday_subset], axis=1)

                binary_holiday_names = self.binary_holiday_names
                if binary_holiday_names:
                    try:
                        from holiday_calendar import (
                            RETAIL_WINDOW_KEYS,
                            binary_holiday_features,
                        )
                    except ImportError:
                        from examples.holiday_calendar import (  # type: ignore
                            RETAIL_WINDOW_KEYS,
                            binary_holiday_features,
                        )
                    window_days = int(meta.get("binary_holiday_window_days", 0))
                    window_keys = meta.get("binary_holiday_window_keys")
                    if window_keys is None and window_days > 0:
                        window_keys = list(RETAIL_WINDOW_KEYS)
                    # Keys implied by is_* names (exclude is_any_holiday).
                    keys = []
                    want_any = False
                    for bname in binary_holiday_names:
                        if bname in ("is_any_holiday", "is_AnyHoliday"):
                            want_any = True
                            continue
                        if not bname.startswith("is_"):
                            raise ValueError(
                                f"binary holiday name must start with is_, got {bname}"
                            )
                        keys.append(bname[len("is_") :])
                    built = binary_holiday_features(
                        holiday_sorted,
                        holiday_keys=keys or [
                            n.replace("days_from_", "", 1) for n in expected_holidays
                        ],
                        window_days=window_days,
                        window_keys=window_keys,
                        include_any=want_any,
                    )
                    for bname in binary_holiday_names:
                        if bname not in built.columns:
                            raise ValueError(
                                f"binary holiday feature {bname} not produced "
                                f"(have {list(built.columns)})"
                            )
                        features_df[bname] = built[bname].to_numpy()
        elif len(holiday_features_df.columns) > 0 and len(holiday_sorted.columns) > 0:
            pass

        if list(features_df.columns) != self.feature_names:
            raise ValueError(
                f"Feature order mismatch!\n"
                f"Expected: {self.feature_names}\n"
                f"Got: {list(features_df.columns)}"
            )

        # Restore original input row order
        inv = np.empty(len(sort_order), dtype=int)
        inv[sort_order] = np.arange(len(sort_order))
        features_df = features_df.iloc[inv].reset_index(drop=True)

        if return_states:
            return features_df, end_states
        return features_df

    def validate_features(self, features_df):
        expected_cols = self.feature_names
        actual_cols = list(features_df.columns)
        if actual_cols != expected_cols:
            raise ValueError(
                f"Feature validation failed!\n"
                f"Expected columns: {expected_cols}\n"
                f"Actual columns: {actual_cols}\n"
                f"Missing: {set(expected_cols) - set(actual_cols)}\n"
                f"Extra: {set(actual_cols) - set(expected_cols)}"
            )
        if len(actual_cols) != self.total_features:
            raise ValueError(
                f"Expected {self.total_features} features, got {len(actual_cols)}"
            )
        return True

    def print_summary(self):
        print("=" * 80)
        print(f"FEATURE CONFIGURATION v{self.config['metadata']['version']}")
        print("=" * 80)
        print(f"\nTotal Features: {self.total_features}")
        print(f"Last Updated: {self.config['metadata']['last_updated']}")
        print(f"\nContext: {self.config['metadata']['dataset_context']}")
        print("\n" + "-" * 80)
        print(f"REGRESSOR (lags + intermittent): indices {self.regressor_indices}")
        print("-" * 80)
        for name in self.lag_names + self.intermittent_names:
            print(f"  {name}")
        print("=" * 80)


def load_feature_config(config_path=None):
    """
    Load feature_config.yaml.

    Search order:
      1. Explicit ``config_path``
      2. Packaged ``deepsequence_hierarchical_attention/feature_config.yaml``
      3. Repo-root ``feature_config.yaml`` next to ``examples/``
    """
    if config_path is not None:
        return FeatureConfig(config_path)

    try:
        import deepsequence_hierarchical_attention as _pkg

        packaged = Path(_pkg.__file__).resolve().parent / "feature_config.yaml"
        if packaged.exists():
            return FeatureConfig(packaged)
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1] / "feature_config.yaml"
    return FeatureConfig(repo_root)


if __name__ == "__main__":
    config = load_feature_config()
    config.print_summary()
    print("\nFeature configuration loaded and validated successfully!")
