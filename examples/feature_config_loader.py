"""
Feature Configuration Loader and Validator
Ensures all models use the exact same feature specification.

Regressor lag + intermittent features are built causally via
``intermittent_features.transform_panel`` (history with ds < t only).
"""

from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

try:
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
    """Load and validate feature configuration from YAML."""

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

        self._validate_config()
        self.config_path = str(resolved)

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
        holiday_features_df: pd.DataFrame,
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
            prior_states: Optional warm-start SKU states
            return_states: If True, also return end-of-panel states
            days_since_sentinel: Value when SKU never sold before t

        Returns:
            features_df, or (features_df, states) if return_states
        """
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
            else:
                raise ValueError(f"Unknown transformation: {transformation}")

        day_of_week = df_sorted["ds"].dt.dayofweek.values
        month = df_sorted["ds"].dt.month.values
        day_of_year = df_sorted["ds"].dt.dayofyear.values

        for feat_config in self.config["cyclical_features"]:
            name = feat_config["name"]
            if "dow" in name:
                period, value = 7, day_of_week
            elif "month" in name:
                period, value = 12, month
            elif "year" in name:
                period, value = 365.25, day_of_year
            else:
                raise ValueError(f"Unknown cyclical feature: {name}")
            if "sin" in name:
                features[name] = np.sin(2 * np.pi * value / period)
            else:
                features[name] = np.cos(2 * np.pi * value / period)

        # Causal lags + intermittent regressor features
        meta = self.config.get("metadata", {}).get("intermittent_features", {})
        sentinel = float(meta.get("cold_start_days_since_sentinel", days_since_sentinel))
        causal_df, end_states = transform_panel(
            df_sorted,
            id_col="id_var",
            date_col="ds",
            quantity_col="Quantity",
            lags=self.lag_periods,
            prior_states=prior_states,
            days_since_sentinel=sentinel,
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
        actual_holidays = [c for c in holiday_sorted.columns if c.startswith("days_from_")]
        missing = set(expected_holidays) - set(actual_holidays)
        if missing:
            raise ValueError(f"Missing holiday features: {missing}")

        holiday_subset = holiday_sorted[expected_holidays].reset_index(drop=True)
        features_df = pd.concat([features_df, holiday_subset], axis=1)

        binary_holiday_names = self.binary_holiday_names
        if binary_holiday_names:
            for dist_name, binary_name in zip(expected_holidays, binary_holiday_names):
                features_df[binary_name] = (holiday_sorted[dist_name].values == 0).astype(int)

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
