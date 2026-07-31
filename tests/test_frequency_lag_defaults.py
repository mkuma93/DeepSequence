"""Frequency-aware lag defaults and feature_config_loader resolution."""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "deepsequence_hierarchical_attention"
PKG_NAME = "deepsequence_hierarchical_attention"


def _ensure_package_stub() -> None:
    """Register package without executing TF-heavy ``__init__.py``."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    existing = sys.modules.get(PKG_NAME)
    if existing is not None and getattr(existing, "__path__", None):
        return
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [str(PKG)]  # type: ignore[attr-defined]
    pkg.__file__ = str(PKG / "__init__.py")
    sys.modules[PKG_NAME] = pkg


def _load_submodule(mod_name: str):
    _ensure_package_stub()
    full = f"{PKG_NAME}.{mod_name}"
    if full in sys.modules and hasattr(sys.modules[full], "__file__"):
        return sys.modules[full]
    path = PKG / f"{mod_name}.py"
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _presets():
    return _load_submodule("frequency_presets")


def _load_loader():
    # Pre-bind light submodules so feature_config_loader does not import package __init__.
    _load_submodule("frequency_presets")
    _load_submodule("intermittent_features")
    path = REPO / "examples" / "feature_config_loader.py"
    spec = importlib.util.spec_from_file_location("feature_config_loader", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_default_lags_presets_d_w_m_q():
    fp = _presets()
    assert fp.default_lags_for_frequency("D") == [1, 2, 7]
    assert fp.default_lags_for_frequency("daily") == [1, 2, 7]
    assert fp.default_lags_for_frequency("W") == [1, 2, 4]
    assert fp.default_lags_for_frequency("weekly") == [1, 2, 4]
    assert fp.default_lags_for_frequency("M") == [1, 2, 12]
    assert fp.default_lags_for_frequency("monthly") == [1, 2, 12]
    assert fp.default_lags_for_frequency("Q") == [1, 2, 4]
    assert fp.default_lags_for_frequency("quarterly") == [1, 2, 4]
    assert list(fp.DEFAULT_LAGS) == [1, 2, 7]
    assert fp.LAGS_BY_FREQUENCY["daily"] == (1, 2, 7)


def test_normalize_frequency_aliases():
    fp = _presets()
    assert fp.normalize_frequency("Days") == "daily"
    assert fp.normalize_frequency("m") == "monthly"
    with pytest.raises(ValueError, match="Unknown frequency"):
        fp.normalize_frequency("hourly")


def test_daily_locked_config_keeps_1_2_7():
    mod = _load_loader()
    cfg = mod.FeatureConfig(REPO / "feature_config.yaml")
    assert cfg.frequency == "daily"
    assert cfg.lag_periods == [1, 2, 7]
    assert cfg.lag_names == ["lag_1", "lag_2", "lag_7"]
    assert cfg.config["metadata"].get("lags_resolved_from") == "lag_features"


def test_monthly_config_keeps_1_2_12():
    fp = _presets()
    mod = _load_loader()
    cfg = mod.FeatureConfig(REPO / "feature_config_monthly.yaml")
    assert cfg.frequency == "monthly"
    assert cfg.lag_periods == [1, 2, 12]
    assert cfg.lag_names == ["lag_1", "lag_2", "lag_12"]
    # Locked monthly YAML matches the generic monthly preset.
    assert fp.default_lags_for_frequency("M") == [1, 2, 12]
    assert cfg.resolved_fourier_periods() == [3.0, 12.0]


def test_loader_auto_lags_for_each_frequency(tmp_path):
    mod = _load_loader()
    daily = yaml.safe_load((REPO / "feature_config.yaml").read_text())
    expected = {
        "D": [1, 2, 7],
        "W": [1, 2, 4],
        "M": [1, 2, 12],
        "Q": [1, 2, 4],
    }
    for freq, lags in expected.items():
        cfg_dict = copy.deepcopy(daily)
        cfg_dict["metadata"]["frequency"] = freq
        cfg_dict["metadata"]["lags"] = "auto"
        path = tmp_path / f"auto_{freq}.yaml"
        path.write_text(yaml.safe_dump(cfg_dict))
        cfg = mod.FeatureConfig(path)
        assert cfg.lag_periods == lags
        assert cfg.lag_names == [f"lag_{p}" for p in lags]
        assert "default_lags_for_frequency" in cfg.config["metadata"]["lags_resolved_from"]


def test_loader_explicit_metadata_lags_override(tmp_path):
    mod = _load_loader()
    daily = yaml.safe_load((REPO / "feature_config.yaml").read_text())
    daily["metadata"]["frequency"] = "daily"
    daily["metadata"]["lags"] = [1, 2, 14]
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(daily))
    cfg = mod.FeatureConfig(path)
    assert cfg.lag_periods == [1, 2, 14]
    assert cfg.lag_names == ["lag_1", "lag_2", "lag_14"]


def test_auto_without_frequency_raises(tmp_path):
    mod = _load_loader()
    daily = yaml.safe_load((REPO / "feature_config.yaml").read_text())
    daily["metadata"].pop("frequency", None)
    daily["metadata"]["lags"] = "auto"
    path = tmp_path / "bad_auto.yaml"
    path.write_text(yaml.safe_dump(daily))
    with pytest.raises(ValueError, match="frequency is unset"):
        mod.FeatureConfig(path)
