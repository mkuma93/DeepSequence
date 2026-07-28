"""Lightweight API contract checks that do not require a full training run."""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "deepsequence_hierarchical_attention"


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_package_exports_lightweight_and_optional_residual():
    text = (SRC_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "build_hierarchical_model_lightweight" in text
    assert "transform_panel" in text
    assert "build_residual_transformer" in text
    assert "create_hierarchical_model" not in text
    assert "DeepSequencePWLHierarchical" not in text
    assert (SRC_ROOT / "residual_transformer.py").exists()


def test_composite_loss_omits_none_base_forecast():
    pytest.importorskip("tensorflow")
    from deepsequence_hierarchical_attention.losses import composite_loss

    cfg = composite_loss(zero_rate=0.9, average_nonzero_demand=5.0, pos_weight=3.0)
    assert "base_forecast" not in cfg["losses"]
    assert set(cfg["losses"]) == {"non_zero_probability", "final_forecast"}
    assert callable(cfg["losses"]["non_zero_probability"])
    assert callable(cfg["losses"]["final_forecast"])


def test_single_temperature_softmax_definition():
    tree = _parse(SRC_ROOT / "components_lightweight.py")
    defs = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TemperatureSoftmax"
    ]
    assert len(defs) == 1


def test_adaptive_train_script_clips_gradients_not_loss():
    text = (
        PACKAGE_ROOT / "examples" / "train_lightweight_adaptive_loss.py"
    ).read_text(encoding="utf-8")
    assert "tf.minimum(total_loss" not in text
    assert "tf.clip_by_global_norm" in text


def test_create_model_from_features_uses_composite_loss_dict():
    tree = _parse(SRC_ROOT / "components_lightweight.py")
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_model_from_features"
    )
    call_names = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.append(node.func.attr)
        if isinstance(node, ast.keyword) and node.arg == "false_negative_weight":
            pytest.fail("create_model_from_features still passes false_negative_weight=")
    assert "composite_loss" in call_names


def test_feature_config_v16_shipped():
    from deepsequence_hierarchical_attention import get_feature_config_path
    import yaml

    path = get_feature_config_path()
    assert path.exists()
    cfg = yaml.safe_load(path.read_text())
    assert cfg["metadata"]["version"] == "1.6"
    assert cfg["metadata"]["total_features"] == 28
    assert not cfg.get("binary_holiday_features")
