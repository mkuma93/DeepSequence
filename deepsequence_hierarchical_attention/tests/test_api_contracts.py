"""Lightweight API contract checks that do not require a full training run."""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "deepsequence_hierarchical_attention"


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_create_hierarchical_model_has_no_use_pwl():
    tree = _parse(SRC_ROOT / "model.py")
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_hierarchical_model"
    )
    for node in ast.walk(fn):
        if isinstance(node, ast.keyword) and node.arg == "use_pwl":
            pytest.fail("create_hierarchical_model still passes use_pwl=")
        if isinstance(node, ast.keyword) and node.arg == "holiday_feature_index":
            # build_model must receive holiday_feature_indices, not the singular alias
            pytest.fail("build_model still called with holiday_feature_index=")
    arg_names = [a.arg for a in fn.args.args]
    assert "holiday_feature_indices" in arg_names
    assert "holiday_feature_index" in arg_names


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
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TemperatureSoftmax"
    ]
    assert len(defs) == 1


def test_train_scripts_clip_gradients_not_loss():
    for name in (
        "train_lightweight_adaptive_loss.py",
        "train_lightweight_mse_loss.py",
    ):
        text = (PACKAGE_ROOT / "examples" / name).read_text(encoding="utf-8")
        assert "tf.minimum(total_loss" not in text
        assert "tf.clip_by_global_norm" in text


def test_create_model_from_features_uses_composite_loss_dict():
    tree = _parse(SRC_ROOT / "components_lightweight.py")
    fn = next(
        node for node in tree.body
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
