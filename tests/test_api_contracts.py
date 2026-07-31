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
    assert "build_hierarchical_model_hybrid" in text
    assert "create_hierarchical_model" not in text
    assert "DeepSequencePWLHierarchical" not in text
    assert (SRC_ROOT / "residual_transformer.py").exists()
    assert (SRC_ROOT / "hybrid_temporal.py").exists()


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


def test_lightweight_builder_removed_dead_api_and_uncertainty_path():
    tree = _parse(SRC_ROOT / "components_lightweight.py")
    top_level_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert "ComponentWithSKUWrapper" not in top_level_names
    assert "create_lightweight_model_simple" not in top_level_names
    assert "LearnableUncertaintyWeight" not in top_level_names

    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_hierarchical_model_lightweight"
    )
    builder_args = {arg.arg for arg in builder.args.args}
    assert "combination_mode" not in builder_args
    assert {
        "horizon",
        "use_sku",
        "component_attention_temperature",
        "component_entropy_weight",
        "intermittent_hidden_dim",
        "orthogonality_weight",
        "context_aware_component_mixer",
        "context_film_seasonal_holiday",
    } <= builder_args

    # Expert output_activation default is softsign (signed, milder than tanh).
    n_defaults = len(builder.args.defaults)
    named_defaults = {
        arg.arg: default.value
        for arg, default in zip(
            builder.args.args[-n_defaults:], builder.args.defaults
        )
        if isinstance(default, ast.Constant)
    }
    assert named_defaults.get("output_activation") == "softsign"
    assert named_defaults.get("context_film_seasonal_holiday") is False
    assert named_defaults.get("context_aware_component_mixer") is True
    assert named_defaults.get("trend_monotonic") is True
    assert named_defaults.get("holiday_monotonic") is True
    assert named_defaults.get("regressor_monotonic") is True

    source = (SRC_ROOT / "components_lightweight.py").read_text(encoding="utf-8")
    assert "forecast_uncertainty" not in source
    assert "classification_uncertainty" not in source


def test_graph_helper_layers_are_keras_serializable():
    tree = _parse(SRC_ROOT / "components_lightweight.py")
    required = {
        "ClipByValue",
        "StackComponentsLayer",
        "ComponentEntropy",
        "ComponentEntropyLoss",
        "PrintAttentionWeights",
        "SumWeightedComponents",
        "OrthogonalityPenalty",
    }
    decorated = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in required:
            continue
        if any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "register_keras_serializable"
            for decorator in node.decorator_list
        ):
            decorated.add(node.name)
    assert decorated == required


def test_builder_helpers_cover_component_and_head_paths():
    tree = _parse(SRC_ROOT / "components_lightweight.py")
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert {
        "_build_components",
        "_build_intermittent_heads",
        "_build_component_attention",
        "_build_component_mixer_source",
        "_apply_context_film_seasonal_holiday",
        "_build_sku_path",
    } <= names
    source = (SRC_ROOT / "components_lightweight.py").read_text(encoding="utf-8")
    assert "component_attention_source" not in source
    assert "package='DeepSequence'" not in source
    assert 'package="DeepSequence"' not in source
    assert "package='DeepSequenceHierarchical'" not in source


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
