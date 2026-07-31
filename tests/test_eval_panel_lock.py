"""Tests for bake-off panel locking (data_seed vs train_seed, frozen SKU lists)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from eval_helpers import resolve_eval_seeds, select_eval_skus  # noqa: E402


def test_resolve_eval_seeds_legacy_sets_both():
    assert resolve_eval_seeds(42) == (42, 42)
    assert resolve_eval_seeds(7, None, None) == (7, 7)


def test_resolve_eval_seeds_split():
    assert resolve_eval_seeds(42, data_seed=1, train_seed=99) == (1, 99)
    assert resolve_eval_seeds(42, data_seed=1, train_seed=None) == (1, 42)
    assert resolve_eval_seeds(42, data_seed=None, train_seed=99) == (42, 99)


def test_select_eval_skus_is_deterministic(tmp_path):
    universe = [f"sku_{i}" for i in range(100)]
    a = select_eval_skus(universe, max_skus=20, data_seed=42)
    b = select_eval_skus(universe, max_skus=20, data_seed=42)
    c = select_eval_skus(universe, max_skus=20, data_seed=43)
    assert a == b
    assert set(a) != set(c)
    assert len(a) == 20


def test_select_eval_skus_round_trip_freeze(tmp_path):
    universe = [f"sku_{i}" for i in range(50)]
    path = tmp_path / "panel_skus.json"
    sampled = select_eval_skus(
        universe, max_skus=10, data_seed=42, save_sku_list_path=str(path)
    )
    payload = json.loads(path.read_text())
    assert payload == [str(x) for x in sampled]

    reloaded = select_eval_skus(
        universe, max_skus=10, data_seed=999, sku_list_path=str(path)
    )
    assert set(reloaded) == set(sampled)


def test_select_eval_skus_accepts_stringified_ids(tmp_path):
    universe = np.array([10, 20, 30, 40], dtype=object)
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["20", "40"]))
    chosen = select_eval_skus(
        universe, max_skus=2, data_seed=0, sku_list_path=str(path)
    )
    assert set(chosen) == {20, 40}
