"""Unit tests for CumMAE / CumIWMAE lead-time cumulative metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from eval_helpers import cummae_from_rollout, kpi_block  # noqa: E402
from multihorizon_rollout import horizon_metrics  # noqa: E402


def test_cummae_matches_hand_formula():
    # Two origins, H=3
    y = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    yhat = np.array([[0.5, 0.5, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    # Cum sums: y → [[1,1,3],[0,0,0]]; yhat → [[0.5,1.0,2.0],[0,1,1]]
    # CumMAE(1)=mean(|0.5-1|, |0-0|)=0.25
    # CumMAE(2)=mean(|1-1|, |1-0|)=0.5
    # CumMAE(3)=mean(|2-3|, |1-0|)=1.0
    out = cummae_from_rollout(y, yhat, report_horizons=(1, 2, 3))
    assert out["by_horizon"]["1"]["cummae"] == pytest.approx(0.25)
    assert out["by_horizon"]["2"]["cummae"] == pytest.approx(0.5)
    assert out["by_horizon"]["3"]["cummae"] == pytest.approx(1.0)
    # Aliases track mae_all / iwmae on the cumsummed series
    for h in ("1", "2", "3"):
        b = out["by_horizon"][h]
        assert b["cummae"] == b["mae_all"]
        assert b["cum_iwmae"] == b["iwmae"]


def test_cummae_h1_equals_pointwise_mae():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=(40, 5)).astype(np.float64)
    yhat = rng.random((40, 5)) * 3.0
    cum = cummae_from_rollout(y, yhat, report_horizons=(1,))
    point = kpi_block(y[:, 0], yhat[:, 0])
    assert cum["by_horizon"]["1"]["cummae"] == pytest.approx(point["mae_all"])


def test_horizon_metrics_includes_by_horizon_cum():
    rng = np.random.default_rng(1)
    n, H = 20, 8
    y = rng.integers(0, 4, size=(n, H)).astype(np.float64)
    yhat = rng.random((n, H)) * 2.0
    skus = np.array([f"s{i % 4}" for i in range(n)])
    volume_map = {f"s{i}": ("low", "mid", "high", "low")[i] for i in range(4)}
    out = horizon_metrics(
        y, yhat, None, skus, volume_map, report_horizons=(1, 4, 8)
    )
    assert "by_horizon" in out and "by_horizon_cum" in out
    for h in ("1", "4", "8"):
        assert "overall" in out["by_horizon_cum"][h]
        assert "cummae" in out["by_horizon_cum"][h]["overall"]
        # H=1 cumulative == pointwise
        if h == "1":
            assert out["by_horizon_cum"][h]["overall"]["cummae"] == pytest.approx(
                out["by_horizon"][h]["overall"]["mae_all"]
            )
