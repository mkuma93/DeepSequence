"""Tests for inventory / newsvendor planning metrics (TF-free)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_inventory_metrics():
    """Load module by path to avoid package ``__init__`` (TensorFlow) import."""
    path = ROOT / "deepsequence_hierarchical_attention" / "inventory_metrics.py"
    spec = importlib.util.spec_from_file_location(
        "ds_inventory_metrics", path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ds_inventory_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


inv = _load_inventory_metrics()


def _kpi_summary_fields(y, yhat) -> dict:
    y = np.asarray(y, np.float64).reshape(-1)
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    nz = y > 0
    return {
        "n_rows": int(len(y)),
        "n_nonzero": int(nz.sum()),
        "mae_nonzero": float(np.mean(np.abs(y[nz] - yhat[nz]))) if nz.any() else None,
        "bias_nonzero": float(yhat[nz].mean() - y[nz].mean()) if nz.any() else None,
        "mean_final": float(yhat.mean()),
        "mean_actual": float(y.mean()),
        "underforecast_rate_nonzero": (
            float(np.mean(yhat[nz] < y[nz])) if nz.any() else None
        ),
        "mae_all": float(np.mean(np.abs(y - yhat))),
    }


def test_nv_cost_cu1_equals_mae():
    rng = np.random.default_rng(0)
    y = rng.choice([0.0, 0.0, 0.0, 1.0, 3.0, 5.0], size=200)
    yhat = np.maximum(rng.normal(loc=0.8, scale=1.2, size=200), 0.0)
    metrics = inv.inventory_cost_metrics(y, yhat, cu_co_ratios=(1.0, 2.0))
    assert metrics["inventory_nv_cost_cu1"] == pytest.approx(
        float(np.mean(np.abs(y - yhat))), rel=1e-12
    )
    yhat_r = np.rint(np.maximum(yhat, 0.0))
    assert metrics["inventory_nv_cost_rounded_cu1"] == pytest.approx(
        float(np.mean(np.abs(y - yhat_r))), rel=1e-12
    )


def test_holding_and_stockout_proxies():
    y = np.array([0.0, 0.0, 0.0, 4.0, 6.0])
    yhat = np.array([1.0, 0.5, 0.0, 2.0, 8.0])
    metrics = inv.inventory_cost_metrics(
        y, yhat, cu_co_ratios=(2.0,), include_rounded=False
    )
    assert metrics["inventory_holding_proxy_zero"] == pytest.approx(1.5 / 3.0)
    assert metrics["inventory_stockout_proxy_nz"] == pytest.approx(1.0)
    assert metrics["inventory_fill_rate_nz"] == pytest.approx(0.5)
    assert metrics["inventory_qty_fill_nz"] == pytest.approx(0.8)
    assert metrics["inventory_nv_cost_cu2"] == pytest.approx(1.5)


def test_sales_revenue_loss_ignores_zero_day_overstock():
    """Zero-day forecast is holding cost, not revenue loss."""
    y = np.array([0.0, 0.0, 0.0, 4.0, 6.0])
    # Loud zeros vs quiet zeros; same sale-day under (2 units on first sale day).
    loud = np.array([2.0, 2.0, 2.0, 2.0, 6.0])
    quiet = np.array([0.0, 0.0, 0.0, 2.0, 6.0])
    loud_m = inv.inventory_cost_metrics(y, loud, include_rounded=False)
    quiet_m = inv.inventory_cost_metrics(y, quiet, include_rounded=False)
    # Sale-day unmet: day3 under=2, day4 under=0 → mean 1.0
    assert loud_m["sales_revenue_loss_units"] == pytest.approx(1.0)
    assert quiet_m["sales_revenue_loss_units"] == pytest.approx(1.0)
    assert loud_m["sales_revenue_loss_units"] == quiet_m["sales_revenue_loss_units"]
    assert loud_m["inventory_holding_cost_zero"] > quiet_m["inventory_holding_cost_zero"]
    assert loud_m["unnecessary_restock_zero"] == loud_m["inventory_holding_cost_zero"]
    # Rate = 1.0 / mean(y_nz)=5 → 0.2
    assert loud_m["sales_revenue_loss_rate"] == pytest.approx(0.2)
    assert loud_m["sales_revenue_loss_per_day"] == pytest.approx(2.0 / 5.0)
    # Combined: rev_per_day + 0.1 * (0.6 * 2.0) = 0.4 + 0.12 = 0.52
    assert loud_m["combined_ops_cost_h0p1"] == pytest.approx(0.4 + 0.1 * 0.6 * 2.0)
    assert quiet_m["combined_ops_cost_h0p1"] == pytest.approx(0.4 + 0.0)


def test_aggregate_rescore_matches_rowwise():
    rng = np.random.default_rng(7)
    y = rng.choice([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 5.0, 10.0], size=500)
    yhat = np.maximum(rng.normal(loc=1.2, scale=2.0, size=500), 0.0)
    row = inv.inventory_cost_metrics(y, yhat, include_rounded=False)
    summary = _kpi_summary_fields(y, yhat)
    recovered = inv.inventory_cost_from_kpi_summary(summary)
    assert recovered["inventory_rescore_source"] == "kpi_summary_continuous"
    assert recovered["inventory_mean_under"] == pytest.approx(
        row["inventory_mean_under"], rel=1e-9, abs=1e-9
    )
    assert recovered["inventory_mean_over"] == pytest.approx(
        row["inventory_mean_over"], rel=1e-9, abs=1e-9
    )
    assert recovered["inventory_holding_proxy_zero"] == pytest.approx(
        row["inventory_holding_proxy_zero"], rel=1e-9, abs=1e-9
    )
    assert recovered["inventory_stockout_proxy_nz"] == pytest.approx(
        row["inventory_stockout_proxy_nz"], rel=1e-9, abs=1e-9
    )
    assert recovered["inventory_nv_cost_cu2"] == pytest.approx(
        row["inventory_nv_cost_cu2"], rel=1e-9, abs=1e-9
    )
    assert recovered["inventory_nv_cost_cu3"] == pytest.approx(
        row["inventory_nv_cost_cu3"], rel=1e-9, abs=1e-9
    )
    assert "inventory_nv_cost_rounded_cu2" not in recovered


def test_cooler_gate_lowers_holding_and_nv_when_stockouts_similar():
    y = np.array([0.0] * 90 + [5.0] * 10)
    loud = np.array([0.8] * 90 + [4.0] * 10)
    cool = np.array([0.1] * 90 + [4.0] * 10)
    loud_m = inv.inventory_cost_metrics(y, loud, include_rounded=False)
    cool_m = inv.inventory_cost_metrics(y, cool, include_rounded=False)
    assert cool_m["inventory_holding_proxy_zero"] < loud_m["inventory_holding_proxy_zero"]
    assert cool_m["inventory_nv_cost_cu1"] < loud_m["inventory_nv_cost_cu1"]
    assert cool_m["inventory_stockout_proxy_nz"] == pytest.approx(
        loud_m["inventory_stockout_proxy_nz"]
    )


def test_rescore_multihorizon_rank_helper():
    """Synthetic MH payload ranks by combined ops at mean horizon."""
    def _overall(under_nz, hold_z, pi=0.12, n=1000):
        n_nz = int(pi * n)
        mean_y_nz = 8.0
        mean_actual = pi * mean_y_nz
        # bias_nz = over_nz - under_nz; mae = under+over. Set over_nz=1.
        over_nz = 1.0
        mae_nz = under_nz + over_nz
        bias_nz = over_nz - under_nz
        mean_yhat_nz = bias_nz + mean_y_nz
        mean_final = pi * mean_yhat_nz + (1 - pi) * hold_z
        return {
            "n_rows": n,
            "n_nonzero": n_nz,
            "mae_nonzero": mae_nz,
            "bias_nonzero": bias_nz,
            "mean_final": mean_final,
            "mean_actual": mean_actual,
            "underforecast_rate_nonzero": 0.5,
            "iwmae_rounded": under_nz,
        }

    models = {
        "cool_gate": {
            "by_horizon": {"1": {"overall": _overall(6.5, 0.5)}},
            "mean_1_to_H": {"overall": _overall(6.5, 0.5)},
        },
        "hot_fill": {
            "by_horizon": {"1": {"overall": _overall(5.5, 1.5)}},
            "mean_1_to_H": {"overall": _overall(5.5, 1.5)},
        },
    }
    by_rev = inv.rank_multihorizon_ops(
        models, horizon="mean", sort_key="sales_revenue_loss_units"
    )
    assert by_rev[0]["model"] == "hot_fill"
    by_hold = inv.rank_multihorizon_ops(
        models, horizon="mean", sort_key="inventory_holding_cost_zero"
    )
    assert by_hold[0]["model"] == "cool_gate"


def test_decision_economics_crossover_and_margins():
    """TST (high U, low H) vs DS (low U, high H): crossover + regime winners."""
    # Synthetic kpi summaries already in inventory_cost_from_kpi_summary shape
    def _inv(u_day, h_day):
        return {
            "inventory_mean_under": u_day,
            "inventory_holding_cost_zero_per_day": h_day,
            "sales_revenue_loss_units": u_day / 0.12,
            "inventory_holding_cost_zero": h_day / 0.88,
        }

    tst = _inv(0.80, 0.70)
    ds = _inv(0.70, 1.10)
    r_star = inv.pairwise_crossover_r(0.80, 0.70, 0.70, 1.10)
    assert r_star == pytest.approx((1.10 - 0.70) / (0.80 - 0.70))
    low = inv.select_model_by_r(
        {"TST": {"U": 0.80, "H": 0.70}, "DS": {"U": 0.70, "H": 1.10}},
        r=r_star * 0.5,
    )
    high = inv.select_model_by_r(
        {"TST": {"U": 0.80, "H": 0.70}, "DS": {"U": 0.70, "H": 1.10}},
        r=r_star * 2.0,
    )
    assert low["winner"] == "TST"
    assert high["winner"] == "DS"

    # Shared policy C_hold; only margin (hence r) differs across regimes.
    regimes = inv.margin_regimes_from_policy(
        holding_cost_per_unit=0.10, margins=(0.08, 0.25, 0.55)
    )
    report = inv.decision_economics_report(
        {"TST lite": tst, "plain DS": ds},
        pair=("TST lite", "plain DS"),
        margin_regimes={
            "low_margin": regimes["low_margin"],
            "high_margin": regimes["high_margin"],
        },
    )
    assert report["crossover"]["r_star"] == pytest.approx(r_star)
    assert report["margin_regimes"]["low_margin"]["holding_cost_per_unit"] == 0.10
    assert report["margin_regimes"]["high_margin"]["holding_cost_per_unit"] == 0.10
    assert report["margin_regimes"]["low_margin"]["cost_ratio_r"] == pytest.approx(0.8)
    assert report["margin_regimes"]["high_margin"]["cost_ratio_r"] == pytest.approx(5.5)
    assert report["margin_regimes"]["low_margin"]["winner"] == "TST lite"
    assert report["margin_regimes"]["high_margin"]["winner"] == "plain DS"
    pl_hi_ds = report["margin_regimes"]["high_margin"]["profit_loss_by_model"][
        "plain DS"
    ]
    pl_hi_tst = report["margin_regimes"]["high_margin"]["profit_loss_by_model"][
        "TST lite"
    ]
    assert (
        pl_hi_ds["total_profit_loss_per_day"]
        < pl_hi_tst["total_profit_loss_per_day"]
    )


def test_margin_regimes_from_policy_shared_hold():
    """Policy C_hold is shared; r increases only with margin."""
    regimes = inv.margin_regimes_from_policy(
        holding_cost_per_unit=0.10, margins=(0.08, 0.25, 0.55)
    )
    holds = {v["holding_cost_per_unit"] for v in regimes.values()}
    assert holds == {0.10}
    rs = [
        inv.cost_ratio_from_margin(v["margin"], v["holding_cost_per_unit"])
        for v in regimes.values()
    ]
    assert rs == pytest.approx([0.8, 2.5, 5.5])
    assert rs[0] < rs[1] < rs[2]


def test_model_ops_cost_and_pi_proxy():
    """C_model tiers / train-time normalize; π ranks by revenue − inv − ops."""
    assert inv.model_ops_cost_from_tiers("LightGBM", base_per_day=0.01) == pytest.approx(
        0.01
    )
    assert inv.model_ops_cost_from_tiers("plain DS", base_per_day=0.01) == pytest.approx(
        0.015
    )
    assert inv.model_ops_cost_from_tiers("TFT lite", base_per_day=0.01) == pytest.approx(
        0.02
    )
    costs = inv.model_ops_cost_from_train_seconds(
        {"fast": 10.0, "slow": 20.0}, base_per_day=0.01
    )
    assert costs["fast"] == pytest.approx(0.01)
    assert costs["slow"] == pytest.approx(0.02)

    inv_a = {
        "inventory_mean_under": 0.5,
        "inventory_holding_cost_zero_per_day": 1.0,
        "mean_actual": 2.0,
    }
    inv_b = {
        "inventory_mean_under": 0.4,
        "inventory_holding_cost_zero_per_day": 1.2,
        "mean_actual": 2.0,
    }
    # Same C_model → ranking by inv loss alone.
    pa = inv.profit_with_model_ops(
        inv_a,
        margin=0.25,
        holding_cost_per_unit=0.10,
        model_ops_cost_per_day=0.01,
        mean_demand_per_day=2.0,
    )
    pb = inv.profit_with_model_ops(
        inv_b,
        margin=0.25,
        holding_cost_per_unit=0.10,
        model_ops_cost_per_day=0.01,
        mean_demand_per_day=2.0,
    )
    # A: inv = 0.25*0.5 + 0.1*1.0 = 0.225; π = 0.5 - 0.225 - 0.01 = 0.265
    # B: inv = 0.25*0.4 + 0.1*1.2 = 0.22;  π = 0.5 - 0.22 - 0.01 = 0.27
    assert pa["pi_per_day"] == pytest.approx(0.265)
    assert pb["pi_per_day"] == pytest.approx(0.27)
    assert pb["pi_per_day"] > pa["pi_per_day"]

