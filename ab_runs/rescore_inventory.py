#!/usr/bin/env python3
"""Re-score locked H=1 bake-off JSONs with inventory / newsvendor costs.

Locked artifacts store aggregate KPIs only (no per-row y/yhat). Continuous
newsvendor costs are recovered exactly via ``inventory_cost_from_kpi_summary``.
Rounded nv costs require a fresh predict pass — see note in the output JSON.

Usage (from repo root)::

    python ab_runs/rescore_inventory.py
    python ab_runs/rescore_inventory.py --out ab_runs/reclaim/daily_h1_inventory_ranking.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_inventory_metrics():
    path = ROOT / "deepsequence_hierarchical_attention" / "inventory_metrics.py"
    spec = importlib.util.spec_from_file_location("ds_inventory_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ds_inventory_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


_inv = _load_inventory_metrics()
PRIMARY_INVENTORY_METRIC = _inv.PRIMARY_INVENTORY_METRIC
inventory_cost_from_kpi_summary = _inv.inventory_cost_from_kpi_summary

# Locked H=1 panel (data_seed=42, sku_list_daily_data42)
MODELS = (
    {
        "name": "TST lite",
        "model_key": "temporal_transformer",
        "source": "ab_runs/reclaim/daily_h1_hybrid_temporal.json",
    },
    {
        "name": "hybrid d64_b1_decouple",
        "model_key": "hybrid_d64_b1_decouple",
        "source": "ab_runs/reclaim/daily_h1_hybrid_d64_b1_decouple.json",
    },
    {
        "name": "plain DS",
        "model_key": "deepsequence",
        "source": "ab_runs/reclaim/daily_h1_hybrid_temporal.json",
    },
)


def _load_overall(source: str, model_key: str) -> dict:
    payload = json.loads((ROOT / source).read_text(encoding="utf-8"))
    return payload["models"][model_key]["overall"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="ab_runs/reclaim/daily_h1_inventory_ranking.json",
        help="Write ranking JSON here.",
    )
    args = ap.parse_args()

    rows = []
    for spec in MODELS:
        overall = _load_overall(spec["source"], spec["model_key"])
        inv = inventory_cost_from_kpi_summary(overall)
        row = {
            "name": spec["name"],
            "model_key": spec["model_key"],
            "source": spec["source"],
            "iwmae_rounded": overall.get("iwmae_rounded"),
            "mae_all": overall.get("mae_all"),
            "bias": overall.get("bias"),
            "mean_p": overall.get("mean_p"),
            "underforecast_rate_nonzero": overall.get("underforecast_rate_nonzero"),
            **inv,
        }
        rows.append(row)

    by_iwmae = sorted(
        rows,
        key=lambda r: (
            r["iwmae_rounded"] is None,
            r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
        ),
    )
    by_nv2 = sorted(
        rows,
        key=lambda r: (
            r.get("inventory_nv_cost_cu2") is None,
            r.get("inventory_nv_cost_cu2") if r.get("inventory_nv_cost_cu2") is not None else 1e9,
        ),
    )
    by_nv3 = sorted(
        rows,
        key=lambda r: (
            r.get("inventory_nv_cost_cu3") is None,
            r.get("inventory_nv_cost_cu3") if r.get("inventory_nv_cost_cu3") is not None else 1e9,
        ),
    )

    payload = {
        "panel": {
            "sku_list": "ab_runs/recompare/sku_list_daily_data42.json",
            "horizon": 1,
            "note": (
                "Rescored from locked aggregate JSONs (continuous yhat). "
                "Rounded inventory_nv_cost_rounded_* unavailable without per-row preds."
            ),
        },
        "primary_iwmae": "iwmae_rounded",
        "primary_inventory": "inventory_nv_cost_cu2",
        "recommended_future_primary": PRIMARY_INVENTORY_METRIC,
        "recommended_future_note": (
            "After re-eval with kpi_block, prefer inventory_nv_cost_rounded_cu2 "
            "(same newsvendor form on count-rounded forecasts)."
        ),
        "ranking_by_iwmae": [
            {"rank": i + 1, **r} for i, r in enumerate(by_iwmae)
        ],
        "ranking_by_nv_cu2": [
            {"rank": i + 1, **r} for i, r in enumerate(by_nv2)
        ],
        "ranking_by_nv_cu3": [
            {"rank": i + 1, **r} for i, r in enumerate(by_nv3)
        ],
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _fmt(v, digits=3):
        return f"{v:.{digits}f}" if v is not None else "n/a"

    print("=" * 88)
    print("H=1 inventory rescore (continuous yhat from locked aggregates)")
    print("=" * 88)
    header = (
        f"{'model':28s} {'IWMAE':>7s} {'NV cu1':>8s} {'NV cu2':>8s} {'NV cu3':>8s} "
        f"{'hold0':>7s} {'stock':>7s} {'fill':>6s} {'bias':>7s}"
    )
    print(header)
    print("-" * 88)
    for r in by_nv2:
        print(
            f"{r['name']:28s} "
            f"{_fmt(r['iwmae_rounded']):>7s} "
            f"{_fmt(r.get('inventory_nv_cost_cu1')):>8s} "
            f"{_fmt(r.get('inventory_nv_cost_cu2')):>8s} "
            f"{_fmt(r.get('inventory_nv_cost_cu3')):>8s} "
            f"{_fmt(r.get('inventory_holding_proxy_zero')):>7s} "
            f"{_fmt(r.get('inventory_stockout_proxy_nz')):>7s} "
            f"{_fmt(r.get('inventory_fill_rate_nz')):>6s} "
            f"{_fmt(r.get('bias'), 3):>7s}"
        )
    print()
    print("Rank by IWMAE:     ", " > ".join(r["name"] for r in by_iwmae))
    print("Rank by NV cu/co=2:", " > ".join(r["name"] for r in by_nv2))
    print("Rank by NV cu/co=3:", " > ".join(r["name"] for r in by_nv3))
    print(
        "\nLimitation: rounded nv costs need per-row preds; "
        f"future bake-offs should sort on {PRIMARY_INVENTORY_METRIC}."
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
