#!/usr/bin/env python3
"""Re-score multi-horizon bake-off JSONs with sales-loss + holding ops costs.

Usage (from repo root)::

    python ab_runs/rescore_inventory_mh.py
    python ab_runs/rescore_inventory_mh.py \\
        --in ab_runs/recompare_retuned/daily_mh14_locked_all.json \\
        --out ab_runs/reclaim/daily_mh14_ops_ranking.json
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
    spec = importlib.util.spec_from_file_location("ds_inventory_metrics_mh", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ds_inventory_metrics_mh"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="in_path",
        default="ab_runs/recompare_retuned/daily_mh14_locked_all.json",
    )
    ap.add_argument(
        "--out",
        default="ab_runs/reclaim/daily_mh14_ops_ranking.json",
    )
    args = ap.parse_args()
    inv = _load_inventory_metrics()

    payload = json.loads((ROOT / args.in_path).read_text(encoding="utf-8"))
    models = payload.get("models") or {}

    # Discover horizons from first model that has by_horizon.
    discovered: list[str] = []
    for payload_m in models.values():
        keys = list((payload_m.get("by_horizon") or {}).keys())
        if keys:
            discovered = sorted(keys, key=lambda x: int(x) if str(x).isdigit() else x)
            break
    horizons = tuple(discovered + (["mean"] if discovered else ["mean"]))

    sort_keys = (
        "iwmae_rounded",
        "sales_revenue_loss_units",
        "inventory_holding_cost_zero",
        inv.PRIMARY_COMBINED_OPS_METRIC,
        "combined_ops_cost_h0p25",
        "combined_ops_cost_h1",
    )

    by_horizon = {}
    for h in horizons:
        by_horizon[h] = {}
        for sk in sort_keys:
            by_horizon[h][sk] = inv.rank_multihorizon_ops(
                models, horizon=h, sort_key=sk
            )

    # Compact print table for mean / h=1,7,14 under primary combined
    print_keys = (
        "sales_revenue_loss_units",
        "inventory_holding_cost_zero",
        "combined_ops_cost_h0p1",
        "iwmae_rounded",
    )
    for h in horizons:
        rows = by_horizon[h][inv.PRIMARY_COMBINED_OPS_METRIC]
        print(f"\n[h={h}] ranked by {inv.PRIMARY_COMBINED_OPS_METRIC}")
        hdr = f"{'model':28}" + "".join(f" {k:>12}" for k in print_keys)
        print(hdr)
        for r in rows:
            line = f"{r['model']:28}"
            for k in print_keys:
                v = r.get(k)
                line += f" {v:12.3f}" if isinstance(v, (int, float)) else f" {'—':>12}"
            print(line)

    out = {
        "source": args.in_path,
        "definition": {
            "sales_revenue_loss_units": "mean unmet demand on sale days (revenue loss)",
            "inventory_holding_cost_zero": "mean stock on no-sale days (carrying cost)",
            "combined_ops_cost_h": "rev_per_day + h * hold_per_day",
            "primary_combined": inv.PRIMARY_COMBINED_OPS_METRIC,
        },
        "by_horizon": by_horizon,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
