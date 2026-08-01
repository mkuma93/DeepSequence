#!/usr/bin/env python3
"""Extract CumMAE comparison tables from MH JSON that already has by_horizon_cum,
or run a short locked-protocol recompute and write reclaim CumMAE artifacts.

Usage (after MH eval with CumMAE hook)::

  .venv-test/bin/python ab_runs/reclaim/extract_cummae_tables.py \\
    --mh_json ab_runs/reclaim/daily_mh_....json \\
    --out_json ab_runs/reclaim/cummae_daily_s42.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_cummae(payload: dict) -> dict:
    models = payload.get("models") or {}
    report = (
        (payload.get("config") or {}).get("report_horizons")
        or sorted(
            {
                int(h)
                for m in models.values()
                for h in (m.get("by_horizon_cum") or {})
            }
        )
    )
    by_h = {}
    for h in report:
        key = str(h)
        rows = []
        for name, m in models.items():
            cum = (m.get("by_horizon_cum") or {}).get(key)
            if cum is None:
                continue
            # daily nests overall; carparts is flat kpi_block
            block = cum.get("overall", cum) if isinstance(cum, dict) else {}
            point = (m.get("by_horizon") or {}).get(key)
            if isinstance(point, dict) and "overall" in point:
                point = point["overall"]
            rows.append(
                {
                    "model": name,
                    "cummae": block.get("cummae"),
                    "cummae_rounded": block.get("cummae_rounded"),
                    "cum_iwmae": block.get("cum_iwmae"),
                    "cum_iwmae_rounded": block.get("cum_iwmae_rounded"),
                    "iwmae_rounded": (point or {}).get("iwmae_rounded"),
                }
            )
        rows = sorted(
            rows,
            key=lambda r: (
                r["cummae_rounded"] is None,
                r["cummae_rounded"] if r["cummae_rounded"] is not None else 1e9,
            ),
        )
        by_h[key] = rows
    return {
        "source": payload.get("config"),
        "metric": "CumMAE(H)=mean|sum_{h=1..H} yhat - sum y|",
        "primary_rank_unchanged": "iwmae_rounded (pointwise)",
        "by_horizon": by_h,
        "comparison_cum": payload.get("comparison_cum"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mh_json", required=True)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    payload = json.loads(Path(args.mh_json).read_text(encoding="utf-8"))
    out = extract_cummae(payload)
    path = Path(args.out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    for h, rows in out["by_horizon"].items():
        print(f"\n[h={h}]")
        for r in rows:
            print(
                f"  {r['model']:28s} cummae={r.get('cummae_rounded')} "
                f"cum_iwmae={r.get('cum_iwmae_rounded')} "
                f"iwmae={r.get('iwmae_rounded')}"
            )


if __name__ == "__main__":
    main()
