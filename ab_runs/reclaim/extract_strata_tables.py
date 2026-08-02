#!/usr/bin/env python3
"""Extract stratified IWMAE / CumMAE tables from locked MH bake-off JSON.

Daily recursive MH already nests volume terciles (train sum Quantity) under
``by_horizon[h]/{overall,low,mid,high}``. Weekly Direct-MH (after strata hook)
adds train **mean-demand** terciles (primary) and train **zero-rate** terciles
(secondary: high_zero / mid / low_zero).

Usage::

  .venv-test/bin/python ab_runs/reclaim/extract_strata_tables.py \\
    --daily_json ab_runs/reclaim/daily_mh_1_60_cummae_s42.json \\
    --weekly_json ab_runs/weekly/weekly_mh8_locked800_s42.json \\
    --out_json ab_runs/reclaim/strata_volume_s42.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_ALIASES = {
    "deepsequence": "DeepSequence",
    "temporal_transformer": "TST",
    "tft_lite": "TFT",
    "deepar_lite": "DeepAR",
    "lightgbm": "LightGBM",
    "tsb": "TSB",
}


def _block(node: dict | None, band: str = "overall") -> dict:
    if not isinstance(node, dict):
        return {}
    if band in node and isinstance(node[band], dict):
        return node[band]
    if band == "overall":
        # Flat kpi_block (weekly pre-strata) or already-overall.
        if "iwmae_rounded" in node or "cummae_rounded" in node:
            return node
        return node.get("overall", {})
    return {}


def _metric(block: dict, key: str):
    if key in block and block[key] is not None:
        return block[key]
    aliases = {
        "cummae_rounded": ("mae_all_rounded", "cummae"),
        "iwmae_rounded": ("iwmae",),
    }
    for a in aliases.get(key, ()):
        if a in block and block[a] is not None:
            return block[a]
    return None


def extract_protocol(
    payload: dict,
    *,
    models: list[str],
    horizons: list[int],
    bands: list[str],
    strata_key: str | None = None,
    label: str,
) -> dict:
    """Pull IWMAE + CumMAE by band × horizon × model."""
    out_models = {}
    for name in models:
        m = (payload.get("models") or {}).get(name)
        if m is None:
            continue
        by_h = m.get("by_horizon") or {}
        by_c = m.get("by_horizon_cum") or {}
        rows = {}
        for h in horizons:
            hk = str(h)
            point = by_h.get(hk) or {}
            cum = by_c.get(hk) or {}
            band_rows = {}
            for band in bands:
                if strata_key and strata_key in point:
                    pb = _block(point.get(strata_key), band)
                    cb = _block(cum.get(strata_key), band) if strata_key in cum else _block(cum, band)
                else:
                    pb = _block(point, band)
                    cb = _block(cum, band)
                band_rows[band] = {
                    "iwmae_rounded": _metric(pb, "iwmae_rounded"),
                    "cummae_rounded": _metric(cb, "cummae_rounded"),
                    "n_rows": pb.get("n_rows"),
                    "n_skus_in_pred": pb.get("n_skus_in_pred"),
                    "zero_rate": pb.get("zero_rate"),
                }
            # Best IWMAE among requested models for this band (filled later).
            rows[hk] = band_rows
        out_models[name] = rows

    # Winners per horizon × band
    winners = {}
    for h in horizons:
        hk = str(h)
        winners[hk] = {}
        for band in bands:
            scored = []
            for name, rows in out_models.items():
                v = rows.get(hk, {}).get(band, {}).get("iwmae_rounded")
                if v is not None:
                    scored.append((name, v))
            if not scored:
                winners[hk][band] = None
                continue
            scored.sort(key=lambda t: t[1])
            winners[hk][band] = {
                "best": scored[0][0],
                "best_iwmae_rounded": scored[0][1],
                "ranking": [{"model": n, "iwmae_rounded": v} for n, v in scored],
            }

    cfg = payload.get("config") or {}
    return {
        "label": label,
        "zone_definition": cfg.get("zone_definition"),
        "volume_stats": cfg.get("volume_stats"),
        "mean_demand_stats": cfg.get("mean_demand_stats"),
        "zero_rate_stats": cfg.get("zero_rate_stats"),
        "models": out_models,
        "winners_iwmae": winners,
        "config_snippet": {
            k: cfg.get(k)
            for k in (
                "protocol",
                "dataset",
                "n_skus",
                "seed",
                "train_seed",
                "data_seed",
                "horizon",
                "report_horizons",
                "n_origins",
                "sku_list",
            )
            if k in cfg or cfg.get(k) is not None
        },
    }


def paper_table(protocol: dict, models: list[str], horizons: list[int], bands: list[str]) -> list[dict]:
    """Flatten to rows suitable for markdown tables."""
    rows = []
    for h in horizons:
        for band in bands:
            row = {"horizon": h, "zone": band}
            best = None
            best_v = None
            for name in models:
                v = (
                    protocol.get("models", {})
                    .get(name, {})
                    .get(str(h), {})
                    .get(band, {})
                    .get("iwmae_rounded")
                )
                row[name] = v
                if v is not None and (best_v is None or v < best_v):
                    best, best_v = name, v
            row["best"] = best
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--daily_json",
        default="ab_runs/reclaim/daily_mh_1_60_cummae_s42.json",
    )
    p.add_argument(
        "--weekly_json",
        default="ab_runs/weekly/weekly_mh8_locked800_s42.json",
    )
    p.add_argument(
        "--daily_direct_json",
        default="",
        help="Optional daily Direct-MH JSON with strata.",
    )
    p.add_argument("--out_json", default="ab_runs/reclaim/strata_volume_s42.json")
    args = p.parse_args()

    daily = json.loads(Path(args.daily_json).read_text(encoding="utf-8"))
    daily_models = [
        "deepsequence",
        "temporal_transformer",
        "tft_lite",
        "deepar_lite",
        "lightgbm",
    ]
    daily_h = [1, 7, 14, 28, 60]
    daily_proto = extract_protocol(
        daily,
        models=daily_models,
        horizons=daily_h,
        bands=["overall", "low", "mid", "high"],
        label="daily_recursive_volume_sum_terciles_s42",
    )
    # Document: locked daily MH used train sum(Quantity) terciles.
    daily_proto["zone_definition"] = {
        "primary": "train_volume_terciles (sum Quantity → low/mid/high volume)",
        "note": (
            "Locked daily recursive bake-off bands (no test leakage). "
            "Spearman(sum, mean demand) ≈ 0.95 on equal-ish panels; "
            "prefer mean-demand terciles for new runs when lengths vary."
        ),
        "no_test_leakage": True,
        "source_json": args.daily_json,
    }

    out = {
        "metric_primary": "iwmae_rounded",
        "metric_secondary": "cummae_rounded",
        "daily_recursive": daily_proto,
        "daily_recursive_table_iwmae": paper_table(
            daily_proto, daily_models, daily_h, ["low", "mid", "high"]
        ),
        "model_aliases": MODEL_ALIASES,
    }

    wpath = Path(args.weekly_json)
    if wpath.exists():
        weekly = json.loads(wpath.read_text(encoding="utf-8"))
        weekly_models = ["deepsequence", "tsb", "lightgbm"]
        weekly_h = [1, 4, 8]
        # Primary: mean-demand (top-level low/mid/high after strata hook)
        has_mean = any(
            "strata_mean_demand" in ((weekly.get("models") or {}).get(m, {}).get("by_horizon") or {}).get("1", {})
            for m in weekly_models
            if m in (weekly.get("models") or {})
        )
        weekly_mean = extract_protocol(
            weekly,
            models=weekly_models,
            horizons=weekly_h,
            bands=["overall", "low", "mid", "high"],
            strata_key="strata_mean_demand" if has_mean else None,
            label="weekly_direct_mh_mean_demand_terciles_s42",
        )
        if not weekly_mean.get("zone_definition"):
            weekly_mean["zone_definition"] = weekly.get("config", {}).get("zone_definition")
        out["weekly_direct_mean_demand"] = weekly_mean
        out["weekly_direct_mean_demand_table_iwmae"] = paper_table(
            weekly_mean, weekly_models, weekly_h, ["low", "mid", "high"]
        )

        has_zr = any(
            "strata_zero_rate" in ((weekly.get("models") or {}).get(m, {}).get("by_horizon") or {}).get("1", {})
            for m in weekly_models
            if m in (weekly.get("models") or {})
        )
        if has_zr:
            weekly_zr = extract_protocol(
                weekly,
                models=weekly_models,
                horizons=weekly_h,
                bands=["overall", "high_zero", "mid", "low_zero"],
                strata_key="strata_zero_rate",
                label="weekly_direct_mh_zero_rate_terciles_s42",
            )
            out["weekly_direct_zero_rate"] = weekly_zr
            out["weekly_direct_zero_rate_table_iwmae"] = paper_table(
                weekly_zr,
                weekly_models,
                weekly_h,
                ["high_zero", "mid", "low_zero"],
            )

    if args.daily_direct_json:
        dpath = Path(args.daily_direct_json)
        if dpath.exists():
            dd = json.loads(dpath.read_text(encoding="utf-8"))
            dd_models = ["deepsequence", "tsb", "lightgbm"]
            dd_h = [1, 7, 14, 28, 56, 60]
            has_mean = any(
                "strata_mean_demand"
                in ((dd.get("models") or {}).get(m, {}).get("by_horizon") or {}).get("1", {})
                for m in dd_models
                if m in (dd.get("models") or {})
            )
            dd_proto = extract_protocol(
                dd,
                models=dd_models,
                horizons=dd_h,
                bands=["overall", "low", "mid", "high"],
                strata_key="strata_mean_demand" if has_mean else None,
                label="daily_direct_mh_mean_demand_terciles_s42",
            )
            out["daily_direct_mean_demand"] = dd_proto
            out["daily_direct_mean_demand_table_iwmae"] = paper_table(
                dd_proto, dd_models, dd_h, ["low", "mid", "high"]
            )

    path = Path(args.out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")

    # Quick console summary: DS vs best baseline by zone
    print("\nDaily recursive (volume terciles) — best IWMAE by zone:")
    for row in out["daily_recursive_table_iwmae"]:
        if row["horizon"] in (1, 28, 60):
            print(
                f"  h={row['horizon']} {row['zone']}: best={row['best']} "
                f"DS={row.get('deepsequence'):.3f} TST={row.get('temporal_transformer'):.3f}"
            )


if __name__ == "__main__":
    main()
