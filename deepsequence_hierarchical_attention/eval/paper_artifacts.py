"""Load locked paper artifacts and build comparison tables/figures.

Used by ``examples/reproduce_paper_findings.ipynb`` and
``paper_figures/make_weekly_daily_direct_compare.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Matched lead times: weekly weeks ↔ ≈ daily days
MATCHED_LEADS = (
    ("1w≈7d", "1", "7"),
    ("4w≈28d", "4", "28"),
    ("8w≈56d", "8", "56"),
)

DEFAULT_PATHS = {
    "weekly_mh": "ab_runs/weekly/weekly_mh8_locked800_s42.json",
    "daily_direct_mh": "ab_runs/weekly/daily_direct_mh60_locked800_s42.json",
    "zero_rate": "ab_runs/weekly/zero_rate_daily_vs_weekly_locked800.json",
    "strata_daily_direct": "ab_runs/weekly/strata_daily_direct_s42.json",
    "strata_weekly_direct": "ab_runs/weekly/strata_weekly_direct_s42.json",
    # Appendix E only — not primary Results tables
    "daily_recursive_mh": "ab_runs/reclaim/daily_mh_1_60_level1_cross_off_all_models.json",
    "cummae_daily": "ab_runs/reclaim/cummae_daily_s42.json",
    "daily_multiseed": "ab_runs/reclaim/multiseed/daily_multiseed_long_loyalty_summary.json",
    "carparts_multiseed": "ab_runs/reclaim/multiseed/carparts_multiseed_long_loyalty_summary.json",
}

# Primary PAPER.md figures (artifact-fast regen targets)
PRIMARY_FIGURES = {
    "D1": "paper_figures/fig_daily_direct_iwmae_horizon.png",
    "D2": "paper_figures/fig_daily_direct_cummae_horizon.png",
    "D3": "paper_figures/fig_daily_direct_strata_iwmae.png",
    "W1": "paper_figures/fig_zero_rate_daily_vs_weekly.png",
    "W2": "paper_figures/fig_weekly_daily_direct_iwmae.png",
    "W3": "paper_figures/fig_weekly_daily_direct_cummae.png",
    "W4": "paper_figures/fig_forecast_weekly_onestep.png",
    "W5": "paper_figures/fig_forecast_weekly_direct.png",
    "W6": "paper_figures/fig_weekly_direct_strata_iwmae.png",
    "m5": "paper_figures/fig_m5_architecture.png",
}

STRATA_TABLE_KEYS = {
    "D-S1": ("strata_daily_direct", "daily_direct_mean_demand_table_iwmae"),
    "W-S1": ("strata_weekly_direct", "weekly_direct_mean_demand_table_iwmae"),
    "W-S2": ("strata_weekly_direct", "weekly_direct_zero_rate_table_iwmae"),
}


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return json.loads(p.read_text())


def _horizon_block(payload: dict, h: str, *, cum: bool = False) -> dict:
    key = "by_horizon_cum" if cum else "by_horizon"
    block = payload[key][str(h)]
    if isinstance(block, dict) and "overall" in block and isinstance(block["overall"], dict):
        return block["overall"]
    return block


def metric_at(
    model_payload: dict,
    h: str,
    field: str = "iwmae_rounded",
    *,
    cum: bool = False,
) -> float | None:
    try:
        block = _horizon_block(model_payload, h, cum=cum)
        val = block.get(field)
        return None if val is None else float(val)
    except (KeyError, TypeError, ValueError):
        return None


def bakeoff_table(
    artifact: dict,
    horizons: list[str] | tuple[str, ...],
    models: tuple[str, ...] = ("deepsequence", "tsb", "lightgbm"),
    *,
    cum: bool = False,
    field: str | None = None,
) -> list[dict[str, Any]]:
    """Rows: model × horizon metrics from a weekly_mh-style JSON."""
    field = field or ("cummae_rounded" if cum else "iwmae_rounded")
    rows = []
    for name in models:
        payload = artifact["models"][name]
        row: dict[str, Any] = {
            "model": name,
            "method": payload.get("method"),
        }
        for h in horizons:
            row[f"h{h}"] = metric_at(payload, h, field, cum=cum)
        rows.append(row)
    return rows


def like_for_like_direct_table(
    weekly: dict | None = None,
    daily: dict | None = None,
) -> list[dict[str, Any]]:
    """Side-by-side weekly direct vs daily direct at matched leads."""
    weekly = weekly or load_json(DEFAULT_PATHS["weekly_mh"])
    daily = daily or load_json(DEFAULT_PATHS["daily_direct_mh"])
    models = ("deepsequence", "tsb", "lightgbm")
    rows = []
    for name in models:
        w = weekly["models"][name]
        d = daily["models"][name]
        for label, hw, hd in MATCHED_LEADS:
            rows.append(
                {
                    "model": name,
                    "lead": label,
                    "weekly_h": int(hw),
                    "daily_h": int(hd),
                    "weekly_iwmae": metric_at(w, hw, "iwmae_rounded"),
                    "daily_iwmae": metric_at(d, hd, "iwmae_rounded"),
                    "weekly_cummae": metric_at(w, hw, "cummae_rounded", cum=True),
                    "daily_cummae": metric_at(d, hd, "cummae_rounded", cum=True),
                    "weekly_method": w.get("method"),
                    "daily_method": d.get("method"),
                }
            )
    return rows


def zero_rate_summary(zr: dict | None = None) -> dict[str, Any]:
    zr = zr or load_json(DEFAULT_PATHS["zero_rate"])
    return {
        "daily_zero_rate": float(zr["daily"]["zero_rate"]),
        "weekly_zero_rate": float(zr["weekly"]["zero_rate"]),
        "delta_zero_rate": float(zr.get("delta_zero_rate", np.nan)),
        "daily_mean_demand": float(zr["daily"].get("mean_demand", np.nan)),
        "weekly_mean_demand": float(zr["weekly"].get("mean_demand", np.nan)),
    }


def strata_table(
    table_id: str = "D-S1",
    artifact: dict | None = None,
    *,
    horizons: list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Load a PAPER.md zone-strata table (D-S1, W-S1, or W-S2).

    Returns rows with horizon, zone, deepsequence, tsb, lightgbm, best.
    """
    if table_id not in STRATA_TABLE_KEYS:
        raise KeyError(f"Unknown strata table_id={table_id!r}; expected one of {sorted(STRATA_TABLE_KEYS)}")
    path_key, json_key = STRATA_TABLE_KEYS[table_id]
    payload = artifact if artifact is not None else load_json(DEFAULT_PATHS[path_key])
    rows_src = payload[json_key]
    out: list[dict[str, Any]] = []
    for r in rows_src:
        h = int(r["horizon"])
        if horizons is not None and h not in horizons:
            continue
        out.append(
            {
                "horizon": h,
                "zone": r["zone"],
                "deepsequence": float(r["deepsequence"]),
                "tsb": float(r["tsb"]),
                "lightgbm": float(r["lightgbm"]),
                "best": r.get("best"),
            }
        )
    return out


def recursive_bakeoff_table(
    artifact: dict | None = None,
    horizons: list[str] | tuple[str, ...] = ("1", "7", "14", "28", "60"),
    models: tuple[str, ...] = ("deepsequence", "temporal_transformer", "lightgbm"),
    field: str = "iwmae_rounded",
) -> list[dict[str, Any]]:
    """Appendix-E recursive daily bake-off (not a primary Results table)."""
    rec = artifact or load_json(DEFAULT_PATHS["daily_recursive_mh"])
    out = []
    for m in models:
        row: dict[str, Any] = {
            "model": m,
            "method": rec["models"][m].get("method", "recursive"),
        }
        for h in horizons:
            block = rec["models"][m]["by_horizon"][str(h)]
            if "overall" in block and isinstance(block["overall"], dict):
                block = block["overall"]
            val = block.get(field)
            row[f"h{h}"] = None if val is None else float(val)
        out.append(row)
    return out


def multiseed_iwmae_pivot(
    artifact: dict | None = None,
    horizons: list[str] | tuple[str, ...] = ("1", "7", "14", "28", "60"),
    models: tuple[str, ...] = ("deepsequence", "temporal_transformer", "lightgbm"),
) -> list[dict[str, Any]]:
    """Appendix-E multi-seed IWMAE mean rows (long form)."""
    ms = artifact or load_json(DEFAULT_PATHS["daily_multiseed"])
    block = ms["iwmae_mean_std"]
    rows = []
    for h in horizons:
        for m in models:
            e = block[str(h)][m]
            rows.append(
                {
                    "h": int(h),
                    "model": m,
                    "mean": float(e["mean"]),
                    "std": float(e["std"]),
                }
            )
    return rows


def figure_path(label: str) -> Path:
    """Resolve a primary figure path by label (D1, W1, m5, …)."""
    if label not in PRIMARY_FIGURES:
        raise KeyError(f"Unknown figure label={label!r}; expected one of {sorted(PRIMARY_FIGURES)}")
    return repo_path(PRIMARY_FIGURES[label])


def format_markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append("—" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
