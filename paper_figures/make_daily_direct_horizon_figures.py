#!/usr/bin/env python3
"""Primary daily Direct-MH horizon figures (IWMAE + CumMAE).

Reads:
  ab_runs/weekly/daily_direct_mh60_locked800_s42.json

Writes under paper_figures/:
  fig_daily_direct_iwmae_horizon.png/.pdf
  fig_daily_direct_cummae_horizon.png/.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

C_DS = "#0072B2"
C_LGBM = "#009E73"
C_TSB = "#CC79A7"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    }
)

MODELS = [
    ("DeepSequence", "deepsequence", C_DS, "o"),
    ("TSB", "tsb", C_TSB, "D"),
    ("LightGBM", "lightgbm", C_LGBM, "^"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def _point(block: dict, key: str) -> float:
    if key in block and block[key] is not None:
        return float(block[key])
    if "overall" in block and isinstance(block["overall"], dict):
        return float(block["overall"][key])
    raise KeyError(key)


def series(payload: dict, model: str, horizons: list[int], *, cum: bool = False) -> np.ndarray:
    m = payload["models"][model]
    by = m["by_horizon_cum" if cum else "by_horizon"]
    key = "cummae_rounded" if cum else "iwmae_rounded"
    return np.asarray([_point(by[str(h)], key) for h in horizons], dtype=float)


def fig_horizon(payload: dict, *, cum: bool, stem: str, ylabel: str, title: str) -> None:
    horizons = list(payload["config"]["report_horizons"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for label, key, color, marker in MODELS:
        vals = series(payload, key, horizons, cum=cum)
        ax.plot(
            horizons,
            vals,
            label=label,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.8,
        )
    ax.set_xlabel(r"Horizon / lead time $h$ (days)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(horizons)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    save(fig, stem)


def main() -> None:
    path = ROOT / "ab_runs/weekly/daily_direct_mh60_locked800_s42.json"
    payload = load(path)
    fig_horizon(
        payload,
        cum=False,
        stem="fig_daily_direct_iwmae_horizon",
        ylabel="IWMAE (rounded)",
        title="Daily Direct-MH IWMAE vs horizon (seed 42, locked 800)",
    )
    fig_horizon(
        payload,
        cum=True,
        stem="fig_daily_direct_cummae_horizon",
        ylabel="CumMAE (rounded)",
        title="Daily Direct-MH CumMAE vs horizon (seed 42, locked 800)",
    )


if __name__ == "__main__":
    main()
