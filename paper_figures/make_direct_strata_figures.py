#!/usr/bin/env python3
"""Direct-MH zone strata figures (daily + weekly mean-demand terciles).

Reads:
  ab_runs/weekly/strata_daily_direct_s42.json
  ab_runs/weekly/strata_weekly_direct_s42.json

Writes under paper_figures/:
  fig_daily_direct_strata_iwmae.png/.pdf
  fig_weekly_direct_strata_iwmae.png/.pdf
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
        "legend.fontsize": 8.5,
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
    ("DeepSequence", "deepsequence", C_DS),
    ("TSB", "tsb", C_TSB),
    ("LightGBM", "lightgbm", C_LGBM),
]
ZONES = [("low", "Low"), ("mid", "Mid"), ("high", "High")]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def _table_lookup(rows: list[dict], h: int, zone: str, model: str) -> float:
    for r in rows:
        if int(r["horizon"]) == h and r["zone"] == zone:
            return float(r[model])
    raise KeyError((h, zone, model))


def fig_strata_bars(
    rows: list[dict],
    horizons: list[int],
    *,
    stem: str,
    title: str,
    xlabel: str,
) -> None:
    n_h = len(horizons)
    fig, axes = plt.subplots(1, n_h, figsize=(2.6 * n_h + 0.6, 3.6), sharey=False)
    if n_h == 1:
        axes = [axes]
    x = np.arange(len(ZONES))
    width = 0.24
    for ax, h in zip(axes, horizons):
        for i, (lab, key, color) in enumerate(MODELS):
            vals = [_table_lookup(rows, h, z, key) for z, _ in ZONES]
            ax.bar(
                x + (i - 1) * width,
                vals,
                width=width,
                label=lab,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([z[1] for z in ZONES])
        ax.set_title(rf"$h={h}$")
        ax.set_xlabel("Train mean-demand zone")
        if ax is axes[0]:
            ax.set_ylabel("IWMAE (rounded)")
            ax.legend(frameon=False, loc="best", fontsize=7.5)
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, stem)


def main() -> None:
    daily = load(ROOT / "ab_runs/weekly/strata_daily_direct_s42.json")
    weekly = load(ROOT / "ab_runs/weekly/strata_weekly_direct_s42.json")
    fig_strata_bars(
        daily["daily_direct_mean_demand_table_iwmae"],
        [1, 7, 28, 60],
        stem="fig_daily_direct_strata_iwmae",
        title="Daily Direct-MH IWMAE by train mean-demand zone (seed 42)",
        xlabel="Train mean-demand zone",
    )
    fig_strata_bars(
        weekly["weekly_direct_mean_demand_table_iwmae"],
        [1, 4, 8],
        stem="fig_weekly_direct_strata_iwmae",
        title="Weekly Direct-MH IWMAE by train mean-demand zone (seed 42)",
        xlabel="Train mean-demand zone",
    )


if __name__ == "__main__":
    main()
