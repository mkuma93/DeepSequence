#!/usr/bin/env python3
"""Like-for-like weekly vs daily *direct* MH comparison figures.

Reads:
  ab_runs/weekly/weekly_mh8_locked800_s42.json
  ab_runs/weekly/daily_direct_mh60_locked800_s42.json
  ab_runs/weekly/zero_rate_daily_vs_weekly_locked800.json

Writes under paper_figures/:
  fig_weekly_daily_direct_iwmae.png/.pdf
  fig_weekly_daily_direct_cummae.png/.pdf
  fig_zero_rate_daily_vs_weekly.png/.pdf
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

# Matched lead times: weekly weeks ↔ daily days
MATCHED = [
    ("1w / 7d", "1", "7"),
    ("4w / 28d", "4", "28"),
    ("8w / 56d", "8", "56"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def _metric(payload: dict, h: str, key: str, cum: bool = False) -> float:
    block = payload["by_horizon_cum" if cum else "by_horizon"][str(h)]
    # overall may be nested or flat depending on kpi_block usage
    if "overall" in block and isinstance(block["overall"], dict):
        block = block["overall"]
    return float(block[key])


def fig_iwmae_compare(weekly: dict, daily: dict) -> None:
    models = [
        ("DeepSequence", "deepsequence", C_DS),
        ("TSB", "tsb", C_TSB),
        ("LightGBM", "lightgbm", C_LGBM),
    ]
    labels = [m[0] for m in MATCHED]
    x = np.arange(len(MATCHED))
    width = 0.12

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=False)
    for ax, grain, src, h_keys in [
        (axes[0], "Weekly (direct MH)", weekly, [m[1] for m in MATCHED]),
        (axes[1], "Daily (direct MH)", daily, [m[2] for m in MATCHED]),
    ]:
        for i, (lab, key, color) in enumerate(models):
            vals = [_metric(src["models"][key], h, "iwmae_rounded") for h in h_keys]
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
        ax.set_xticklabels(labels)
        ax.set_ylabel("IWMAE (rounded)")
        ax.set_title(grain)
        ax.legend(frameon=False, loc="best")

    fig.suptitle(
        "Like-for-like direct MH: weekly vs daily (seed 42, locked 800)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "fig_weekly_daily_direct_iwmae")


def fig_cummae_compare(weekly: dict, daily: dict) -> None:
    models = [
        ("DeepSequence", "deepsequence", C_DS),
        ("TSB", "tsb", C_TSB),
        ("LightGBM", "lightgbm", C_LGBM),
    ]
    labels = [m[0] for m in MATCHED]
    x = np.arange(len(MATCHED))
    width = 0.12

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=False)
    for ax, grain, src, h_keys in [
        (axes[0], "Weekly CumMAE (direct)", weekly, [m[1] for m in MATCHED]),
        (axes[1], "Daily CumMAE (direct)", daily, [m[2] for m in MATCHED]),
    ]:
        for i, (lab, key, color) in enumerate(models):
            vals = [
                _metric(src["models"][key], h, "cummae_rounded", cum=True) for h in h_keys
            ]
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
        ax.set_xticklabels(labels)
        ax.set_ylabel("CumMAE (rounded)")
        ax.set_title(grain)
        ax.legend(frameon=False, loc="best")

    fig.suptitle(
        "Like-for-like direct MH CumMAE: weekly vs daily (seed 42)",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "fig_weekly_daily_direct_cummae")


def fig_zero_rate(zr: dict) -> None:
    daily_zr = float(zr["daily"]["zero_rate"])
    weekly_zr = float(zr["weekly"]["zero_rate"])
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    bars = ax.bar(
        ["Daily", "Weekly"],
        [daily_zr, weekly_zr],
        color=["#4C78A8", "#F58518"],
        edgecolor="white",
        width=0.55,
    )
    for b, v in zip(bars, [daily_zr, weekly_zr]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.015,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Zero rate (pooled locked 800)")
    ax.set_title("Intermittency: daily vs weekly grain")
    save(fig, "fig_zero_rate_daily_vs_weekly")


def main() -> None:
    weekly = load(ROOT / "ab_runs/weekly/weekly_mh8_locked800_s42.json")
    daily_path = ROOT / "ab_runs/weekly/daily_direct_mh60_locked800_s42.json"
    if not daily_path.exists():
        raise SystemExit(f"Missing {daily_path}; run daily direct MH bake-off first.")
    daily = load(daily_path)
    zr = load(ROOT / "ab_runs/weekly/zero_rate_daily_vs_weekly_locked800.json")
    fig_zero_rate(zr)
    fig_iwmae_compare(weekly, daily)
    fig_cummae_compare(weekly, daily)


if __name__ == "__main__":
    main()
