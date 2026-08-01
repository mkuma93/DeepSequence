#!/usr/bin/env python3
"""Publication-grade Results figures for PAPER.md (locked multi-seed + ablations).

Reads ab_runs/reclaim JSON summaries; writes PNG (+ PDF) under paper_figures/.
No prediction dumps are required — forecast SKU traces are intentionally omitted
when yhat/y_true series are not present in the repo.
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

# Colorblind-safe (Okabe–Ito-inspired)
C_DS = "#0072B2"
C_TST = "#E69F00"
C_LGBM = "#009E73"
C_TSB = "#CC79A7"
C_PROPHET = "#D55E00"
C_FULL = "#0072B2"
C_ABL = "#56B4E9"

FONT = "DejaVu Sans"
plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
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


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def mean_std_series(block: dict, model: str, horizons: list[int]):
    means, stds = [], []
    for h in horizons:
        entry = block[str(h)][model]
        means.append(float(entry["mean"]))
        stds.append(float(entry["std"]))
    return np.asarray(means), np.asarray(stds)


def fig1_daily_iwmae() -> None:
    summary = load(ROOT / "ab_runs/reclaim/multiseed/daily_multiseed_long_loyalty_summary.json")
    block = summary["iwmae_mean_std"]
    horizons = [1, 7, 14, 28, 60]
    series = [
        ("DeepSequence", "deepsequence", C_DS, "o"),
        ("TST", "temporal_transformer", C_TST, "s"),
        ("LightGBM", "lightgbm", C_LGBM, "^"),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for label, key, color, marker in series:
        mu, sd = mean_std_series(block, key, horizons)
        ax.errorbar(
            horizons,
            mu,
            yerr=sd,
            label=label,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            capsize=3,
            elinewidth=1.1,
        )
    ax.set_xlabel(r"Horizon $h$ (days)")
    ax.set_ylabel("IWMAE (mean ± std)")
    ax.set_xticks(horizons)
    ax.set_title("Daily multi-seed IWMAE vs horizon (seeds 42–46)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig1_daily_iwmae_horizon")


def fig2_carparts_iwmae() -> None:
    summary = load(ROOT / "ab_runs/reclaim/multiseed/carparts_multiseed_long_loyalty_summary.json")
    block = summary["iwmae_mean_std"]
    prophet = load(ROOT / "ab_runs/reclaim/prophet_carparts/carparts_mh_1_2_6.json")
    horizons = [1, 2, 6]
    prophet_iwmae = [
        float(prophet["models"]["prophet"]["by_horizon"][str(h)]["iwmae"]) for h in horizons
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    ax = axes[0]
    for label, key, color, marker in [
        ("DeepSequence", "deepsequence", C_DS, "o"),
        ("TSB", "tsb", C_TSB, "D"),
        ("LightGBM", "lightgbm", C_LGBM, "^"),
    ]:
        mu, sd = mean_std_series(block, key, horizons)
        ax.errorbar(
            horizons,
            mu,
            yerr=sd,
            label=label,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            capsize=3,
            elinewidth=1.1,
        )
    ax.set_xlabel(r"Horizon $h$ (months)")
    ax.set_ylabel("IWMAE (mean ± std)")
    ax.set_xticks(horizons)
    ax.set_title("Car Parts multi-seed IWMAE")
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    # Seed-42 bake-off panel (Table 5) — includes Prophet; do not mix with
    # multi-seed rounded IWMAE on the left (see PAPER.md protocol note).
    ax2 = axes[1]
    bake = {
        "TSB": [0.850, 0.767, 0.834],
        "DeepSequence": [0.882, 0.778, 0.834],
        "Prophet": [round(v, 3) for v in prophet_iwmae],
        "LightGBM": [0.889, 0.832, 0.890],
    }
    x = np.arange(len(horizons))
    width = 0.18
    colors = {"TSB": C_TSB, "DeepSequence": C_DS, "Prophet": C_PROPHET, "LightGBM": C_LGBM}
    for i, (name, vals) in enumerate(bake.items()):
        ax2.bar(
            x + (i - 1.5) * width,
            vals,
            width=width,
            label=name,
            color=colors[name],
            edgecolor="white",
            linewidth=0.4,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"h={h}" for h in horizons])
    ax2.set_ylabel("IWMAE")
    ax2.set_title("Seed-42 bake-off (+ Prophet)")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    ax2.set_ylim(0.7, 1.0)

    fig.tight_layout()
    save(fig, "fig2_carparts_iwmae_horizon")


def fig3_daily_pi() -> None:
    summary = load(ROOT / "ab_runs/reclaim/multiseed/daily_multiseed_long_loyalty_summary.json")
    block = summary["pi_mid_margin_loyalty_0p25_mean_std"]
    horizons = [7, 14, 28, 60]
    series = [
        ("DeepSequence", "deepsequence", C_DS, "o"),
        ("TST", "temporal_transformer", C_TST, "s"),
        ("LightGBM", "lightgbm", C_LGBM, "^"),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for label, key, color, marker in series:
        mu, sd = mean_std_series(block, key, horizons)
        ax.errorbar(
            horizons,
            mu,
            yerr=sd,
            label=label,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            capsize=3,
            elinewidth=1.1,
        )
    ax.set_xlabel(r"Horizon / lead time $h$ (days)")
    ax.set_ylabel(r"Mid-margin $\pi$ (mean ± std)")
    ax.set_xticks(horizons)
    ax.set_title(r"Daily multi-seed decision $\pi$ ($m{=}0.25$, $C_{\mathrm{loyalty}}{=}0.25$)")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig3_daily_decision_pi_horizon")


def fig4_novelty_ablation() -> None:
    h1 = load(ROOT / "ab_runs/reclaim/ablate_novelty/daily_h1_summary.json")["arms"]
    mh = load(ROOT / "ab_runs/reclaim/ablate_novelty/daily_mh60_summary.json")["arms"]

    # Panel A: gate ablation at H=1 (gate not in MH summary)
    arms_h1 = [
        ("Full", "full"),
        ("−mixer", "minus_mixer"),
        ("−L1 attn", "minus_level1_attn"),
        ("−mono", "minus_mono"),
        ("−gate", "minus_gate"),
        ("+cross", "plus_cross"),
    ]
    # Panel B: long-horizon Full vs −mixer/−L1/−mono/+cross at h=28 and h=60
    arms_mh = [
        ("Full", "full"),
        ("−mixer", "minus_mixer"),
        ("−L1 attn", "minus_level1_attn"),
        ("−mono", "minus_mono"),
        ("+cross", "plus_cross"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))

    ax = axes[0]
    labels = [a[0] for a in arms_h1]
    vals = [float(h1[k]["iwmae"]) for _, k in arms_h1]
    colors = [C_FULL if lab == "Full" else ("#D55E00" if lab == "−gate" else C_ABL) for lab in labels]
    bars = ax.bar(np.arange(len(labels)), vals, color=colors, edgecolor="white", width=0.72)
    ax.axhline(vals[0], color=C_FULL, linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("IWMAE")
    ax.set_title("One-step novelty ablation (seed 42)")
    ax.set_ylim(min(vals) - 0.15, max(vals) + 0.15)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax = axes[1]
    x = np.arange(len(arms_mh))
    w = 0.36
    v28 = [float(mh[k]["28"]) for _, k in arms_mh]
    v60 = [float(mh[k]["60"]) for _, k in arms_mh]
    ax.bar(x - w / 2, v28, width=w, label="h=28", color="#0072B2", edgecolor="white")
    ax.bar(x + w / 2, v60, width=w, label="h=60", color="#56B4E9", edgecolor="white")
    ax.axhline(v28[0], color="#0072B2", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(v60[0], color="#56B4E9", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([a[0] for a in arms_mh], rotation=30, ha="right")
    ax.set_ylabel("IWMAE")
    ax.set_title("Long-horizon novelty ablation (seed 42)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(3.5, 7.0)

    fig.tight_layout()
    save(fig, "fig4_novelty_ablation")


def main() -> None:
    fig1_daily_iwmae()
    fig2_carparts_iwmae()
    fig3_daily_pi()
    fig4_novelty_ablation()
    print(
        "Note: SKU forecast line plots skipped — no yhat/y_true dumps in repo "
        "(metrics-only JSON under ab_runs/ and eval_results_*)."
    )


if __name__ == "__main__":
    main()
