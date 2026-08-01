"""Render a dedicated hierarchical-attention internals diagram for DeepSequence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2933"
C_L1 = "#e8f5e9"
C_L1_E = "#2e7d32"
C_L2 = "#fff3e0"
C_L2_E = "#ef6c00"
C_OUT = "#eceff1"
C_OUT_E = "#37474f"
FONT = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(13.5, 8.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, *, fs=11, bold=False, sub=None):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=1.8",
        linewidth=1.7,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(
            cx,
            cy + h * 0.17,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=INK,
            family=FONT,
            fontweight="bold" if bold else "normal",
            zorder=3,
        )
        ax.text(
            cx,
            cy - h * 0.24,
            sub,
            ha="center",
            va="center",
            fontsize=fs - 2.5,
            color="#52606d",
            family=FONT,
            zorder=3,
        )
    else:
        ax.text(
            cx,
            cy,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=INK,
            family=FONT,
            fontweight="bold" if bold else "normal",
            zorder=3,
        )
    return {"x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy}


def arrow(p0, p1, *, color=INK, lw=1.6, dashed=False, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            zorder=1,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
        )
    )


ax.text(
    50,
    96.5,
    "DeepSequence — Hierarchical Attention Internals",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=INK,
    family=FONT,
)
ax.text(
    50,
    92.8,
    "Dedicated view: intra-expert attention (Level 1) and inter-expert mixer (Level 2)",
    ha="center",
    va="center",
    fontsize=11.5,
    color="#52606d",
    family=FONT,
)

# Level 1 inputs
l1_in = box(
    7.0,
    67.0,
    25.0,
    17.0,
    "Level 1 Inputs",
    C_L1,
    C_L1_E,
    bold=True,
    sub="Seasonal Fourier features\nHoliday distance channels\nLag channels (1,2,7 / freq-aware)",
)
l1_attn = box(
    38.0,
    67.0,
    27.0,
    17.0,
    "Level 1 Attention",
    C_L1,
    C_L1_E,
    bold=True,
    sub="α = softmax(z / T)\nmasked-entropy regularization",
)
l1_out = box(
    71.0,
    67.0,
    22.0,
    17.0,
    "Expert Scalars",
    C_L1,
    C_L1_E,
    bold=True,
    sub="trend, seasonal,\nholiday, regressor",
)

arrow((32.0, 75.5), (38.0, 75.5), color=C_L1_E)
arrow((65.0, 75.5), (71.0, 75.5), color=C_L1_E)

ax.text(
    50,
    61.8,
    "Level 1: attention is applied within each expert block where multiple candidate features exist.",
    ha="center",
    va="center",
    fontsize=9.2,
    color="#52606d",
    family=FONT,
    style="italic",
)

# Level 2
l2_ctx = box(
    7.0,
    34.0,
    25.0,
    18.0,
    "Mixer Context",
    C_L2,
    C_L2_E,
    bold=True,
    sub="SKU embedding ⊕ lag context",
)
l2_mix = box(
    38.0,
    34.0,
    27.0,
    18.0,
    "Level 2 Mixer",
    C_L2,
    C_L2_E,
    bold=True,
    sub="wi = softmax(score(ei, c) / T)\nbase = Σ wi · ei",
)
l2_out = box(
    71.0,
    34.0,
    22.0,
    18.0,
    "Mixture Output",
    C_L2,
    C_L2_E,
    bold=True,
    sub="context-adaptive\nexpert weighting",
)

arrow((32.0, 43.0), (38.0, 43.0), color=C_L2_E)
arrow((65.0, 43.0), (71.0, 43.0), color=C_L2_E)

# bridge
arrow((82.0, 67.0), (82.0, 52.0), color="#7b8794", lw=1.5, dashed=True)
ax.text(
    83.8,
    59.8,
    "expert vectors ei",
    ha="left",
    va="center",
    fontsize=8.2,
    color="#7b8794",
    family=FONT,
)

# final
final = box(
    29.0,
    9.0,
    42.0,
    15.0,
    "How this fits the full model",
    C_OUT,
    C_OUT_E,
    bold=True,
    sub="Level 2 output feeds base forecast head;\nintermittent gate p then produces final ŷ = p × b",
)
arrow((82.0, 34.0), (50.0, 24.0), color=C_OUT_E, rad=-0.1)

out = Path(__file__).resolve().parent / "fig_hierarchical_attention_internals.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")

