"""Render a dedicated changepoint + monotone mechanism diagram for DeepSequence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2933"
C_CP = "#e3f2fd"
C_CP_E = "#1e88e5"
C_MONO = "#e8f5e9"
C_MONO_E = "#2e7d32"
C_ATTN = "#fff3e0"
C_ATTN_E = "#ef6c00"
C_OUT = "#eceff1"
C_OUT_E = "#37474f"
FONT = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(14.5, 8.8))
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
            fontsize=fs - 2.6,
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
    "DeepSequence — Changepoint + Monotone Mechanism",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=INK,
    family=FONT,
)
ax.text(
    50,
    92.7,
    "How monotone experts are built from changepoint basis features",
    ha="center",
    va="center",
    fontsize=11.5,
    color="#52606d",
    family=FONT,
)

# Top lane: single-feature (trend-like)
inp = box(
    6.0,
    66.0,
    16.0,
    16.0,
    "Input x",
    C_CP,
    C_CP_E,
    bold=True,
    sub="time (trend)\nor |distance|",
)
cp = box(
    27.0,
    66.0,
    22.0,
    16.0,
    "Changepoint Basis",
    C_CP,
    C_CP_E,
    bold=True,
    sub="ϕk(x)=ReLU(x-τk)\nfor k=1..K",
)
slope = box(
    54.0,
    66.0,
    20.0,
    16.0,
    "Monotone Slopes",
    C_MONO,
    C_MONO_E,
    bold=True,
    sub="wk = softplus(rk)\n· tanh(sk)",
)
mono = box(
    79.0,
    66.0,
    16.0,
    16.0,
    "Mono Scalar",
    C_MONO,
    C_MONO_E,
    bold=True,
    sub="m(x)=Σ wkϕk(x)+b",
)

arrow((22.0, 74.0), (27.0, 74.0), color=C_CP_E)
arrow((49.0, 74.0), (54.0, 74.0), color=C_MONO_E)
arrow((74.0, 74.0), (79.0, 74.0), color=C_MONO_E)

ax.text(
    50,
    59.8,
    "Single feature path (Trend): monotonicity comes from constrained slope magnitudes/sign.",
    ha="center",
    va="center",
    fontsize=9.0,
    color="#52606d",
    family=FONT,
    style="italic",
)

# Bottom lane: multi-channel + attention
multi = box(
    6.0,
    28.0,
    18.0,
    18.0,
    "Multi-channel Inputs",
    C_CP,
    C_CP_E,
    bold=True,
    sub="holiday distances\nor lag features",
)
per_ch = box(
    29.0,
    28.0,
    23.0,
    18.0,
    "Per-channel Mono",
    C_MONO,
    C_MONO_E,
    bold=True,
    sub="apply top path\nindependently per channel",
)
attn = box(
    57.0,
    28.0,
    20.0,
    18.0,
    "Channel Attention",
    C_ATTN,
    C_ATTN_E,
    bold=True,
    sub="α = softmax(z/T)\n(+ entropy reg.)",
)
agg = box(
    82.0,
    28.0,
    13.0,
    18.0,
    "Expert Output",
    C_OUT,
    C_OUT_E,
    bold=True,
    sub="Σ αi · mi",
)

arrow((24.0, 37.0), (29.0, 37.0), color=C_CP_E)
arrow((52.0, 37.0), (57.0, 37.0), color=C_ATTN_E)
arrow((77.0, 37.0), (82.0, 37.0), color=C_OUT_E)

ax.text(
    50,
    21.0,
    "Multi-feature path (Holiday/Regressor): monotone channels first, then attention over channels.",
    ha="center",
    va="center",
    fontsize=9.0,
    color="#52606d",
    family=FONT,
    style="italic",
)

# bridge
arrow((87.0, 66.0), (87.0, 46.0), color="#7b8794", lw=1.4, dashed=True)
ax.text(
    88.5,
    55.8,
    "same monotone\nconstruction",
    ha="left",
    va="center",
    fontsize=7.8,
    color="#7b8794",
    family=FONT,
)

ax.text(
    50,
    6.0,
    "This expert output is then consumed by the Level-2 hierarchical component mixer.",
    ha="center",
    va="center",
    fontsize=8.8,
    color="#9aa5b1",
    family=FONT,
)

out = Path(__file__).resolve().parent / "fig_changepoint_monotone.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")

