"""Render the DeepSequence architecture diagram (base DS, no hybrid branch).

Emphasizes the two-level attention hierarchy the package is named for:
  Level 1 (intra-expert): feature selection / weighting inside each expert
    - trend:     softplus monotone (single changepoint basis — no attention needed)
    - seasonal:  masked-entropy freq attention over Fourier frequencies
    - holiday:   monotone attention over multiple holiday-distance features
    - regressor: lag attention over multiple lag features (1, 2, 7)
  Level 2 (hierarchical / inter-expert): context-aware component mixer that
    weights the four experts by SKU + lag regime.

Then gated heads: magnitude b (softplus) and occurrence p (intermittent gate),
final y = p * b.  Output: fig_architecture_ds.png (300 dpi).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2933"
C_INPUT = "#e3f2fd"
C_INPUT_E = "#1e88e5"
C_SKU = "#ede7f6"
C_SKU_E = "#5e35b1"
C_EXPERT = "#e8f5e9"
C_EXPERT_E = "#2e7d32"
C_LOCAL = "#f1f8e9"
C_LOCAL_E = "#7cb342"
C_MIX = "#fff3e0"
C_MIX_E = "#ef6c00"
C_HEAD = "#fce4ec"
C_HEAD_E = "#c2185b"
C_OUT = "#eceff1"
C_OUT_E = "#37474f"

FONT = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(15.5, 11.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, *, fs=10.5, bold=False, sub=None,
        dashed=False, rounding=1.6):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.6, edgecolor=ec, facecolor=fc,
        linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + h * 0.18, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, family=FONT,
                zorder=3)
        ax.text(cx, cy - h * 0.27, sub, ha="center", va="center",
                fontsize=fs - 2.7, color="#52606d", family=FONT, zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, family=FONT,
                zorder=3)
    return {"cx": cx, "cy": cy, "x": x, "y": y, "w": w, "h": h,
            "top": (cx, y + h), "bot": (cx, y), "l": (x, cy), "r": (x + w, cy)}


def arrow(p0, p1, *, color=INK, lw=1.6, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color,
        zorder=1, connectionstyle=f"arc3,rad={rad}",
        linestyle="--" if dashed else "-"))


# ---- title ---------------------------------------------------------------
ax.text(50, 98, "DeepSequence — Hierarchical Attention Forecaster",
        ha="center", va="center", fontsize=18, fontweight="bold", color=INK,
        family=FONT)
ax.text(50, 94.7,
        "Default:  DS + softsign + experts + context-aware mixer",
        ha="center", va="center", fontsize=11.5, color="#ef6c00", family=FONT,
        fontweight="bold")

# ---- right-rail band labels ---------------------------------------------
for yy, lab in [(88.5, "INPUTS"), (72.0, "EXPERTS"),
                (60.0, "LEVEL 1\nINTRA-EXPERT"), (44.0, "LEVEL 2\nHIER. ATTN"),
                (29.5, "GATED HEADS"), (12.0, "OUTPUTS")]:
    ax.text(98.6, yy, lab, ha="center", va="center", fontsize=8.2,
            color="#9aa5b1", family=FONT, fontweight="bold", rotation=90,
            linespacing=1.0)

# ---- INPUTS --------------------------------------------------------------
iy, ih, iw, gap, x0 = 87.0, 6.4, 16.5, 2.0, 6.5
inputs = [("Trend time", "time index"), ("Fourier", "fixed (\u03c9 opt.)"),
          ("Holiday", "days-from-event"), ("Lags / state", "lags + intermitt.\u2020")]
in_boxes = [box(x0 + i * (iw + gap), iy, iw, ih, t, C_INPUT, C_INPUT_E, sub=s)
            for i, (t, s) in enumerate(inputs)]
sku_in = box(x0 + 4 * (iw + gap), iy, 12.5, ih, "SKU id", C_INPUT, C_INPUT_E,
             sub="integer")
sku_emb = box(sku_in["x"], 76.5, 12.5, 5.4, "SKU Embedding", C_SKU, C_SKU_E,
              sub="dim 8", fs=9.4)
arrow(sku_in["bot"], sku_emb["top"], color=C_SKU_E)

# ---- EXPERTS (row) -------------------------------------------------------
ey, eh, ew = 68.5, 8.2, 16.5
experts = [("Trend", "monotone trend"),
           ("Seasonal", "fixed Fourier"),
           ("Holiday", "monotone |distance|"),
           ("Regressor", "lags + intermittent")]
ex_boxes = []
for i, (t, s) in enumerate(experts):
    x = x0 + i * (iw + gap)
    b = box(x, ey, ew, eh, t, C_EXPERT, C_EXPERT_E, fs=11, bold=True)
    ax.text(b["cx"], ey + eh * 0.30, s, ha="center", va="center", fontsize=7.9,
            color="#52606d", family=FONT, zorder=3)
    ax.text(b["cx"], ey + 1.2, "softsign + SKU-FiLM", ha="center", va="center",
            fontsize=7.3, color=C_EXPERT_E, family=FONT, fontstyle="italic",
            zorder=3)
    ex_boxes.append(b)
    arrow(in_boxes[i]["bot"], b["top"], color=C_INPUT_E)
    arrow((sku_emb["l"][0], sku_emb["cy"]), (b["x"] + b["w"], ey + 1.4),
          color=C_SKU_E, lw=0.9, dashed=True, rad=-0.2)

# ---- LEVEL 1: LOCAL (intra-expert) MECHANISM -----------------------------
# Trend:     single changepoint basis → softplus monotone, no attention needed.
# Seasonal:  multiple Fourier freqs   → masked-entropy freq attention.
# Holiday:   multiple distance feats  → monotone attention over distances.
# Regressor: multiple lag features    → lag attention over lag features.
ly, lh = 56.5, 6.6
local_labels = [("softplus monotone", "single basis"),
                ("freq attention", "masked-entropy"),
                ("monotone attention", "over distances"),
                ("regressor attention", "over lags / state")]
l_boxes = []
for i, (t, s) in enumerate(local_labels):
    x = x0 + i * (iw + gap)
    is_attn = (t == "freq attention")
    b = box(x, ly, ew, lh, t, C_LOCAL, C_LOCAL_E, fs=9.0, sub=s)
    l_boxes.append(b)
    arrow(ex_boxes[i]["bot"], b["top"], color=C_EXPERT_E)
ax.text(x0 - 0.3, ly + lh / 2,
        "intra-expert\nmechanism", ha="right", va="center", fontsize=7.6,
        color=C_LOCAL_E, family=FONT, fontstyle="italic", linespacing=1.1)

# ---- LEVEL 2: HIERARCHICAL (inter-expert) ATTENTION = MIXER ---------------
my, mh, mw = 40.0, 9.0, 4 * ew + 3 * gap
mx = x0
mixer = box(mx, my, mw, mh, "Context-Aware Component Mixer",
            C_MIX, C_MIX_E, fs=13, bold=True,
            sub="temperature-softmax attention over experts  \u00b7  "
                "entropy + orthogonality reg.")
for b in l_boxes:
    arrow(b["bot"], (b["cx"], my + mh), color=C_LOCAL_E)
# mixer query = SKU + lag context
ax.text(mx + mw + 0.5, my + mh * 0.5, "query =\nSKU embedding \u2295 lag context",
        ha="left", va="center", fontsize=8.0, color=C_MIX_E, family=FONT,
        fontstyle="italic", linespacing=1.15)
arrow((sku_emb["cx"], sku_emb["y"]), (mx + mw, my + mh * 0.7), color=C_SKU_E,
      lw=0.9, dashed=True, rad=-0.28)
arrow((in_boxes[3]["cx"], ey + eh), (mx + mw, my + mh * 0.3), color=C_INPUT_E,
      lw=0.9, dashed=True, rad=0.32)

# ---- attention internals (explicit hierarchical view) ----------------------
att = box(81.0, 47.0, 16.2, 23.5, "Attention Internals", C_OUT, C_OUT_E,
          fs=9.0, bold=True, sub="local + hierarchical")
ax.text(89.1, 64.3, "Level-1 (intra-expert)", ha="center", va="center",
        fontsize=7.1, color=INK, family=FONT, fontweight="bold")
ax.text(89.1, 61.8,
        "Seasonal:  \u03b1freq = softmax(z/T)\n"
        "Holiday:   \u03b1hol  = softmax(z/T)\n"
        "Regressor: \u03b1lag  = softmax(z/T)",
        ha="center", va="top", fontsize=6.5, color="#52606d", family=FONT,
        linespacing=1.2)
ax.plot([82.0, 96.2], [56.1, 56.1], color="#cbd2d9", linewidth=0.9, zorder=3)
ax.text(89.1, 54.7, "Level-2 (hierarchical mixer)", ha="center", va="center",
        fontsize=7.1, color=INK, family=FONT, fontweight="bold")
ax.text(89.1, 52.8,
        "c = [sku_emb \u2295 lag_ctx]\n"
        "wi = softmax(score(ei, c)/T)\n"
        "base = \u03a3 wi \u00b7 ei",
        ha="center", va="top", fontsize=6.5, color="#52606d", family=FONT,
        linespacing=1.2)
arrow((78.6, 59.8), (81.0, 59.8), color=C_LOCAL_E, lw=1.0, dashed=True)
arrow((78.6, 45.0), (81.0, 51.0), color=C_MIX_E, lw=1.0, dashed=True)

# base forecast + optional DCN cross
base = box(mx + mw / 2 + 3, 31.5, 20, 5.2, "Base Forecast", C_MIX, C_MIX_E,
           fs=10.5, bold=True, sub="\u03a3 w\u1d62 \u00b7 expert\u1d62")
cross = box(mx + mw / 2 - 23, 31.5, 20, 5.2, "DCN Cross", C_MIX, C_MIX_E,
            fs=9.4, sub="optional \u00b7 ablation (default OFF)", dashed=True)
arrow(mixer["bot"], base["top"], color=C_MIX_E, rad=0.0)
arrow((mixer["cx"] - 6, my), cross["top"], color=C_MIX_E, lw=1.0, dashed=True)
arrow(cross["r"], base["l"], color=C_MIX_E, lw=1.0, dashed=True)

# ---- GATED HEADS ---------------------------------------------------------
hy, hh, hw = 20.0, 8.4, 24
mag = box(mx + mw / 2 - 27, hy, hw, hh, "Magnitude  b", C_HEAD, C_HEAD_E,
          fs=11, bold=True, sub="softplus  (\u2265 0)")
gate = box(mx + mw / 2 + 3, hy, hw, hh, "Occurrence gate  p", C_HEAD, C_HEAD_E,
           fs=11, bold=True, sub="intermittent handler \u00b7 zero-rate prior")
arrow(base["bot"], (mag["cx"], mag["top"][1]), color=C_HEAD_E, rad=0.14)
arrow(base["bot"], (gate["cx"], gate["top"][1]), color=C_HEAD_E, rad=-0.10)
arrow((in_boxes[3]["cx"], ey + eh / 2), (gate["r"][0], gate["cy"]),
      color=C_INPUT_E, lw=0.9, dashed=True, rad=-0.5)

# ---- COMBINE + OUTPUTS ---------------------------------------------------
comb = box(mx + mw / 2 - 9, 9.5, 18, 5.6, "\u0177 = p \u00d7 b", C_OUT, C_OUT_E,
           fs=13, bold=True)
arrow(mag["bot"], (comb["cx"] - 3, comb["top"][1]), color=C_HEAD_E, rad=-0.12)
arrow(gate["bot"], (comb["cx"] + 3, comb["top"][1]), color=C_HEAD_E, rad=0.12)

outs = [("final_forecast", "\u0177"), ("non_zero_probability", "p"),
        ("base_forecast", "b")]
oy, oh, ow, ogap = 1.4, 5.4, 21, 2.5
ox0 = mx + (mw - (3 * ow + 2 * ogap)) / 2
for i, (t, s) in enumerate(outs):
    b = box(ox0 + i * (ow + ogap), oy, ow, oh, t, C_OUT, C_OUT_E, fs=9.6, sub=s)
    arrow((comb["cx"], comb["y"]), (b["cx"], b["top"][1]), color=C_OUT_E,
          lw=1.3, rad=(-0.14 if i == 1 else (0.14 if i == 2 else 0.0)))

# ---- legend --------------------------------------------------------------
ax.text(50, -1.6,
        "solid = default path   \u00b7   dashed = optional / conditioning / "
        "ablation   \u00b7   \u2020 lags are frequency-aware "
        "(daily 1,2,7 \u00b7 monthly 1,2,12 preset \u00b7 quarterly 1,4)   \u00b7   "
        "monotone maps: trend/holiday/regressor only",
        ha="center", va="center", fontsize=8.6, color="#9aa5b1", family=FONT)

out = Path(__file__).resolve().parent / "fig_architecture_ds.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
