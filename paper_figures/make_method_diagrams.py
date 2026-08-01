#!/usr/bin/env python3
"""Publication-grade Method diagrams for DeepSequence (code-faithful).

Matches ``components_lightweight.py``:
  - ChangepointReLU: softplus(deltas) → cumsum → scale to [tmin,tmax] → ReLU(t−cp)
  - Monotone slopes: softplus(raw) * tanh(raw_sign)
  - Trend (locked): monotone PWL only — no Level-1 attention
  - Holiday/Regressor (locked): per-channel softplus-PWL → MaskedEntropyAttention
  - Context mixer: q = [sku_emb ; Dense(lag/intermittent context)] → temp-softmax over experts
  - Gate: ŷ = p · b

Primary end-to-end architecture (Figure 5) is the user PowerPoint slide
(``deepsequence.pptx`` slide 3) with code-faithful label overlays from
``annotate_architecture_labels.py`` → ``fig_m5_architecture.png``.
The matplotlib schematic is optional only.

Writes PNG+PDF under paper_figures/ for figs M1–M4 (+ optional schematic).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

OUT = Path(__file__).resolve().parent
INK = "#1f2933"
MUTED = "#52606d"
LIGHT = "#9aa5b1"
C_BLUE = ("#e3f2fd", "#1e88e5")
C_GREEN = ("#e8f5e9", "#2e7d32")
C_ORANGE = ("#fff3e0", "#ef6c00")
C_PINK = ("#fce4ec", "#c2185b")
C_GRAY = ("#eceff1", "#37474f")
C_PURPLE = ("#ede7f6", "#5e35b1")
FONT = "DejaVu Sans"


def _fig(w=12.5, h=7.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc, ec, *, fs=11, bold=False, sub=None, dashed=False):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=1.5",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(
            cx,
            cy + h * 0.18,
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
            cy - h * 0.26,
            sub,
            ha="center",
            va="center",
            fontsize=max(7.0, fs - 2.4),
            color=MUTED,
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


def arrow(ax, p0, p1, *, color=INK, lw=1.5, dashed=False, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            zorder=1,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def title(ax, main, sub=None, y=95.5):
    ax.text(
        50,
        y,
        main,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=INK,
        family=FONT,
    )
    if sub:
        ax.text(
            50,
            y - 4.0,
            sub,
            ha="center",
            va="center",
            fontsize=10.5,
            color=MUTED,
            family=FONT,
        )


def save(fig, stem: str):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {png.name} + {pdf.name}")


# ---------------------------------------------------------------------------
# Fig M1 — Changepoint selection / parameterization
# ---------------------------------------------------------------------------
def fig_changepoint():
    fig, ax = _fig(13.2, 7.6)
    title(
        ax,
        "Changepoint selection (Trend expert)",
        "Ordered locations via softplus+cumsum — ChangepointReLU in components_lightweight.py",
    )

    box(
        ax,
        4,
        68,
        18,
        14,
        "Time feature t",
        *C_BLUE,
        bold=True,
        sub="date_numeric\n[batch, 1]",
    )
    box(
        ax,
        28,
        68,
        22,
        14,
        "Learnable deltas δ",
        *C_BLUE,
        bold=True,
        sub="δ ∈ ℝᴷ  (K=10 trend)\ninit: equal spacing",
    )
    box(
        ax,
        56,
        68,
        20,
        14,
        "Ordered CPs",
        *C_GREEN,
        bold=True,
        sub="δ⁺ = softplus(δ)\ncp = cumsum(δ⁺)",
    )
    box(
        ax,
        82,
        68,
        14,
        14,
        "Scale",
        *C_GREEN,
        bold=True,
        sub="→ [tₘᵢₙ, tₘₐₓ]",
    )
    arrow(ax, (22, 75), (28, 75), color=C_BLUE[1])
    arrow(ax, (50, 75), (56, 75), color=C_GREEN[1])
    arrow(ax, (76, 75), (82, 75), color=C_GREEN[1])

    # hinge illustration
    box(
        ax,
        10,
        28,
        36,
        28,
        "ReLU hinge basis",
        *C_BLUE,
        bold=True,
        sub="ϕₖ(t) = ReLU(t − cpₖ)\noutput shape [batch, K]",
        fs=12,
    )
    # mini schematic of hinges
    xs = np.linspace(14, 42, 80)
    cps = [18, 26, 34]
    for i, cp in enumerate(cps):
        yy = 32 + 2.2 * i
        ax.plot([14, cp], [yy, yy], color=LIGHT, lw=1.2, zorder=4)
        ax.plot([cp, 42], [yy, yy + (42 - cp) * 0.35], color=C_BLUE[1], lw=1.6, zorder=4)
        ax.plot(cp, yy, "o", color=C_BLUE[1], ms=4, zorder=5)
        ax.text(cp, yy + 2.0, f"cp{i+1}", ha="center", fontsize=7, color=MUTED, family=FONT)

    box(
        ax,
        54,
        28,
        40,
        28,
        "Why ordered CPs?",
        *C_GRAY,
        bold=True,
        sub=(
            "softplus(δ) > 0 ⇒ strictly increasing\n"
            "cumsum keeps cp₀ < cp₁ < … < cpₖ₋₁\n"
            "rescale preserves coverage of time range\n"
            "no discrete selection — locations are continuous params"
        ),
        fs=11,
    )
    arrow(ax, (89, 68), (74, 56), color=C_GREEN[1], rad=0.15)
    arrow(ax, (28, 68), (28, 56), color=C_BLUE[1])

    ax.text(
        50,
        8,
        "Locked stack: Trend uses these hinges with softplus-monotone slopes (next figure) — no Level-1 attention over CPs.",
        ha="center",
        fontsize=9,
        color=LIGHT,
        family=FONT,
    )
    save(fig, "fig_m1_changepoint_selection")


# ---------------------------------------------------------------------------
# Fig M2 — Monotone softplus maps
# ---------------------------------------------------------------------------
def fig_monotone():
    fig, ax = _fig(13.2, 8.0)
    title(
        ax,
        "Monotone softplus–PWL maps",
        "slope = softplus(raw) × tanh(raw_sign)  — positivity of magnitude, learned direction",
    )

    # shared formula
    box(
        ax,
        18,
        72,
        64,
        12,
        "Shared hinge slope constraint",
        *C_GREEN,
        bold=True,
        sub="m = softplus(s) · tanh(σ)   ⇒   |m| = softplus(s) ≥ 0,   sign(m) = sign(tanh(σ))",
        fs=12,
    )

    # three experts
    experts = [
        (
            4,
            "Trend",
            "g(t) = b + Σₖ mₖ · ReLU(t−cpₖ)\nshared sign σ across hinges\nno Level-1 attention",
            C_BLUE,
        ),
        (
            36,
            "Holiday",
            "per holiday h: |dₕ| → PWL mono\nmₕ = softplus(sₕ)·tanh(σₕ)\nthen Level-1 over holidays",
            C_ORANGE,
        ),
        (
            68,
            "Regressor",
            "per lag/state channel j:\nxⱼ → PWL mono in xⱼ\nmⱼ = softplus(sⱼ)·tanh(σⱼ)\nthen Level-1 over channels",
            C_PINK,
        ),
    ]
    for x, name, sub, col in experts:
        box(ax, x, 40, 28, 22, name, *col, bold=True, sub=sub, fs=12)

    # monotonicity sketch
    ax.text(50, 32, "Effect on response shape (fixed SKU)", ha="center", fontsize=10, color=MUTED, family=FONT)
    xx = np.linspace(12, 88, 200)
    # monotone increasing sketch
    yy = 12 + 12 * (1 / (1 + np.exp(-(xx - 50) / 8)))
    ax.plot(xx, yy, color=C_GREEN[1], lw=2.2, zorder=4)
    ax.text(90, yy[-1], "mono ↑ or ↓\n(shared sign)", fontsize=8, color=C_GREEN[1], family=FONT, va="center")
    # unconstrained dashed wiggle
    yy2 = 18 + 4 * np.sin((xx - 12) / 6) + 0.08 * (xx - 12)
    ax.plot(xx, yy2, color=LIGHT, lw=1.4, ls="--", zorder=3)
    ax.text(90, yy2[-1], "−mono ablation\n(unconstrained)", fontsize=7.5, color=LIGHT, family=FONT, va="center")

    ax.text(
        50,
        3.5,
        "SKU FiLM uses softplus scale so per-SKU affine personalization preserves monotonicity in the structured input.",
        ha="center",
        fontsize=8.5,
        color=LIGHT,
        family=FONT,
    )
    save(fig, "fig_m2_monotone_softplus")


# ---------------------------------------------------------------------------
# Fig M3 — Level-1 selection attention
# ---------------------------------------------------------------------------
def fig_level1_attention():
    fig, ax = _fig(13.0, 7.8)
    title(
        ax,
        "Level-1 selection attention (intra-expert)",
        "Holiday & regressor: MaskedEntropyAttention over monotone channel scalars; seasonal: freq attention",
    )

    # holiday path
    box(ax, 4, 62, 20, 16, "Holiday |dₕ|", *C_ORANGE, bold=True, sub="H distance channels")
    box(ax, 28, 62, 20, 16, "Per-channel mono", *C_GREEN, bold=True, sub="softplus-PWL → mₕ")
    box(
        ax,
        52,
        62,
        24,
        16,
        "Selection attention",
        *C_ORANGE,
        bold=True,
        sub="α = softmax(z / T)\nentropy regularized",
    )
    box(ax, 80, 62, 16, 16, "Holiday scalar", *C_GRAY, bold=True, sub="→ Dense / softsign")
    arrow(ax, (24, 70), (28, 70), color=C_ORANGE[1])
    arrow(ax, (48, 70), (52, 70), color=C_GREEN[1])
    arrow(ax, (76, 70), (80, 70), color=C_ORANGE[1])

    # regressor path
    box(ax, 4, 34, 20, 16, "Lag / state xⱼ", *C_PINK, bold=True, sub="e.g. lags 1,2,7\ndays since sale")
    box(ax, 28, 34, 20, 16, "Per-channel mono", *C_GREEN, bold=True, sub="softplus-PWL → mⱼ")
    box(
        ax,
        52,
        34,
        24,
        16,
        "Lag selection attn",
        *C_PINK,
        bold=True,
        sub="same MaskedEntropy\nablation: uniform 1/n",
    )
    box(ax, 80, 34, 16, 16, "Regressor scalar", *C_GRAY, bold=True, sub="→ Dense / softsign")
    arrow(ax, (24, 42), (28, 42), color=C_PINK[1])
    arrow(ax, (48, 42), (52, 42), color=C_GREEN[1])
    arrow(ax, (76, 42), (80, 42), color=C_PINK[1])

    # notes
    box(
        ax,
        10,
        8,
        36,
        16,
        "Seasonal (also Level-1)",
        *C_BLUE,
        bold=True,
        sub="masked-entropy attention over\nFourier frequency channels",
    )
    box(
        ax,
        54,
        8,
        36,
        16,
        "Trend (deliberately no Level-1)",
        *C_GRAY,
        bold=True,
        sub="single monotone temporal basis\navoids competing CP heads",
    )
    save(fig, "fig_m3_level1_attention")


# ---------------------------------------------------------------------------
# Fig M4 — Context-aware component mixer
# ---------------------------------------------------------------------------
def fig_context_mixer():
    fig, ax = _fig(13.0, 7.8)
    title(
        ax,
        "Context-aware component mixer (Level-2)",
        "Inter-expert reweighting conditioned on lag / intermittent regime — not calendar features",
    )

    box(ax, 4, 70, 18, 14, "Expert scalars", *C_GREEN, bold=True, sub="e_trend … e_reg\n[batch, 4]")
    box(
        ax,
        4,
        46,
        18,
        14,
        "Shared SKU emb. eᵢ",
        *C_PURPLE,
        bold=True,
        sub="one table (optional)\nsame eᵢ as FiLM / gate",
    )
    box(
        ax,
        4,
        22,
        18,
        14,
        "Regime context c",
        *C_BLUE,
        bold=True,
        sub="lags + days/months\nsince last sale",
    )

    box(ax, 32, 34, 20, 16, "Dense(c)", *C_ORANGE, bold=True, sub="mish, no bias\nproj_dim ≈ H/4")
    box(ax, 56, 46, 20, 16, "Query q", *C_ORANGE, bold=True, sub="q = [eᵢ ; Dense(c)]\nconcat")
    box(
        ax,
        80,
        46,
        16,
        16,
        "Temp-softmax",
        *C_ORANGE,
        bold=True,
        sub="w = softmax(z/T)\nentropy + ortho",
    )

    arrow(ax, (22, 29), (32, 40), color=C_BLUE[1])
    arrow(ax, (22, 53), (56, 56), color=C_PURPLE[1], dashed=True)
    arrow(ax, (52, 42), (56, 50), color=C_ORANGE[1])
    arrow(ax, (76, 54), (80, 54), color=C_ORANGE[1])
    arrow(ax, (22, 77), (88, 62), color=C_GREEN[1], dashed=True, rad=-0.15)

    box(
        ax,
        32,
        8,
        48,
        16,
        "Mixed base  →  softplus magnitude b",
        *C_PINK,
        bold=True,
        sub="base = Σₖ wₖ · eₖ     then     b = softplus(·)     (gate p applied in architecture figure)",
        fs=11,
    )
    arrow(ax, (88, 46), (56, 24), color=C_ORANGE[1], rad=0.2)

    ax.text(
        50,
        2.5,
        "eᵢ is the shared SKU embedding (one table), not a mixer-only lookup; calendar stays inside experts.",
        ha="center",
        fontsize=8.5,
        color=LIGHT,
        family=FONT,
    )
    save(fig, "fig_m4_context_mixer")


# ---------------------------------------------------------------------------
# Optional schematic (NOT primary Figure 5 — primary is deepsequence.pptx slide 3)
# ---------------------------------------------------------------------------
def fig_architecture_schematic():
    """Secondary matplotlib schematic; does not overwrite fig_m5_architecture.*."""
    fig, ax = _fig(14.5, 9.4)
    title(
        ax,
        "DeepSequence architecture (schematic)",
        "Secondary sketch only — paper Figure 5 uses deepsequence.pptx slide 3",
        y=96.5,
    )

    # structural inputs (SKU is NOT a fifth parallel feature stream)
    iw, gap, x0 = 15.5, 2.2, 4
    labels = [
        ("Time", "trend index"),
        ("Fourier", "seasonality"),
        ("Holiday |d|", "distances"),
        ("Lags / state", "regime"),
    ]
    inb = []
    for i, (t, s) in enumerate(labels):
        b = box(ax, x0 + i * (iw + gap), 84, iw, 8, t, *C_BLUE, sub=s, fs=9.5, bold=True)
        inb.append(b)

    # single shared SKU embedding path (right rail)
    sku_id = box(ax, 78, 84, 18, 8, "sku_id", *C_PURPLE, bold=True, sub="integer index", fs=10)
    sku_emb = box(
        ax,
        78,
        70,
        18,
        9,
        "Embedding → eᵢ",
        *C_PURPLE,
        bold=True,
        sub="shared SKU embedding\n(one table)",
        fs=10,
    )
    arrow(ax, (sku_id["cx"], sku_id["y"]), (sku_emb["cx"], sku_emb["y"] + sku_emb["h"]), color=C_PURPLE[1])

    # experts (+ FiLM note)
    experts = ["Trend", "Seasonal", "Holiday", "Regressor"]
    exb = []
    for i, name in enumerate(experts):
        b = box(
            ax,
            x0 + i * (iw + gap),
            66,
            iw,
            9,
            name,
            *C_GREEN,
            bold=True,
            sub="softsign + FiLM(eᵢ)",
            fs=10.5,
        )
        exb.append(b)
        arrow(ax, (inb[i]["cx"], inb[i]["y"]), (b["cx"], b["y"] + b["h"]), color=C_BLUE[1])
        # dashed shared eᵢ → each expert FiLM
        arrow(
            ax,
            (sku_emb["x"], sku_emb["cy"]),
            (b["x"] + b["w"], b["cy"] + 1.5),
            color=C_PURPLE[1],
            dashed=True,
            lw=1.1,
            rad=-0.12 + 0.04 * i,
        )

    # L1
    l1 = [
        ("softplus-PWL", "no attn"),
        ("freq attn", "masked-ent."),
        ("sel. attn", "over holidays"),
        ("sel. attn", "over lags"),
    ]
    l1b = []
    for i, (t, s) in enumerate(l1):
        b = box(ax, x0 + i * (iw + gap), 52, iw, 8, t, *C_GREEN, sub=s, fs=9)
        l1b.append(b)
        arrow(ax, (exb[i]["cx"], exb[i]["y"]), (b["cx"], b["y"] + b["h"]), color=C_GREEN[1])

    # mixer
    mixer = box(
        ax,
        10,
        34,
        52,
        10,
        "Level-2 context-aware mixer",
        *C_ORANGE,
        bold=True,
        sub="q = [eᵢ ; Dense(lag context)]  →  w = softmax(z/T)  →  base = Σ w·e",
        fs=11,
    )
    for b in l1b:
        arrow(ax, (b["cx"], b["y"]), (b["cx"], mixer["y"] + mixer["h"]), color=C_ORANGE[1])
    # lag context dashed into mixer
    arrow(
        ax,
        (inb[3]["cx"], inb[3]["y"]),
        (mixer["x"] + mixer["w"] - 4, mixer["y"] + mixer["h"]),
        color=C_BLUE[1],
        dashed=True,
        rad=0.28,
    )
    # shared eᵢ → mixer
    arrow(
        ax,
        (sku_emb["cx"], sku_emb["y"]),
        (mixer["x"] + mixer["w"], mixer["cy"] + 2),
        color=C_PURPLE[1],
        dashed=True,
        rad=-0.18,
    )

    # heads
    mag = box(ax, 10, 16, 22, 10, "Magnitude b", *C_PINK, bold=True, sub="softplus(base)")
    gate = box(ax, 40, 16, 24, 10, "Occurrence p", *C_PINK, bold=True, sub="σ(g(·, eᵢ))")
    out = box(ax, 72, 16, 18, 10, "ŷ = p · b", *C_GRAY, bold=True, sub="final forecast")
    arrow(ax, (mixer["cx"] - 10, mixer["y"]), (mag["cx"], mag["y"] + mag["h"]), color=C_PINK[1])
    arrow(ax, (mixer["cx"] + 6, mixer["y"]), (gate["cx"], gate["y"] + gate["h"]), color=C_PINK[1])
    # shared eᵢ → intermittent gate
    arrow(
        ax,
        (sku_emb["cx"], sku_emb["y"]),
        (gate["x"] + gate["w"], gate["cy"] + 3),
        color=C_PURPLE[1],
        dashed=True,
        rad=-0.35,
    )
    arrow(ax, (mag["x"] + mag["w"], mag["cy"]), (out["x"], out["cy"]), color=C_GRAY[1])
    arrow(ax, (gate["x"] + gate["w"], gate["cy"]), (out["x"], out["cy"]), color=C_GRAY[1])

    ax.text(
        50,
        5.5,
        "Dashed purple: shared eᵢ → expert FiLM, Level-2 mixer, intermittent gate  ·  "
        "not per-expert Embedding tables  ·  calendar FiLM / cross-net default off",
        ha="center",
        fontsize=8.2,
        color=LIGHT,
        family=FONT,
    )
    save(fig, "fig_m5_architecture_schematic")


def export_architecture_from_pptx(pptx_name: str = "deepsequence.pptx", slide_media: str = "ppt/media/image3.png"):
    """Deprecated raw export — use annotate_architecture_labels (code-faithful).

    Kept for debugging only; ``main()`` runs the annotate pipeline so regen
    does not overwrite Figure 5 with uncorrected PPT labels.
    """
    import io
    import shutil
    import zipfile

    from PIL import Image

    pptx = OUT / pptx_name
    if not pptx.is_file():
        raise FileNotFoundError(pptx)
    with zipfile.ZipFile(pptx) as zf:
        data = zf.read(slide_media)
    png = OUT / "fig_m5_architecture_raw_from_pptx.png"
    png.write_bytes(data)
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.save(OUT / "fig_m5_architecture_raw_from_pptx.pdf", "PDF", resolution=300.0)
    print(f"raw export {pptx.name}:{slide_media} → {png.name} (not primary fig_m5)")


def export_architecture_code_faithful():
    """Apply code-faithful label overlays and write primary fig_m5_* (+ patch PPT)."""
    import annotate_architecture_labels as aal

    aal.main()


def main():
    fig_changepoint()
    fig_monotone()
    fig_level1_attention()
    fig_context_mixer()
    # Primary overall architecture: pristine slide art + code-faithful labels.
    export_architecture_code_faithful()


if __name__ == "__main__":
    main()
