#!/usr/bin/env python3
"""Code-faithful label overlays for the overall architecture figure.

Source art is the embedded slide PNG from ``deepsequence.pptx``
(``ppt/media/image3.png``). This script covers incorrect labels and redraws
text to match ``components_lightweight.py`` / feature configs:

  - Temporal → trend time index (not calendar DoW/month)
  - Fourier → fixed by default; learnable ω optional
  - Lag Expert → Regressor (lags + intermittent state)
  - Avoid blanket “monotone experts” (seasonal is not monotone)
  - Mixer query note: SKU + lag/intermittent context
  - Monthly lags footnote: frequency preset {1,2,12} (feature_config_monthly)

Writes: fig_m5_architecture.{png,pdf}, fig_architecture_ds.*, 
        deepsequence_shared_from_ppt_adjusted_v4.png
and patches the PNG back into deepsequence.pptx.
"""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
SRC = OUT / "fig_m5_architecture.png"
PPTX = OUT / "deepsequence.pptx"
MEDIA = "ppt/media/image3.png"

# Colors matching the slide
C_BLUE = (30, 100, 180)
C_GREEN = (46, 125, 50)
C_ORANGE = (230, 126, 34)
C_PINK = (194, 24, 91)
C_INK = (40, 48, 56)
C_MUTED = (70, 80, 90)
C_WHITE = (255, 255, 255)
C_CREAM = (252, 250, 246)  # mixer panel fill approx
C_NOTES_BG = (248, 250, 252)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf"
    )
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _cover(draw: ImageDraw.ImageDraw, box, fill=C_WHITE) -> None:
    draw.rectangle(box, fill=fill)


def _center_text(draw, box, text, font, fill, *, max_width: int | None = None):
    x0, y0, x1, y1 = box
    tw = (max_width if max_width is not None else (x1 - x0)) - 4
    # simple wrap
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= tw:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if not lines:
        return
    line_h = font.size + 2
    total_h = line_h * len(lines)
    cy = y0 + max(0, (y1 - y0 - total_h) // 2)
    for i, line in enumerate(lines):
        lw = draw.textlength(line, font=font)
        cx = x0 + (x1 - x0 - lw) / 2
        draw.text((cx, cy + i * line_h), line, font=font, fill=fill)


def annotate(im: Image.Image) -> Image.Image:
    im = im.copy().convert("RGB")
    d = ImageDraw.Draw(im)

    f_title = _font(15, bold=True)
    f_sub = _font(11)
    f_sub_sm = _font(10)
    f_exp = _font(14, bold=True)
    f_exp_sub = _font(11)
    f_rail = _font(11)
    f_note = _font(11)
    f_foot = _font(11)
    f_subhead = _font(14, bold=True)
    f_mixer = _font(11)

    # ---- Subtitle: drop blanket "Monotone Experts" ----
    # "DS + Softsign + Monotone Experts + Context-Aware Mixer"
    _cover(d, (420, 58, 1130, 86), C_WHITE)
    # redraw colored segments
    parts = [
        ("DS + Softsign", C_PINK),
        (" + ", C_INK),
        ("Experts", C_GREEN),
        (" + ", C_INK),
        ("Context-Aware Mixer", C_ORANGE),
    ]
    total_w = sum(d.textlength(t, font=f_subhead) for t, _ in parts)
    x = 768 - total_w / 2  # center under title (~image mid ~768)
    y = 62
    for t, c in parts:
        d.text((x, y), t, font=f_subhead, fill=c)
        x += d.textlength(t, font=f_subhead)

    # ---- INPUT boxes (measured top borders) ----
    # Temporal 208–392
    _cover(d, (250, 122, 385, 148), C_WHITE)  # title text after icon
    d.text((268, 124), "Trend time", font=f_title, fill=C_BLUE)
    _cover(d, (230, 148, 385, 182), C_WHITE)
    _center_text(d, (230, 148, 385, 182), "time index (not DoW/month)", f_sub_sm, C_MUTED)

    # Fourier 421–597
    _cover(d, (470, 122, 590, 148), C_WHITE)
    d.text((478, 124), "Fourier", font=f_title, fill=C_BLUE)
    _cover(d, (430, 148, 590, 182), C_WHITE)
    _center_text(d, (430, 148, 590, 182), "fixed Fourier (learnable ω opt.)", f_sub_sm, C_MUTED)

    # Lag Features 814–987 → lags + intermittent state
    _cover(d, (860, 122, 980, 148), C_WHITE)
    d.text((868, 124), "Lags / state", font=f_title, fill=C_BLUE)
    _cover(d, (822, 148, 980, 182), C_WHITE)
    _center_text(d, (822, 148, 980, 182), "lags + intermittent state (†)", f_sub_sm, C_MUTED)

    # ---- Left rail under EXPERTS: remove blanket monotone ----
    _cover(d, (28, 268, 198, 355), C_WHITE)
    d.text((32, 270), "Specialized", font=f_rail, fill=C_MUTED)
    d.text((32, 286), "expert networks", font=f_rail, fill=C_MUTED)
    d.text((32, 308), "(monotone: trend,", font=f_rail, fill=C_MUTED)
    d.text((32, 324), "holiday, regressor;", font=f_rail, fill=C_MUTED)
    d.text((32, 340), "seasonal not)", font=f_rail, fill=C_MUTED)

    # ---- Expert titles / subs (aligned under input columns) ----
    # Seasonal sub: learnable Fourier → fixed Fourier
    _cover(d, (470, 288, 625, 310), C_WHITE)
    _center_text(d, (470, 288, 625, 310), "fixed Fourier (ω opt.)", f_exp_sub, C_MUTED)

    # Lag Expert → Regressor (column ~900–1140; title text after bar icon ~960+)
    _cover(d, (955, 252, 1145, 286), C_WHITE)
    d.text((968, 260), "Regressor", font=f_exp, fill=C_GREEN)
    _cover(d, (920, 282, 1145, 314), C_WHITE)
    _center_text(d, (920, 282, 1145, 314), "lags + intermittent state", f_exp_sub, C_MUTED)

    # L1: Lag Attention → Regressor Attention
    _cover(d, (930, 415, 1145, 448), C_WHITE)
    _center_text(d, (930, 415, 1145, 448), "Regressor Attention", f_exp, C_GREEN)
    _cover(d, (930, 446, 1145, 480), C_WHITE)
    _center_text(d, (930, 446, 1145, 480), "(over lags / state)", f_sub_sm, C_MUTED)

    # ---- Mixer query note (cover old "SKU-conditioned query generation") ----
    _cover(d, (312, 555, 575, 605), C_CREAM)
    d.text((318, 562), "SKU + lag/intermittent", font=f_mixer, fill=C_ORANGE)
    d.text((318, 578), "context query", font=f_mixer, fill=C_ORANGE)

    # FiLM stays on expert softsign path in the art; clarifying note only (no layout move).

    # ---- Notes panel ----
    _cover(d, (1205, 805, 1495, 948), C_NOTES_BG)
    notes = [
        "† Daily: lags 1,2,7",
        "  Monthly: lags 1,2,12",
        "  (freq. preset / monthly YAML)",
        "  Quarterly: lags 1,4",
        "Softsign: bounded & stable.",
        "Monotone maps: trend /",
        "  holiday / regressor only",
        "  (seasonal not monotone).",
        "Query = SKU ⊕ lag/state;",
        "  FiLM on expert softsign path.",
    ]
    y = 810
    for line in notes:
        d.text((1212, y), line, font=f_note, fill=C_INK)
        y += 13

    # ---- Footer ----
    _cover(d, (55, 978, 1485, 1018), C_WHITE)
    foot = (
        "DeepSequence decomposes demand into occurrence (p) and magnitude (b), "
        "combines specialized experts (monotone trend/holiday/regressor; seasonal Fourier) "
        "with hierarchical attention mixing, and forecasts intermittent and non-intermittent SKUs."
    )
    _center_text(d, (55, 978, 1485, 1018), foot, f_foot, C_MUTED)

    return im


def export_all(im: Image.Image) -> None:
    png = OUT / "fig_m5_architecture.png"
    im.save(png, "PNG", optimize=True)
    im.convert("RGB").save(OUT / "fig_m5_architecture.pdf", "PDF", resolution=300.0)
    shutil.copyfile(png, OUT / "fig_architecture_ds.png")
    shutil.copyfile(OUT / "fig_m5_architecture.pdf", OUT / "fig_architecture_ds.pdf")
    shutil.copyfile(png, OUT / "deepsequence_shared_from_ppt_adjusted_v4.png")
    print(f"wrote {png.name} (+ pdf, aliases, adjusted_v4)")


def patch_pptx(im: Image.Image) -> None:
    if not PPTX.is_file():
        print(f"skip pptx patch: missing {PPTX.name}")
        return
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    tmp = PPTX.with_suffix(".pptx.tmp")
    with zipfile.ZipFile(PPTX, "r") as zin, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            payload = data if item.filename == MEDIA else zin.read(item.filename)
            zout.writestr(item, payload)
    tmp.replace(PPTX)
    print(f"patched {PPTX.name}:{MEDIA}")


def load_source() -> Image.Image:
    """Prefer measured pristine twin (coords keyed to this art), not shared_from_ppt."""
    pristine = OUT / "fig_m5_architecture_pristine.png"
    if pristine.is_file():
        print(f"base art: {pristine.name}")
        return Image.open(pristine).convert("RGB")
    # Fallbacks
    bak = OUT / "deepsequence_shared_from_ppt_adjusted_v4.png"
    if bak.is_file():
        # May already be annotated; prefer only if pristine missing
        print(f"base art fallback: {bak.name}")
        return Image.open(bak).convert("RGB")
    if PPTX.is_file():
        with zipfile.ZipFile(PPTX) as zf:
            raw = zf.read(MEDIA)
        print(f"base art: pptx {MEDIA}")
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(SRC).convert("RGB")


def main() -> None:
    base = load_source()
    out = annotate(base)
    export_all(out)
    patch_pptx(out)


if __name__ == "__main__":
    main()
