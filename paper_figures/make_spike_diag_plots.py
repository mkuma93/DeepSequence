#!/usr/bin/env python3
"""Spike-aware loss diagnostic: plot y, ŷ=p·b, p, b on lumpy daily SKUs.

Selects 5–10 locked SKUs with visible lumps (nonzero days + spike height in
the test window — not max-sparsity only). Trains DeepSequence with the
opt-in ``spike_aware`` loss (additive stack, country holidays) and dumps:

  paper_figures/fig_spike_diag_panel.{png,pdf,json}
  paper_figures/fig_spike_diag_<sku>.{png,pdf}   (optional per-SKU)

Example:
  TF_USE_LEGACY_KERAS=1 .venv-test/bin/python paper_figures/make_spike_diag_plots.py \\
    --epochs 30 --n_plot 8 --feature_config_daily feature_config_daily_country_holiday.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from eval_helpers import filter_aligned, resolve_sku_zero_rates, split_components
from feature_config_loader import load_feature_config
from holiday_calendar import (
    RETAIL_WINDOW_KEYS,
    binary_holiday_features,
    build_country_holiday_distances,
)
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss

INK = "#1f2933"
C_ACT = "#37474f"
C_YH = "#1e88e5"
C_P = "#00897b"
C_B = "#ef6c00"
C_HOL = "#c62828"
FONT = "DejaVu Sans"


def _save_fig(fig, stem: str):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {png.name} + {pdf.name}")


def _country_calendar_enabled(cfg) -> bool:
    meta = cfg.config.get("metadata", {}) or {}
    mode = str(meta.get("holiday_calendar", "static")).lower()
    return mode in ("country", "per_country", "country_aware")


def _attach_binary_holidays(hol_df: pd.DataFrame, cfg) -> pd.DataFrame:
    names = cfg.binary_holiday_names
    if not names:
        return hol_df
    out = hol_df.copy()
    meta = cfg.config.get("metadata", {}) or {}
    window_days = int(meta.get("binary_holiday_window_days", 0))
    window_keys = meta.get("binary_holiday_window_keys")
    if window_keys is None and window_days > 0:
        window_keys = list(RETAIL_WINDOW_KEYS)
    keys = []
    want_any = False
    for bname in names:
        if bname in ("is_any_holiday", "is_AnyHoliday"):
            want_any = True
            continue
        if bname.startswith("is_"):
            keys.append(bname[len("is_") :])
    built = binary_holiday_features(
        out,
        holiday_keys=keys
        or [n.replace("days_from_", "", 1) for n in cfg.holiday_names],
        window_days=window_days,
        window_keys=window_keys,
        include_any=want_any,
    )
    for col in built.columns:
        out[col] = built[col].to_numpy()
    return out


def _rebuild_holidays_for_split(df: pd.DataFrame, cfg) -> pd.DataFrame:
    keys = [n.replace("days_from_", "", 1) for n in cfg.holiday_names]
    meta = cfg.config.get("metadata", {}) or {}
    country_col = meta.get("holiday_country_column")
    distance_scope = str(
        meta.get("holiday_distance_scope", meta.get("distance_scope", "year"))
    )
    hol = build_country_holiday_distances(
        df,
        holiday_keys=keys or None,
        sku_col="id_var",
        date_col="ds",
        country_col=country_col if country_col in df.columns else None,
        default_country=str(meta.get("holiday_country_default", "US")),
        distance_scope=distance_scope,
    )
    return _attach_binary_holidays(hol, cfg)


def _mark_holidays(ax, dates, mark_dates):
    if not mark_dates:
        return
    dset = set(mark_dates)
    for d in pd.to_datetime(dates):
        if str(d)[:10] in dset:
            ax.axvline(d, color=C_HOL, alpha=0.22, lw=1.0, zorder=0)


def select_lumpy_skus(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    locked: list[str],
    n_plot: int = 8,
    min_test_nonzero: int = 3,
) -> list[str]:
    """Prefer locked SKUs with visible lumps in the test window.

    Score = test nonzero count × (max_test / mean_nonzero_test), among SKUs
    that are not near-empty on test. Falls back to higher train nonzero rate.
    """
    locked_set = set(locked)
    rows = []
    for sku, g in test_df.groupby(test_df["id_var"].astype(str), sort=False):
        if sku not in locked_set:
            continue
        y = g["Quantity"].to_numpy(np.float64)
        nz = y > 0
        n_nz = int(nz.sum())
        if n_nz < min_test_nonzero:
            continue
        mean_nz = float(y[nz].mean()) if n_nz else 0.0
        max_y = float(y.max())
        lump = max_y / max(mean_nz, 1e-6)
        rows.append(
            {
                "sku": sku,
                "n_nz": n_nz,
                "nz_rate": float(nz.mean()),
                "max": max_y,
                "lump": lump,
                "score": n_nz * lump,
            }
        )
    if not rows:
        # Fallback: highest train nonzero rates on locked list
        for sku, g in train_df.groupby(train_df["id_var"].astype(str), sort=False):
            if sku not in locked_set:
                continue
            y = g["Quantity"].to_numpy(np.float64)
            rows.append(
                {
                    "sku": sku,
                    "n_nz": int((y > 0).sum()),
                    "nz_rate": float((y > 0).mean()),
                    "max": float(y.max()),
                    "lump": 1.0,
                    "score": float((y > 0).mean()),
                }
            )
    rows.sort(key=lambda r: (-r["score"], -r["nz_rate"], r["sku"]))
    chosen = [r["sku"] for r in rows[:n_plot]]
    print("Lumpy plot SKUs:")
    for r in rows[:n_plot]:
        print(
            f"  {r['sku']}: n_nz_test={r['n_nz']} nz_rate={r['nz_rate']:.3f} "
            f"max={r['max']:.1f} lump={r['lump']:.2f} score={r['score']:.1f}"
        )
    return chosen


def _train_ds(
    cfg,
    tr,
    va,
    y_train,
    y_val,
    sku_train,
    sku_val,
    zero_rate,
    n_skus,
    epochs,
    batch,
    *,
    loss_recipe="spike_aware",
    alpha_bce=1.0,
    w_gated=0.0,
    w_mag=1.0,
    zero_mag_weight=0.05,
    positive_bce_boost=2.0,
    focal_gamma=0.0,
    patience=8,
):
    _, sku_rates = resolve_sku_zero_rates(y_train, sku_train, n_skus=n_skus)
    base = build_hierarchical_model_lightweight(
        n_temporal_features=len(cfg.trend_indices),
        n_fourier_features=len(cfg.seasonal_indices),
        n_holiday_features=len(cfg.holiday_indices),
        n_lag_features=len(cfg.regressor_indices),
        n_skus=n_skus,
        hidden_dim=48,
        sku_embedding_dim=4,
        dropout_rate=0.23,
        n_changepoints=15,
        component_combine="additive",
    )
    _ = base(
        [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
        training=False,
    )
    pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))
    model = AdaptiveWeightedModel(
        base_model=base,
        bce_loss_fn=WeightedBCELoss(weight_nonzero=pos_weight, weight_zero=1.0),
        mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        zero_rate=zero_rate,
        avg_nonzero_demand=float(y_train[y_train > 0].mean()) if (y_train > 0).any() else 1.0,
        pos_weight=pos_weight,
        sku_zero_rates=sku_rates,
        loss_recipe=loss_recipe,
        alpha_bce=alpha_bce,
        w_gated=w_gated,
        w_mag=w_mag,
        zero_mag_weight=zero_mag_weight,
        positive_bce_boost=positive_bce_boost,
        focal_gamma=focal_gamma,
        use_fixed_weights=True,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
    ytr = {"final_forecast": y_train.reshape(-1, 1), "base_forecast": y_train.reshape(-1, 1)}
    yva = {"final_forecast": y_val.reshape(-1, 1), "base_forecast": y_val.reshape(-1, 1)}
    model.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=epochs,
        batch_size=batch,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
        ],
        verbose=2,
    )
    return model


def _plot_sku_triptych(sku, d, holiday_marks=True, stem=None):
    dates = pd.to_datetime(d["dates"])
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.2), sharex=True)
    # Panel 1: y and ŷ
    ax = axes[0]
    if holiday_marks:
        _mark_holidays(ax, dates, d.get("holiday_dates") or [])
    ax.plot(dates, d["y"], color=C_ACT, lw=1.4, label="Actual y", drawstyle="steps-mid")
    ax.plot(dates, d["yhat"], color=C_YH, lw=1.5, label=r"$\hat{y}=p\cdot b$", alpha=0.95)
    ax.set_ylabel("Demand", fontsize=9, color=INK, family=FONT)
    ax.set_title(sku, fontsize=10, color=INK, family=FONT, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    # Panel 2: p
    ax = axes[1]
    if holiday_marks:
        _mark_holidays(ax, dates, d.get("holiday_dates") or [])
    ax.plot(dates, d["p"], color=C_P, lw=1.4, label="Occurrence p")
    ax.axhline(0.5, color="#90a4ae", ls=":", lw=0.8)
    ax.set_ylabel("p", fontsize=9, color=INK, family=FONT)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Panel 3: b
    ax = axes[2]
    if holiday_marks:
        _mark_holidays(ax, dates, d.get("holiday_dates") or [])
    ax.plot(dates, d["b"], color=C_B, lw=1.4, label="Magnitude b")
    ax.plot(
        dates,
        d["y"],
        color=C_ACT,
        lw=1.0,
        alpha=0.45,
        label="Actual y",
        drawstyle="steps-mid",
    )
    ax.set_ylabel("b / y", fontsize=9, color=INK, family=FONT)
    ax.set_xlabel("Date", fontsize=9, color=INK, family=FONT)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    for a in axes:
        a.tick_params(labelsize=8)
    fig.suptitle(
        "Spike-aware loss — occurrence / magnitude diagnostics",
        fontsize=12,
        fontweight="bold",
        color=INK,
        family=FONT,
        y=1.01,
    )
    fig.tight_layout()
    if stem:
        _save_fig(fig, stem)
    return fig


def _plot_panel_grid(series: dict, holiday_marks=True, stem="fig_spike_diag_panel"):
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(12.0, 2.4 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (sku, d) in zip(axes, series.items()):
        dates = pd.to_datetime(d["dates"])
        if holiday_marks:
            _mark_holidays(ax, dates, d.get("holiday_dates") or [])
        ax.plot(dates, d["y"], color=C_ACT, lw=1.3, label="Actual y", drawstyle="steps-mid")
        ax.plot(dates, d["yhat"], color=C_YH, lw=1.4, label=r"$\hat{y}$", alpha=0.95)
        ax2 = ax.twinx()
        ax2.plot(dates, d["p"], color=C_P, lw=1.1, ls="--", label="p", alpha=0.85)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("p", fontsize=8, color=C_P, family=FONT)
        ax2.tick_params(labelsize=7, colors=C_P)
        ax.set_ylabel("Demand", fontsize=8, color=INK, family=FONT)
        short = sku if len(sku) < 42 else sku[:39] + "…"
        ax.set_title(
            f"{short}  ·  p̄_spike={d.get('p_mean_spike', float('nan')):.2f} "
            f"p̄_quiet={d.get('p_mean_quiet', float('nan')):.2f}",
            fontsize=9,
            color=INK,
            family=FONT,
            loc="left",
        )
        ax.spines["top"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
        if ax is axes[0]:
            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=7, frameon=False)
    axes[-1].set_xlabel("Date", fontsize=9, color=INK, family=FONT)
    fig.suptitle(
        "Spike-aware loss diagnostics (test window) — y, ŷ, p",
        fontsize=13,
        fontweight="bold",
        color=INK,
        family=FONT,
        y=1.005,
    )
    fig.tight_layout()
    _save_fig(fig, stem)


def parse_args():
    p = argparse.ArgumentParser(description="Spike-aware loss p/b/ŷ diagnostics")
    p.add_argument(
        "--data_dir",
        default=os.environ.get(
            "DEEPSEQUENCE_DATA_DIR",
            "/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
        ),
    )
    p.add_argument(
        "--sku_list_daily",
        default=str(ROOT / "ab_runs/recompare/sku_list_daily_data42.json"),
    )
    p.add_argument(
        "--feature_config_daily",
        default=str(ROOT / "feature_config_daily_country_holiday.yaml"),
    )
    p.add_argument("--max_skus", type=int, default=40)
    p.add_argument("--n_plot", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--alpha_bce", type=float, default=1.0)
    p.add_argument("--w_mag", type=float, default=1.0)
    p.add_argument("--w_gated", type=float, default=0.0)
    p.add_argument("--zero_mag_weight", type=float, default=0.05)
    p.add_argument("--positive_bce_boost", type=float, default=2.0)
    p.add_argument("--focal_gamma", type=float, default=0.0)
    p.add_argument("--fig_prefix", default="fig_spike_diag")
    p.add_argument(
        "--per_sku",
        type=int,
        default=1,
        help="Also write per-SKU triptych figs (1/0).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    locked = json.loads(Path(args.sku_list_daily).read_text())
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    h_tr = pd.read_csv(data_dir / "holiday_features_train.csv")
    h_va = pd.read_csv(data_dir / "holiday_features_val.csv")
    h_te = pd.read_csv(data_dir / "holiday_features_test.csv")

    plot_skus = select_lumpy_skus(train_df, test_df, locked, n_plot=args.n_plot)
    pool = list(dict.fromkeys(plot_skus + [s for s in locked if s not in plot_skus]))
    chosen_list = pool[: args.max_skus]
    chosen = set(chosen_list)
    print(f"Train SKUs={len(chosen_list)} plot={plot_skus}")

    train_df, h_tr = filter_aligned(train_df, h_tr, chosen)
    val_df, h_va = filter_aligned(val_df, h_va, chosen)
    test_df, h_te = filter_aligned(test_df, h_te, chosen)

    cats = pd.Categorical(train_df["id_var"].astype(str))
    sku_map = {str(k): i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return df["id_var"].astype(str).map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val = enc(train_df), enc(val_df)
    zero_rate = float((y_train == 0).mean())

    cfg = load_feature_config(args.feature_config_daily)
    use_country = _country_calendar_enabled(cfg)
    print(
        f"feature_config={args.feature_config_daily} n_feat={cfg.total_features} "
        f"holiday_calendar={cfg.config.get('metadata', {}).get('holiday_calendar', 'static')}"
    )
    if use_country:
        print("rebuilding holiday distances from per-country calendars")
        h_tr = _rebuild_holidays_for_split(train_df, cfg)
        h_va = _rebuild_holidays_for_split(val_df, cfg)
        h_te = _rebuild_holidays_for_split(test_df, cfg)
    else:
        h_tr = _attach_binary_holidays(h_tr, cfg)
        h_va = _attach_binary_holidays(h_va, cfg)
        h_te = _attach_binary_holidays(h_te, cfg)

    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, _ = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    t_idx = cfg.trend_indices[0]
    tmin = float(X_train[:, t_idx].min())
    tmax = float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    X_train_n, X_val_n, X_test_n = X_train.copy(), X_val.copy(), X_test.copy()
    for X in (X_train_n, X_val_n, X_test_n):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span
    tr = split_components(X_train_n, cfg)
    va = split_components(X_val_n, cfg)
    te = split_components(X_test_n, cfg)

    print("=== train DeepSequence (spike_aware) ===")
    knobs = {
        "loss_recipe": "spike_aware",
        "alpha_bce": args.alpha_bce,
        "w_gated": args.w_gated,
        "w_mag": args.w_mag,
        "zero_mag_weight": args.zero_mag_weight,
        "positive_bce_boost": args.positive_bce_boost,
        "focal_gamma": args.focal_gamma,
        "component_combine": "additive",
    }
    print(f"knobs: {knobs}")
    model = _train_ds(
        cfg,
        tr,
        va,
        y_train,
        y_val,
        sku_train,
        sku_val,
        zero_rate,
        n_skus,
        args.epochs,
        args.batch_size,
        loss_recipe="spike_aware",
        alpha_bce=args.alpha_bce,
        w_gated=args.w_gated,
        w_mag=args.w_mag,
        zero_mag_weight=args.zero_mag_weight,
        positive_bce_boost=args.positive_bce_boost,
        focal_gamma=args.focal_gamma,
        patience=args.patience,
    )

    sku_test = enc(test_df)
    pred = model.predict([*te, sku_test], batch_size=4096, verbose=0)
    yhat = np.asarray(pred["final_forecast"]).reshape(-1)
    p = np.asarray(pred["non_zero_probability"]).reshape(-1)
    b = np.asarray(pred["base_forecast"]).reshape(-1)

    hol_block_names = cfg.holiday_block_names
    n_dist = len(cfg.holiday_names)
    series = {}
    dump = {
        "protocol": "spike_aware_diag",
        "seed": args.seed,
        "epochs": args.epochs,
        "feature_config": str(args.feature_config_daily),
        "knobs": knobs,
        "positive_bce_weight_resolved": float(model.pos_weight),
        "plot_skus": plot_skus,
        "skus": {},
        "summary": {},
    }
    p_spike_all, p_quiet_all = [], []
    for sku in plot_skus:
        m = test_df["id_var"].astype(str).to_numpy() == sku
        if not m.any():
            continue
        dates = [str(x)[:10] for x in pd.to_datetime(test_df.loc[m, "ds"])]
        y = test_df.loc[m, "Quantity"].to_numpy(np.float64)
        hol_m = h_te.loc[m, hol_block_names].to_numpy(np.float32) if hol_block_names else None
        if hol_m is not None and hol_m.shape[1] > n_dist:
            hol_dates = [
                d for d, row in zip(dates, hol_m) if float(row[n_dist:].max()) > 0.5
            ]
        elif hol_m is not None:
            hol_dates = [
                d for d, row in zip(dates, hol_m) if float(np.abs(row).min()) < 0.5
            ]
        else:
            hol_dates = []
        yhat_s = yhat[m].astype(np.float64)
        p_s = p[m].astype(np.float64)
        b_s = b[m].astype(np.float64)
        spike = y > 0
        p_mean_spike = float(p_s[spike].mean()) if spike.any() else float("nan")
        p_mean_quiet = float(p_s[~spike].mean()) if (~spike).any() else float("nan")
        if spike.any():
            p_spike_all.append(p_mean_spike)
        if (~spike).any():
            p_quiet_all.append(p_mean_quiet)
        d = {
            "dates": dates,
            "y": y.tolist(),
            "yhat": yhat_s.tolist(),
            "p": p_s.tolist(),
            "b": b_s.tolist(),
            "holiday_dates": hol_dates,
            "p_mean_spike": p_mean_spike,
            "p_mean_quiet": p_mean_quiet,
            "delta_p_spike_minus_quiet": (
                p_mean_spike - p_mean_quiet
                if np.isfinite(p_mean_spike) and np.isfinite(p_mean_quiet)
                else float("nan")
            ),
            "corr_p_vs_occurrence": (
                float(np.corrcoef(p_s, spike.astype(np.float64))[0, 1])
                if spike.any() and spike.sum() < len(spike) and p_s.std() > 1e-12
                else float("nan")
            ),
        }
        series[sku] = d
        dump["skus"][sku] = d
        print(
            f"  {sku}: p̄_spike={p_mean_spike:.3f} p̄_quiet={p_mean_quiet:.3f} "
            f"Δp={d['delta_p_spike_minus_quiet']:.3f} "
            f"corr(p,z)={d['corr_p_vs_occurrence']:.3f}"
        )
        if args.per_sku:
            safe = sku.replace("/", "_").replace(" ", "_")
            _plot_sku_triptych(
                sku, d, holiday_marks=True, stem=f"{args.fig_prefix}_{safe}"
            )

    _plot_panel_grid(series, holiday_marks=True, stem=f"{args.fig_prefix}_panel")
    dump["summary"] = {
        "mean_p_spike": float(np.mean(p_spike_all)) if p_spike_all else float("nan"),
        "mean_p_quiet": float(np.mean(p_quiet_all)) if p_quiet_all else float("nan"),
        "mean_delta_p": (
            float(np.mean(p_spike_all) - np.mean(p_quiet_all))
            if p_spike_all and p_quiet_all
            else float("nan")
        ),
        "p_moves_on_spike_days": (
            bool(np.mean(p_spike_all) > np.mean(p_quiet_all) + 0.02)
            if p_spike_all and p_quiet_all
            else False
        ),
    }
    print(f"summary: {dump['summary']}")
    (OUT / f"{args.fig_prefix}_panel.json").write_text(json.dumps(dump, indent=2))
    print(f"wrote {args.fig_prefix}_panel.json")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
