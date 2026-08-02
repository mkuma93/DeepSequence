#!/usr/bin/env python3
"""Weekly forecast vs actual line plots (locked Direct-MH protocol).

Trains a small locked-SKU subset on the ISO-Monday weekly panel and dumps
per-SKU panels (not averages):

  fig_forecast_weekly_onestep.{png,pdf,json}
  fig_forecast_weekly_direct.{png,pdf,json}

Protocol mirrors ``eval.weekly_mh``: DeepSequence / LightGBM direct MH,
TSB classical recursive. One-step uses H=1 heads over the full test window;
direct uses H=8 from the first eligible test origin (panels h=1..4 and h=1..8).

Example::

  TF_USE_LEGACY_KERAS=1 .venv-test/bin/python \\
    paper_figures/make_forecast_weekly_plots.py --epochs 15 --max_skus 800
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
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.data.feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.eval.classical import croston_variants
from deepsequence_hierarchical_attention.eval.helpers import (
    class_balance_pos_weight,
    filter_aligned,
    split_components,
)
from deepsequence_hierarchical_attention.eval.weekly_mh import (
    build_mh_xy,
    classical_recursive,
    train_gated,
)

INK = "#1f2933"
C_ACT = "#37474f"
C_DS = "#1e88e5"
C_TSB = "#ef6c00"
C_LGBM = "#43a047"
FONT = "DejaVu Sans"

# Locked weekly exemplars chosen for mid intermittency + visible DS planning-rate
# variation under Direct-MH (not max sparsity).
WEEKLY_PLOT_SKUS = [
    "United Kingdom_22047",  # zero_rate≈0.25; DS tracks mid-volume levels
    "United Kingdom_79000",  # zero_rate≈0.42; DS mean near demand scale
    "United Kingdom_22710",  # zero_rate≈0.17; lower sparsity, responsive DS
    "United Kingdom_22594",  # zero_rate≈0.17; DS shows week-to-week swing
]


def _save_fig(fig, stem: str) -> None:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {png.name} + {pdf.name}")


def _load_holiday(data_dir: Path, split: str, n_rows: int) -> pd.DataFrame:
    path = data_dir / f"holiday_features_{split}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(index=range(n_rows))


def _plot_onestep(series_dict: dict, title: str, stem: str) -> None:
    n = len(series_dict)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 2.55 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (sku, d) in zip(axes, series_dict.items()):
        dates = pd.to_datetime(d["dates"])
        ax.plot(dates, d["y"], color=C_ACT, lw=1.4, label="Actual", drawstyle="steps-mid")
        ax.plot(dates, d["ds"], color=C_DS, lw=1.5, label="DeepSequence", alpha=0.95)
        ax.plot(dates, d["tsb"], color=C_TSB, lw=1.3, ls="--", label="TSB", alpha=0.95)
        if "lgbm" in d:
            ax.plot(
                dates,
                d["lgbm"],
                color=C_LGBM,
                lw=1.2,
                ls=":",
                label="LightGBM",
                alpha=0.95,
            )
        zr = d.get("zero_rate_test")
        subtitle = sku if zr is None else f"{sku}  ·  test zero-rate={zr:.2f}"
        ax.set_ylabel("Week demand", fontsize=9, color=INK, family=FONT)
        ax.set_title(subtitle, fontsize=10, color=INK, family=FONT, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Week (Monday)", fontsize=9, color=INK, family=FONT)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, family=FONT, y=1.01)
    fig.tight_layout()
    _save_fig(fig, stem)


def _plot_direct(series_dict: dict, title: str, stem: str, h_short: int, h_long: int) -> None:
    n = len(series_dict)
    fig, axes = plt.subplots(n, 2, figsize=(12.5, 2.5 * n), sharey=False)
    if n == 1:
        axes = np.array([axes])
    for i, (sku, d) in enumerate(series_dict.items()):
        for j, h in enumerate((h_short, h_long)):
            ax = axes[i, j]
            steps = np.arange(1, h + 1)
            ax.plot(steps, d["y"][:h], color=C_ACT, lw=1.5, marker="o", ms=3.5, label="Actual")
            ax.plot(steps, d["ds"][:h], color=C_DS, lw=1.5, marker="s", ms=3.2, label="DeepSequence")
            ax.plot(
                steps,
                d["tsb"][:h],
                color=C_TSB,
                lw=1.3,
                ls="--",
                marker="^",
                ms=3.2,
                label="TSB",
            )
            if "lgbm" in d:
                ax.plot(
                    steps,
                    d["lgbm"][:h],
                    color=C_LGBM,
                    lw=1.2,
                    ls=":",
                    marker="D",
                    ms=3.0,
                    label="LightGBM",
                )
            ax.set_title(f"{sku}  ·  direct h=1..{h}", fontsize=9.5, color=INK, family=FONT, loc="left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.set_ylim(bottom=0)
            if i == n - 1:
                ax.set_xlabel("Horizon (weeks)", fontsize=9, color=INK, family=FONT)
            if j == 0:
                ax.set_ylabel("Week demand", fontsize=9, color=INK, family=FONT)
            if i == 0 and j == 1:
                ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, family=FONT, y=1.01)
    fig.tight_layout()
    _save_fig(fig, stem)


def _train_ds_h(
    cfg,
    X_train,
    y_train,
    sku_train_ids,
    X_val,
    y_val,
    sku_val_ids,
    sku_map,
    H: int,
    zero_rate: float,
    avg_nz: float,
    pos_weight: float,
    n_skus: int,
    epochs: int,
    batch_size: int,
    mh_stride: int = 1,
):
    Xtr_mh, ytr_mh, sktr_mh = build_mh_xy(X_train, y_train, sku_train_ids, H, stride=mh_stride)
    Xva_mh, yva_mh, skva_mh = build_mh_xy(X_val, y_val, sku_val_ids, H, stride=mh_stride)
    if len(Xva_mh) == 0:
        Xva_mh, yva_mh, skva_mh = Xtr_mh[-n_skus:], ytr_mh[-n_skus:], sktr_mh[-n_skus:]
    sktr = np.array([sku_map[str(s)] for s in sktr_mh], np.int32).reshape(-1, 1)
    skva = np.array([sku_map[str(s)] for s in skva_mh], np.int32).reshape(-1, 1)
    tr = split_components(Xtr_mh, cfg)
    va = split_components(Xva_mh, cfg)
    base = build_hierarchical_model_lightweight(
        n_temporal_features=len(cfg.trend_indices),
        n_fourier_features=len(cfg.seasonal_indices),
        n_holiday_features=max(len(cfg.holiday_indices), 0),
        n_lag_features=len(cfg.regressor_indices),
        n_skus=n_skus,
        hidden_dim=48,
        sku_embedding_dim=4,
        dropout_rate=0.23,
        use_intermittent=True,
        n_changepoints=15,
        use_sku=True,
        horizon=H,
    )
    model = train_gated(
        base,
        tr,
        va,
        ytr_mh,
        yva_mh,
        sktr,
        skva,
        zero_rate,
        avg_nz,
        pos_weight,
        epochs,
        batch_size,
        horizon_decay=0.95 if H > 1 else 1.0,
    )
    return model


def _train_lgbm(X_train, y_train, sku_train_ids, sku_map, H: int, seed: int, mh_stride: int = 1):
    import lightgbm as lgb

    Xtr_mh, ytr_mh, sktr_mh = build_mh_xy(X_train, y_train, sku_train_ids, H, stride=mh_stride)
    sktr = np.array([sku_map[str(s)] for s in sktr_mh], np.float32).reshape(-1, 1)
    Xtr = np.concatenate([Xtr_mh, sktr], axis=1)
    model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    )
    model.fit(Xtr, ytr_mh)
    return model


def run(args) -> None:
    data_dir = Path(args.data_dir)
    locked = json.loads(Path(args.sku_list).read_text())
    pool = list(dict.fromkeys(WEEKLY_PLOT_SKUS + [s for s in locked if s not in WEEKLY_PLOT_SKUS]))
    chosen_list = pool[: args.max_skus]
    plot_skus = [s for s in WEEKLY_PLOT_SKUS if s in chosen_list]
    print(f"Weekly train SKUs={len(chosen_list)} plot={plot_skus}")

    tf.keras.utils.set_random_seed(args.seed)
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    h_tr = _load_holiday(data_dir, "train", len(train_df))
    h_va = _load_holiday(data_dir, "val", len(val_df))
    h_te = _load_holiday(data_dir, "test", len(test_df))

    chosen = set(chosen_list)
    train_df, h_tr = filter_aligned(train_df, h_tr, chosen)
    val_df, h_va = filter_aligned(val_df, h_va, chosen)
    test_df, h_te = filter_aligned(test_df, h_te, chosen)
    for df in (train_df, val_df, test_df):
        df.sort_values(["id_var", "ds"], kind="mergesort", inplace=True)
        df.reset_index(drop=True, inplace=True)

    all_ids = (
        pd.concat(
            [train_df["id_var"], val_df["id_var"], test_df["id_var"]],
            ignore_index=True,
        )
        .astype(str)
        .unique()
    )
    cats = pd.Categorical(all_ids)
    sku_map = {str(k): i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return df["id_var"].astype(str).map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val = enc(train_df), enc(val_df)
    zero_rate = float((y_train == 0).mean())
    avg_nz = float(y_train[y_train > 0].mean()) if (y_train > 0).any() else 1.0
    pos_weight = class_balance_pos_weight(y_train)
    print(f"n_skus={n_skus} train_zero_rate={zero_rate:.3f}")

    cfg = load_feature_config(args.feature_config)
    hol_tr = h_tr if len(h_tr.columns) else None
    hol_va = h_va if len(h_va.columns) else None
    hol_te = h_te if len(h_te.columns) else None
    Xtr_df, states = cfg.create_features(train_df, hol_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, hol_va, prior_states=states, return_states=True)
    Xte_df, _ = cfg.create_features(test_df, hol_te, prior_states=states, return_states=True)
    X_train = Xtr_df.to_numpy(np.float32, copy=True)
    X_val = Xva_df.to_numpy(np.float32, copy=True)
    X_test = Xte_df.to_numpy(np.float32, copy=True)
    t_idx = cfg.trend_indices[0]
    tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    sk_tr = train_df["id_var"].astype(str).to_numpy()
    sk_va = val_df["id_var"].astype(str).to_numpy()
    sk_te = test_df["id_var"].astype(str).to_numpy()
    H = int(args.horizon)
    sku_test = enc(test_df)

    # Single Direct-MH train (locked weekly protocol); one-step uses horizon-1 head.
    print(f"=== DeepSequence direct MH (H={H}) ===")
    t0 = time.time()
    dsH = _train_ds_h(
        cfg,
        X_train,
        y_train,
        sk_tr,
        X_val,
        y_val,
        sk_va,
        sku_map,
        H,
        zero_rate,
        avg_nz,
        pos_weight,
        n_skus,
        args.epochs,
        args.batch_size,
    )
    print(f"  DS H={H} train {time.time()-t0:.1f}s")

    print("=== LightGBM multi-output MH ===")
    t0 = time.time()
    lgbH = _train_lgbm(X_train, y_train, sk_tr, sku_map, H, args.seed)
    print(f"  LGBM H={H} train {time.time()-t0:.1f}s")

    te_all = split_components(X_test, cfg)
    pred_all = dsH.predict([*te_all, sku_test], batch_size=2048, verbose=0)
    yhat_ds_mh = np.asarray(pred_all["final_forecast"], np.float32)
    if yhat_ds_mh.ndim == 1:
        yhat_ds_mh = yhat_ds_mh.reshape(-1, H)
    yhat_ds1 = yhat_ds_mh[:, 0]

    Xte_lgb = np.concatenate([X_test, sku_test.astype(np.float32)], axis=1)
    yhat_lgbm_mh = np.maximum(lgbH.predict(Xte_lgb), 0.0).astype(np.float32)
    if yhat_lgbm_mh.ndim == 1:
        yhat_lgbm_mh = yhat_lgbm_mh.reshape(-1, H)
    yhat_lgbm1 = yhat_lgbm_mh[:, 0]

    # TSB one-step: expand history week-by-week through test
    yhat_tsb1 = np.zeros(len(test_df), np.float32)
    for sku in chosen_list:
        m_te = sk_te == sku
        if not m_te.any():
            continue
        hist = np.concatenate(
            [
                train_df.loc[train_df["id_var"].astype(str) == sku, "Quantity"].to_numpy(np.float64),
                val_df.loc[val_df["id_var"].astype(str) == sku, "Quantity"].to_numpy(np.float64),
            ]
        )
        idxs = np.where(m_te)[0]
        y_te_sku = y_test[m_te]
        for j, row_i in enumerate(idxs):
            preds = croston_variants(hist, alpha=0.1)
            yhat_tsb1[row_i] = preds["tsb"]
            hist = np.append(hist, float(y_te_sku[j]))

    onestep = {}
    dump_os = {
        "protocol": "weekly_direct_mh_h1_over_test",
        "grain": "weekly_ISO_Monday",
        "seed": args.seed,
        "epochs": args.epochs,
        "max_skus": args.max_skus,
        "horizon_train": H,
        "feature_config": str(args.feature_config),
        "zero_rate_note": (
            "Weekly zero-rate on locked 800 is ≈0.65 vs daily ≈0.90; "
            "panels show mid-sparsity exemplars with visible variation."
        ),
        "train_zero_rate": zero_rate,
        "skus": {},
    }
    for sku in plot_skus:
        m = sk_te == sku
        if not m.any():
            print(f"  skip missing plot sku {sku}")
            continue
        y = y_test[m].astype(np.float64)
        d = {
            "dates": [str(x)[:10] for x in pd.to_datetime(test_df.loc[m, "ds"])],
            "y": y.tolist(),
            "ds": yhat_ds1[m].astype(np.float64).tolist(),
            "tsb": yhat_tsb1[m].astype(np.float64).tolist(),
            "lgbm": yhat_lgbm1[m].astype(np.float64).tolist(),
            "zero_rate_test": float((y == 0).mean()),
        }
        onestep[sku] = d
        dump_os["skus"][sku] = d
        print(
            f"  {sku}: zr={d['zero_rate_test']:.2f} "
            f"mean_y={y.mean():.1f} mean_ds={np.mean(d['ds']):.1f} "
            f"std_ds={np.std(d['ds']):.2f}"
        )

    stem_os = args.fig_prefix + "_onestep"
    _plot_onestep(
        onestep,
        "Weekly Direct-MH — one-step (h=1) over test weeks",
        stem_os,
    )
    (OUT / f"{stem_os}.json").write_text(json.dumps(dump_os, indent=2) + "\n")

    # Direct MH panels: first origin per plot SKU with ≥H future steps
    X_origin, y_true_mh, sk_origin = build_mh_xy(X_test, y_test, sk_te, H, stride=1)
    keep, seen = [], set()
    for i, s in enumerate(sk_origin):
        if s in seen:
            continue
        seen.add(s)
        keep.append(i)
    keep = np.asarray(keep, np.int64)
    X_origin = X_origin[keep]
    y_true_mh = y_true_mh[keep]
    sk_origin = sk_origin[keep]
    origin_idx = {str(s): i for i, s in enumerate(sk_origin)}

    plot_origins = [s for s in plot_skus if s in origin_idx]
    oi = np.asarray([origin_idx[s] for s in plot_origins], np.int64)
    yo = y_true_mh[oi]

    # Reuse full-test MH preds aligned to first test week of each plot SKU
    yhat_dsH = np.stack([yhat_ds_mh[sk_te == s][0] for s in plot_origins], axis=0)
    yhat_lgbmH = np.stack([yhat_lgbm_mh[sk_te == s][0] for s in plot_origins], axis=0)

    hist_y = []
    for sku in plot_origins:
        tr = train_df[train_df["id_var"].astype(str) == sku]
        va = val_df[val_df["id_var"].astype(str) == sku]
        hist_y.append(
            np.concatenate(
                [tr["Quantity"].to_numpy(np.float64), va["Quantity"].to_numpy(np.float64)]
            )
        )
    yhat_tsbH = classical_recursive(hist_y, H)["tsb"]

    h_short, h_long = 4, H
    horiz = {}
    dump_h = {
        "protocol": "weekly_direct_mh",
        "grain": "weekly_ISO_Monday",
        "seed": args.seed,
        "epochs": args.epochs,
        "max_skus": args.max_skus,
        "horizon": H,
        "h_short": h_short,
        "h_long": h_long,
        "feature_config": str(args.feature_config),
        "zero_rate_note": dump_os["zero_rate_note"],
        "skus": {},
    }
    for i, sku in enumerate(plot_origins):
        d = {
            "y": yo[i].astype(np.float64).tolist(),
            "ds": yhat_dsH[i].astype(np.float64).tolist(),
            "tsb": yhat_tsbH[i].astype(np.float64).tolist(),
            "lgbm": yhat_lgbmH[i].astype(np.float64).tolist(),
            "origin_date": str(
                pd.to_datetime(test_df.loc[sk_te == sku, "ds"]).iloc[0]
            )[:10],
        }
        horiz[sku] = d
        dump_h["skus"][sku] = d
        print(
            f"  direct {sku}: origin={d['origin_date']} "
            f"ds_h1={d['ds'][0]:.2f}"
        )

    stem_dir = args.fig_prefix + "_direct"
    _plot_direct(
        horiz,
        "Weekly direct multi-horizon forecasts from first test origin (DS / TSB / LGBM)",
        stem_dir,
        h_short,
        h_long,
    )
    (OUT / f"{stem_dir}.json").write_text(json.dumps(dump_h, indent=2) + "\n")
    print("weekly forecasts done")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_dir",
        default=str(ROOT / "ab_runs/weekly/panel_locked800"),
    )
    p.add_argument(
        "--feature_config",
        default=str(ROOT / "feature_config_weekly.yaml"),
    )
    p.add_argument(
        "--sku_list",
        default=str(ROOT / "ab_runs/recompare/sku_list_daily_data42.json"),
    )
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--fig_prefix", default="fig_forecast_weekly")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
