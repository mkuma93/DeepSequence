#!/usr/bin/env python3
"""Train a small locked-SKU subset and dump actual-vs-forecast line plots.

Daily: DeepSequence vs temporal transformer (TST).
Car Parts: DeepSequence vs TSB.

Outputs under paper_figures/:
  fig_forecast_daily_*.{png,pdf,json}
  fig_forecast_carparts_*.{png,pdf,json}

Binary-holiday qualitative (forecast-only; does not touch locked v1.6):
  python paper_figures/make_forecast_line_plots.py --only daily --epochs 30 \\
    --feature_config_daily feature_config_daily_binary_holiday.yaml \\
    --fig_prefix_daily fig_forecast_daily_binary_hol --holiday_markers 1
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

from classical_intermittent import croston_variants
from eval_helpers import (
    build_transformer,
    filter_aligned,
    resolve_sku_zero_rates,
    select_eval_skus,
    split_components,
)
from feature_config_loader import load_feature_config
from holiday_calendar import RETAIL_WINDOW_KEYS, binary_holiday_features
from multihorizon_rollout import (
    build_sku_timelines,
    collect_origins,
    rollout_sequence,
    rollout_tabular,
)
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss

INK = "#1f2933"
C_ACT = "#37474f"
C_DS = "#1e88e5"
C_BASE = "#ef6c00"
C_HOL = "#c62828"
FONT = "DejaVu Sans"

# Locked-list intermittent exemplars (chosen for sparse but visible sales).
DAILY_PLOT_SKUS = [
    "United Kingdom_22155",
    "United Kingdom_21870",
    "United Kingdom_85159A",
]
CARPARTS_PLOT_SKUS = ["T1851", "T1979", "T1746"]


def _save_fig(fig, stem: str):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {png.name} + {pdf.name}")


def _attach_binary_holidays(hol_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Ensure binary is_* columns exist on a days_from_* holiday frame."""
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


def _mark_holidays(ax, dates, mark_dates):
    if not mark_dates:
        return
    dset = set(mark_dates)
    for d in pd.to_datetime(dates):
        if str(d)[:10] in dset:
            ax.axvline(d, color=C_HOL, alpha=0.22, lw=1.0, zorder=0)


def _build_1step_windows(train_df, val_df, X_train, X_val, lookback: int):
    metas, feats, offset = [], [], 0
    for split, df, X in (("train", train_df, X_train), ("val", val_df, X_val)):
        y = df["Quantity"].to_numpy(np.float32).reshape(-1, 1)
        block = np.concatenate([y, X.astype(np.float32)], axis=1)
        feats.append(block)
        n = len(df)
        metas.append(
            pd.DataFrame(
                {
                    "id_var": df["id_var"].astype(str).to_numpy(),
                    "ds": pd.to_datetime(df["ds"]),
                    "y": df["Quantity"].to_numpy(np.float32),
                    "split": split,
                    "_pos": np.arange(offset, offset + n, dtype=np.int64),
                }
            )
        )
        offset += n
    feat_all = np.concatenate(feats, axis=0)
    meta = pd.concat(metas, ignore_index=True)
    meta = meta.sort_values(["id_var", "ds"], kind="mergesort").reset_index(drop=True)
    n_channels = feat_all.shape[1]
    xs, ys, skus, splits = [], [], [], []
    for sku, g in meta.groupby("id_var", sort=False):
        pos = g["_pos"].to_numpy()
        arr = feat_all[pos]
        y = g["y"].to_numpy(np.float32)
        sp = g["split"].to_numpy()
        n = len(g)
        if n <= lookback:
            continue
        for t in range(lookback, n):
            xs.append(arr[t - lookback : t])
            ys.append(y[t])
            skus.append(sku)
            splits.append(sp[t])
    X = (
        np.stack(xs).astype(np.float32)
        if xs
        else np.zeros((0, lookback, n_channels), np.float32)
    )
    return X, np.asarray(ys, np.float32), np.asarray(skus), np.asarray(splits), n_channels


def _train_ds(cfg, tr, va, y_train, y_val, sku_train, sku_val, zero_rate, n_skus, epochs, batch):
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
        loss_recipe="three_term",
        alpha_bce=0.2,
        w_gated=1.0,
        w_mag=1.0,
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
                monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
            )
        ],
        verbose=2,
    )
    return model


def _train_tst(Xseq, yseq, sku_seq, split_seq, n_skus, n_channels, lookback, zero_rate, epochs, batch, sku_rates):
    from deepsequence_hierarchical_attention.losses import three_term_loss_config

    model = build_transformer(lookback, n_skus, n_channels=n_channels)
    cfg = three_term_loss_config(zero_rate, alpha_bce=0.2, w_gated=1.0, w_mag=1.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0025),
        loss=cfg["losses"],
        loss_weights=cfg["weights"],
    )
    m_tr = split_seq == "train"
    m_va = split_seq == "val"
    ytr_d = {
        "final_forecast": yseq[m_tr].reshape(-1, 1),
        "base_forecast": yseq[m_tr].reshape(-1, 1),
        "non_zero_probability": (yseq[m_tr] > 0).astype(np.float32).reshape(-1, 1),
    }
    yva_d = {
        "final_forecast": yseq[m_va].reshape(-1, 1),
        "base_forecast": yseq[m_va].reshape(-1, 1),
        "non_zero_probability": (yseq[m_va] > 0).astype(np.float32).reshape(-1, 1),
    }
    model.fit(
        [Xseq[m_tr], sku_seq[m_tr]],
        ytr_d,
        validation_data=([Xseq[m_va], sku_seq[m_va]], yva_d),
        epochs=epochs,
        batch_size=batch,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
            )
        ],
        verbose=2,
    )
    return model


def _plot_onestep_panel(series_dict, title, stem, baseline_name, holiday_marks=False):
    n = len(series_dict)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 2.6 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (sku, d) in zip(axes, series_dict.items()):
        dates = pd.to_datetime(d["dates"])
        if holiday_marks:
            _mark_holidays(ax, dates, d.get("holiday_dates") or [])
        ax.plot(dates, d["y"], color=C_ACT, lw=1.4, label="Actual", drawstyle="steps-mid")
        ax.plot(dates, d["ds"], color=C_DS, lw=1.5, label="DeepSequence", alpha=0.95)
        ax.plot(dates, d["baseline"], color=C_BASE, lw=1.3, ls="--", label=baseline_name, alpha=0.95)
        ax.set_ylabel("Demand", fontsize=9, color=INK, family=FONT)
        ax.set_title(sku, fontsize=10, color=INK, family=FONT, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Date", fontsize=9, color=INK, family=FONT)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, family=FONT, y=1.01)
    fig.tight_layout()
    _save_fig(fig, stem)


def _plot_horizon_panel(series_dict, title, stem, baseline_name, h_short, h_long):
    n = len(series_dict)
    fig, axes = plt.subplots(n, 2, figsize=(12.5, 2.55 * n), sharey=False)
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
                d["baseline"][:h],
                color=C_BASE,
                lw=1.3,
                ls="--",
                marker="^",
                ms=3.2,
                label=baseline_name,
            )
            ax.set_title(f"{sku}  ·  h=1..{h}", fontsize=9.5, color=INK, family=FONT, loc="left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.set_ylim(bottom=0)
            if i == n - 1:
                ax.set_xlabel("Horizon step", fontsize=9, color=INK, family=FONT)
            if j == 0:
                ax.set_ylabel("Demand", fontsize=9, color=INK, family=FONT)
            if i == 0 and j == 1:
                ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, family=FONT, y=1.01)
    fig.tight_layout()
    _save_fig(fig, stem)


def run_daily(args):
    data_dir = Path(args.data_dir)
    locked = json.loads(Path(args.sku_list_daily).read_text())
    # train pool: plot SKUs first, then fill from locked list
    pool = list(dict.fromkeys(DAILY_PLOT_SKUS + [s for s in locked if s not in DAILY_PLOT_SKUS]))
    chosen_list = pool[: args.max_skus]
    plot_skus = [s for s in DAILY_PLOT_SKUS if s in chosen_list]
    print(f"Daily train SKUs={len(chosen_list)} plot={plot_skus}")

    tf.keras.utils.set_random_seed(args.seed)
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    h_tr = pd.read_csv(data_dir / "holiday_features_train.csv")
    h_va = pd.read_csv(data_dir / "holiday_features_val.csv")
    h_te = pd.read_csv(data_dir / "holiday_features_test.csv")
    chosen = set(chosen_list)
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

    cfg_path = args.feature_config_daily
    cfg = load_feature_config(cfg_path) if cfg_path else load_feature_config()
    print(
        f"feature_config={cfg_path or 'default'} "
        f"n_feat={cfg.total_features} n_holiday={len(cfg.holiday_indices)} "
        f"binary={cfg.binary_holiday_names}"
    )
    stem_os = args.fig_prefix_daily + "_onestep"
    stem_rec = args.fig_prefix_daily + "_recursive"
    use_hol_marks = bool(cfg.binary_holiday_names) or bool(args.holiday_markers)

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
    epoch = pd.Timestamp("1970-01-01")
    raw_tr = (pd.to_datetime(train_df["ds"]) - epoch).dt.days.to_numpy(np.float64)
    tmin_raw, tmax_raw = float(raw_tr.min()), float(raw_tr.max())
    span_raw = max(tmax_raw - tmin_raw, 1.0)
    X_train_n, X_val_n, X_test_n = X_train.copy(), X_val.copy(), X_test.copy()
    for X in (X_train_n, X_val_n, X_test_n):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span
    tr, va, te = split_components(X_train_n, cfg), split_components(X_val_n, cfg), split_components(X_test_n, cfg)

    print("=== train DeepSequence (daily) ===")
    ds_model = _train_ds(
        cfg, tr, va, y_train, y_val, sku_train, sku_val, zero_rate, n_skus, args.epochs, args.batch_size
    )

    print("=== train TST (daily) ===")
    Xseq, yseq, sku_seq_raw, split_seq, n_channels = _build_1step_windows(
        train_df, val_df, X_train_n, X_val_n, args.lookback_daily
    )
    sku_seq = np.array([sku_map[str(s)] for s in sku_seq_raw], dtype=np.int32).reshape(-1, 1)
    _, sku_rates = resolve_sku_zero_rates(y_train, sku_train, n_skus=n_skus)
    tst_model = _train_tst(
        Xseq,
        yseq,
        sku_seq,
        split_seq,
        n_skus,
        n_channels,
        args.lookback_daily,
        zero_rate,
        args.epochs,
        args.batch_size,
        sku_rates,
    )

    # one-step test preds
    sku_test = enc(test_df)
    ds_pred = ds_model.predict([*te, sku_test], batch_size=4096, verbose=0)
    yhat_ds = np.asarray(ds_pred["final_forecast"]).reshape(-1)

    # sequence one-step on test: build windows ending at each test day
    panel = pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")],
        ignore_index=True,
    )
    hol = pd.concat([h_tr, h_va, h_te], ignore_index=True)
    X_all_df, _ = cfg.create_features(panel, hol, return_states=True)
    X_all = X_all_df.to_numpy(np.float32)
    X_all[:, t_idx] = (X_all[:, t_idx] - tmin) / span
    meta = panel.assign(_pos=np.arange(len(panel))).sort_values(["id_var", "ds"], kind="mergesort")
    yhat_tst = np.full(len(test_df), np.nan, np.float32)
    test_index = { (str(r.id_var), pd.Timestamp(r.ds)): i for i, r in test_df.reset_index(drop=True).iterrows() }
    for sku, g in meta.groupby("id_var", sort=False):
        pos = g["_pos"].to_numpy()
        arr = np.concatenate(
            [panel.loc[pos, "Quantity"].to_numpy(np.float32).reshape(-1, 1), X_all[pos]],
            axis=1,
        )
        dates = pd.to_datetime(g["ds"]).tolist()
        splits = g["split"].to_numpy()
        for t in range(args.lookback_daily, len(g)):
            if splits[t] != "test":
                continue
            key = (str(sku), pd.Timestamp(dates[t]))
            if key not in test_index:
                continue
            win = arr[t - args.lookback_daily : t][None, ...]
            sk = np.array([[sku_map[str(sku)]]], np.int32)
            pred = tst_model.predict([win, sk], verbose=0)
            yhat_tst[test_index[key]] = float(np.asarray(pred["final_forecast"]).reshape(-1)[0])

    hol_block_names = cfg.holiday_block_names
    n_dist = len(cfg.holiday_names)
    onestep = {}
    dump_onestep = {
        "protocol": "one_step_test",
        "seed": args.seed,
        "epochs": args.epochs,
        "feature_config": str(cfg_path or "default"),
        "binary_holiday_features": cfg.binary_holiday_names,
        "skus": {},
    }
    for sku in plot_skus:
        m = test_df["id_var"].astype(str).to_numpy() == sku
        if not m.any():
            continue
        dates = [str(x)[:10] for x in pd.to_datetime(test_df.loc[m, "ds"])]
        hol_m = h_te.loc[m, hol_block_names].to_numpy(np.float32) if hol_block_names else None
        if use_hol_marks and hol_m is not None and hol_m.shape[1] > n_dist:
            hol_dates = [
                d for d, row in zip(dates, hol_m) if float(row[n_dist:].max()) > 0.5
            ]
        elif use_hol_marks and hol_m is not None:
            hol_dates = [
                d for d, row in zip(dates, hol_m) if float(np.abs(row).min()) < 0.5
            ]
        else:
            hol_dates = []
        d = {
            "dates": dates,
            "y": test_df.loc[m, "Quantity"].to_numpy(np.float64).tolist(),
            "ds": yhat_ds[m].astype(np.float64).tolist(),
            "baseline": np.nan_to_num(yhat_tst[m], nan=0.0).astype(np.float64).tolist(),
            "holiday_dates": hol_dates,
        }
        onestep[sku] = d
        dump_onestep["skus"][sku] = d
    title_os = "Daily intermittent demand — one-step forecasts (test window)"
    if cfg.binary_holiday_names:
        title_os = "Daily one-step forecasts — binary holidays ON (test window)"
    _plot_onestep_panel(
        onestep,
        title_os,
        stem_os,
        "TST",
        holiday_marks=use_hol_marks,
    )
    (OUT / f"{stem_os}.json").write_text(json.dumps(dump_onestep, indent=2))

    # recursive from one origin per plot SKU
    timelines = build_sku_timelines(panel, hol, hol_block_names or cfg.holiday_names)
    origin_mask = {}
    for sku, g in panel.groupby(panel["id_var"].astype(str), sort=False):
        g = g.sort_values("ds", kind="mergesort")
        origin_mask[str(sku)] = (g["split"].to_numpy() == "test")
    H = args.horizon_daily
    origins_all = collect_origins(
        timelines,
        sku_map,
        horizon=H,
        origin_split_mask=origin_mask,
        max_origins_per_sku=1,
        seed=args.seed,
    )
    origins = [o for o in origins_all if o[0] in plot_skus]
    print(f"daily recursive origins for plots: {len(origins)} H={H}")

    def ds_predict(X, sku):
        parts = split_components(X, cfg)
        pred = ds_model.predict([*parts, sku], batch_size=512, verbose=0)
        return (
            np.asarray(pred["final_forecast"]).reshape(-1),
            np.asarray(pred["non_zero_probability"]).reshape(-1),
        )

    def tst_predict(windows, sku):
        pred = tst_model.predict([windows, sku], batch_size=512, verbose=0)
        return (
            np.asarray(pred["final_forecast"]).reshape(-1),
            np.asarray(pred["non_zero_probability"]).reshape(-1),
        )

    roll_ds = rollout_tabular(
        timelines, origins, sku_map, ds_predict, cfg.lag_periods, tmin_raw, span_raw, H
    )
    roll_tst = rollout_sequence(
        timelines,
        origins,
        sku_map,
        tst_predict,
        cfg.lag_periods,
        tmin_raw,
        span_raw,
        H,
        lookback=args.lookback_daily,
    )

    h_short, h_long = 7, min(28, H)
    horiz = {}
    dump_h = {
        "protocol": "recursive_rollout",
        "seed": args.seed,
        "epochs": args.epochs,
        "feature_config": str(cfg_path or "default"),
        "binary_holiday_features": cfg.binary_holiday_names,
        "horizon": H,
        "h_short": h_short,
        "h_long": h_long,
        "skus": {},
    }
    for i, (sku, t_idx) in enumerate(origins):
        d = {
            "y": roll_ds["y_true"][i].astype(np.float64).tolist(),
            "ds": roll_ds["yhat"][i].astype(np.float64).tolist(),
            "baseline": roll_tst["yhat"][i].astype(np.float64).tolist(),
            "origin_idx": int(t_idx),
            "origin_date": str(pd.Timestamp(timelines[sku].dates[t_idx]))[:10],
        }
        horiz[sku] = d
        dump_h["skus"][sku] = d
    title_rec = "Daily recursive forecasts from a locked test origin (DS vs TST)"
    if cfg.binary_holiday_names:
        title_rec = "Daily recursive forecasts — binary holidays ON (DS vs TST)"
    _plot_horizon_panel(
        horiz,
        title_rec,
        stem_rec,
        "TST",
        h_short,
        h_long,
    )
    (OUT / f"{stem_rec}.json").write_text(json.dumps(dump_h, indent=2))
    print("daily forecasts done")


def run_carparts(args):
    data_dir = Path(args.carparts_dir)
    locked = json.loads(Path(args.sku_list_carparts).read_text())
    pool = list(dict.fromkeys(CARPARTS_PLOT_SKUS + [s for s in locked if s not in CARPARTS_PLOT_SKUS]))
    chosen_list = pool[: args.max_skus]
    plot_skus = [s for s in CARPARTS_PLOT_SKUS if s in chosen_list]
    print(f"CarParts train SKUs={len(chosen_list)} plot={plot_skus}")

    tf.keras.utils.set_random_seed(args.seed)
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    h_tr = pd.read_csv(data_dir / "holiday_features_train.csv")
    h_va = pd.read_csv(data_dir / "holiday_features_val.csv")
    h_te = pd.read_csv(data_dir / "holiday_features_test.csv")
    chosen = set(chosen_list)
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

    cfg = load_feature_config(str(ROOT / "feature_config_monthly.yaml"))
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
    # Monthly raw scale (year*12+month), matching eval_public_carparts_mh_all
    raw_mi = (
        pd.to_datetime(train_df["ds"]).dt.year * 12 + pd.to_datetime(train_df["ds"]).dt.month
    ).to_numpy(np.float64)
    tmin_raw = float(raw_mi.min())
    span_raw = max(float(raw_mi.max()) - tmin_raw, 1.0)
    X_train_n, X_val_n, X_test_n = X_train.copy(), X_val.copy(), X_test.copy()
    for X in (X_train_n, X_val_n, X_test_n):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span
    tr, va, te = split_components(X_train_n, cfg), split_components(X_val_n, cfg), split_components(X_test_n, cfg)

    print("=== train DeepSequence (carparts) ===")
    ds_model = _train_ds(
        cfg, tr, va, y_train, y_val, sku_train, sku_val, zero_rate, n_skus, args.epochs, args.batch_size
    )

    # one-step DS + TSB on test
    sku_test = enc(test_df)
    ds_pred = ds_model.predict([*te, sku_test], batch_size=1024, verbose=0)
    yhat_ds = np.asarray(ds_pred["final_forecast"]).reshape(-1)

    hist = pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values(
        ["id_var", "ds"], kind="mergesort"
    )
    series = {}
    for sku, g in hist.groupby("id_var", sort=False):
        series[str(sku)] = {
            "ds": pd.to_datetime(g["ds"]).to_numpy(),
            "y": g["Quantity"].to_numpy(np.float64),
        }
    yhat_tsb = np.zeros(len(test_df), np.float32)
    test_r = test_df.reset_index(drop=True)
    for i, row in test_r.iterrows():
        s = series[str(row.id_var)]
        mask = s["ds"] < np.datetime64(pd.Timestamp(row.ds))
        yhat_tsb[i] = croston_variants(s["y"][mask])["tsb"]

    onestep = {}
    dump_onestep = {"protocol": "one_step_test", "seed": args.seed, "skus": {}}
    for sku in plot_skus:
        m = test_df["id_var"].astype(str).to_numpy() == sku
        if not m.any():
            continue
        d = {
            "dates": [str(x)[:10] for x in pd.to_datetime(test_df.loc[m, "ds"])],
            "y": test_df.loc[m, "Quantity"].to_numpy(np.float64).tolist(),
            "ds": yhat_ds[m].astype(np.float64).tolist(),
            "baseline": yhat_tsb[m].astype(np.float64).tolist(),
        }
        onestep[sku] = d
        dump_onestep["skus"][sku] = d
    _plot_onestep_panel(
        onestep,
        "Car Parts (monthly) — one-step forecasts (test window)",
        "fig_forecast_carparts_onestep",
        "TSB",
    )
    (OUT / "fig_forecast_carparts_onestep.json").write_text(json.dumps(dump_onestep, indent=2))

    # Recursive MH with monthly feature assembler (not daily rollout_tabular).
    from eval_public_carparts_mh_all import assemble_monthly_row
    from deepsequence_hierarchical_attention.intermittent_features import empty_state

    panel = pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")],
        ignore_index=True,
    )
    H = args.horizon_carparts
    lag_periods = cfg.lag_periods
    horiz = {}
    dump_h = {
        "protocol": "recursive_monthly_from_pre_test_origin",
        "seed": args.seed,
        "horizon": H,
        "skus": {},
    }
    for sku in plot_skus:
        g = panel[panel["id_var"].astype(str) == sku].sort_values("ds", kind="mergesort")
        if len(g) <= H:
            continue
        dates = pd.to_datetime(g["ds"]).tolist()
        ys = g["Quantity"].to_numpy(np.float64)
        splits = g["split"].to_numpy()
        test_pos = [i for i, s in enumerate(splits) if s == "test"]
        if len(test_pos) < H:
            continue
        # Origin = last observation before first test month
        first_test = test_pos[0]
        origin_i = first_test - 1
        if origin_i < 0:
            continue
        st = empty_state(max_lag=max(lag_periods), rate_window=12)
        for d, q in zip(dates[: origin_i + 1], ys[: origin_i + 1]):
            st.update(pd.Timestamp(d), float(q))
        y_true = ys[first_test : first_test + H]
        yhat_ds = np.zeros(H, np.float64)
        yhat_tsb = np.zeros(H, np.float64)
        y_hist_tsb = list(ys[: origin_i + 1].astype(float))
        for h in range(H):
            date = pd.Timestamp(dates[first_test + h])
            feat = assemble_monthly_row(date, st, lag_periods, tmin_raw, span_raw)
            Xrow = feat.reshape(1, -1).astype(np.float32)
            parts = split_components(Xrow, cfg)
            sk = np.array([[sku_map[sku]]], np.int32)
            pred = ds_model.predict([*parts, sk], verbose=0)
            yh = float(np.asarray(pred["final_forecast"]).reshape(-1)[0])
            yhat_ds[h] = max(yh, 0.0)
            st.update(date, yhat_ds[h])
            tsb = croston_variants(np.asarray(y_hist_tsb, float))["tsb"]
            yhat_tsb[h] = tsb
            y_hist_tsb.append(tsb)
        d = {
            "y": y_true.tolist(),
            "ds": yhat_ds.tolist(),
            "baseline": yhat_tsb.tolist(),
            "origin_date": str(dates[origin_i])[:10],
            "test_start": str(dates[first_test])[:10],
        }
        horiz[sku] = d
        dump_h["skus"][sku] = d

    if horiz:
        _plot_horizon_panel(
            horiz,
            "Car Parts recursive forecasts from pre-test origin (DS vs TSB)",
            "fig_forecast_carparts_recursive",
            "TSB",
            2,
            H,
        )
        (OUT / "fig_forecast_carparts_recursive.json").write_text(json.dumps(dump_h, indent=2))
    print("carparts forecasts done")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default=os.environ.get(
            "DEEPSEQUENCE_DATA_DIR",
            "/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
        ),
    )
    p.add_argument(
        "--carparts_dir",
        default=str(ROOT / "public_data/car_parts/panel"),
    )
    p.add_argument("--sku_list_daily", default=str(ROOT / "ab_runs/recompare/sku_list_daily_data42.json"))
    p.add_argument(
        "--sku_list_carparts", default=str(ROOT / "ab_runs/recompare/sku_list_carparts_data42.json")
    )
    p.add_argument("--max_skus", type=int, default=40)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lookback_daily", type=int, default=14)
    p.add_argument("--horizon_daily", type=int, default=28)
    p.add_argument("--horizon_carparts", type=int, default=6)
    p.add_argument("--only", choices=("daily", "carparts", "both"), default="both")
    p.add_argument(
        "--feature_config_daily",
        default=None,
        help="Override daily feature YAML (e.g. feature_config_daily_binary_holiday.yaml).",
    )
    p.add_argument(
        "--fig_prefix_daily",
        default="fig_forecast_daily",
        help="Stem prefix for daily figs (e.g. fig_forecast_daily_binary_hol).",
    )
    p.add_argument(
        "--holiday_markers",
        type=int,
        default=0,
        help="Force holiday axvlines on one-step plots (1/0). Auto-on when binaries present.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    if args.only in ("daily", "both"):
        run_daily(args)
    if args.only in ("carparts", "both"):
        run_carparts(args)
    print(f"all done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
