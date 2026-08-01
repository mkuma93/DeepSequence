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

Country-calendar + binary qualitative (rebuilds days_from_* per sku prefix):
  python paper_figures/make_forecast_line_plots.py --only daily --epochs 30 \\
    --feature_config_daily feature_config_daily_country_holiday.yaml \\
    --fig_prefix_daily fig_forecast_daily_country_hol --holiday_markers 1

Additive vs multiplicative Level-2 combine (same SKUs/seed; no TST):
  python paper_figures/make_forecast_line_plots.py --only daily_combine \\
    --epochs 20 --feature_config_daily feature_config_daily_country_holiday.yaml \\
    --fig_prefix_daily fig_forecast_daily_mult --holiday_markers 1

Monthly Car Parts country months_from + month_has (year-scoped; locked YAML stays none):
  python paper_figures/make_forecast_line_plots.py --only carparts --epochs 20 \\
    --feature_config_monthly feature_config_monthly_country_holiday.yaml \\
    --fig_prefix_carparts fig_forecast_carparts_country_hol --holiday_markers 1
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

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from deepsequence_hierarchical_attention.eval.classical import croston_variants
from deepsequence_hierarchical_attention.eval.helpers import (
    build_transformer,
    filter_aligned,
    resolve_sku_zero_rates,
    select_eval_skus,
    split_components,
)
from deepsequence_hierarchical_attention.data.feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.holidays.calendar import (
    RETAIL_WINDOW_KEYS,
    binary_holiday_features,
    build_country_holiday_distances,
    build_country_month_holiday_features,
    country_from_sku_id,
    month_has_holiday_features,
    months_from_holiday_features,
)
from deepsequence_hierarchical_attention.eval.multihorizon_rollout import (
    build_sku_timelines,
    collect_origins,
    rollout_sequence,
    rollout_tabular,
)
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.training.adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss

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


def _country_calendar_enabled(cfg) -> bool:
    meta = cfg.config.get("metadata", {}) or {}
    mode = str(meta.get("holiday_calendar", "static")).lower()
    return mode in ("country", "per_country", "country_aware")


def _rebuild_holidays_for_split(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Build days_from_* (+ optional is_*) from country calendars for one split."""
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


def _corr_yhat_holiday(yhat, dates, hol_dates) -> float:
    if not hol_dates:
        return float("nan")
    dset = set(hol_dates)
    flag = np.asarray([1.0 if d in dset else 0.0 for d in dates], dtype=np.float64)
    y = np.asarray(yhat, dtype=np.float64)
    if flag.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(y, flag)[0, 1])


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
    component_combine="additive",
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
        component_combine=component_combine,
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


def _series_cv(yhat) -> float:
    y = np.asarray(yhat, dtype=np.float64)
    mu = float(np.mean(np.abs(y)))
    if mu < 1e-12:
        return 0.0
    return float(np.std(y) / mu)


def _iwmae(y_true, y_hat) -> float:
    """Tiny intermittent-weighted MAE: weight positive days more."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_hat, dtype=np.float64).reshape(-1)
    w = np.where(y > 0, 1.0, 0.25)
    return float(np.average(np.abs(y - p), weights=w))


def _plot_add_vs_mul_panel(series_dict, title, stem, holiday_marks=False):
    n = len(series_dict)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 2.7 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (sku, d) in zip(axes, series_dict.items()):
        dates = pd.to_datetime(d["dates"])
        if holiday_marks:
            _mark_holidays(ax, dates, d.get("holiday_dates") or [])
        ax.plot(dates, d["y"], color=C_ACT, lw=1.4, label="Actual", drawstyle="steps-mid")
        ax.plot(dates, d["additive"], color=C_DS, lw=1.5, label="Additive mix", alpha=0.95)
        ax.plot(
            dates,
            d["multiplicative"],
            color="#8e24aa",
            lw=1.5,
            ls="--",
            label="Multiplicative mix",
            alpha=0.95,
        )
        ax.set_ylabel("Demand", fontsize=9, color=INK, family=FONT)
        cv_a = d.get("cv_additive")
        cv_m = d.get("cv_multiplicative")
        subtitle = sku
        if cv_a is not None and cv_m is not None:
            subtitle = f"{sku}  ·  CV add={cv_a:.2f} mul={cv_m:.2f}"
        ax.set_title(subtitle, fontsize=10, color=INK, family=FONT, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Date", fontsize=9, color=INK, family=FONT)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, family=FONT, y=1.01)
    fig.tight_layout()
    _save_fig(fig, stem)


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
        f"binary={cfg.binary_holiday_names} "
        f"holiday_calendar={cfg.config.get('metadata', {}).get('holiday_calendar', 'static')}"
    )
    stem_os = args.fig_prefix_daily + "_onestep"
    stem_rec = args.fig_prefix_daily + "_recursive"
    use_hol_marks = bool(cfg.binary_holiday_names) or bool(args.holiday_markers)
    use_country = _country_calendar_enabled(cfg)

    if use_country:
        scope = cfg.config.get("metadata", {}).get(
            "holiday_distance_scope",
            cfg.config.get("metadata", {}).get("distance_scope", "year"),
        )
        print(
            f"rebuilding holiday distances from per-country calendars "
            f"(sku_id prefix; distance_scope={scope})"
        )
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
        component_combine=getattr(args, "component_combine", "additive"),
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
        "holiday_calendar": cfg.config.get("metadata", {}).get("holiday_calendar", "static"),
        "holiday_distance_scope": cfg.config.get("metadata", {}).get(
            "holiday_distance_scope",
            cfg.config.get("metadata", {}).get("distance_scope", "year"),
        ),
        "skus": {},
        "corr_yhat_vs_holiday_flag": {},
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
        yhat_sku = yhat_ds[m].astype(np.float64)
        corr = _corr_yhat_holiday(yhat_sku, dates, hol_dates)
        d = {
            "dates": dates,
            "y": test_df.loc[m, "Quantity"].to_numpy(np.float64).tolist(),
            "ds": yhat_sku.tolist(),
            "baseline": np.nan_to_num(yhat_tst[m], nan=0.0).astype(np.float64).tolist(),
            "holiday_dates": hol_dates,
            "corr_yhat_vs_holiday_flag": corr,
        }
        onestep[sku] = d
        dump_onestep["skus"][sku] = d
        dump_onestep["corr_yhat_vs_holiday_flag"][sku] = corr
        print(f"  {sku}: corr(yhat, holiday_flag)={corr:.4f} n_hol_days={len(hol_dates)}")
    title_os = "Daily intermittent demand — one-step forecasts (test window)"
    if use_country:
        title_os = "Daily one-step forecasts — country calendars + binary holidays"
    elif cfg.binary_holiday_names:
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
        "holiday_calendar": cfg.config.get("metadata", {}).get("holiday_calendar", "static"),
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
    if use_country:
        title_rec = "Daily recursive forecasts — country calendars + binary holidays"
    elif cfg.binary_holiday_names:
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


def run_daily_combine_compare(args):
    """Train additive vs multiplicative Level-2 combine on the same daily subset."""
    data_dir = Path(args.data_dir)
    locked = json.loads(Path(args.sku_list_daily).read_text())
    pool = list(dict.fromkeys(DAILY_PLOT_SKUS + [s for s in locked if s not in DAILY_PLOT_SKUS]))
    chosen_list = pool[: args.max_skus]
    plot_skus = [s for s in DAILY_PLOT_SKUS if s in chosen_list]
    print(f"Daily combine-compare train SKUs={len(chosen_list)} plot={plot_skus}")

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
    stem_os = args.fig_prefix_daily + "_onestep"
    stem_rec = args.fig_prefix_daily + "_recursive"
    use_hol_marks = bool(cfg.binary_holiday_names) or bool(args.holiday_markers)
    use_country = _country_calendar_enabled(cfg)

    if use_country:
        scope = cfg.config.get("metadata", {}).get(
            "holiday_distance_scope",
            cfg.config.get("metadata", {}).get("distance_scope", "year"),
        )
        print(
            f"rebuilding holiday distances from per-country calendars "
            f"(sku_id prefix; distance_scope={scope})"
        )
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
    epoch = pd.Timestamp("1970-01-01")
    raw_tr = (pd.to_datetime(train_df["ds"]) - epoch).dt.days.to_numpy(np.float64)
    tmin_raw, tmax_raw = float(raw_tr.min()), float(raw_tr.max())
    span_raw = max(tmax_raw - tmin_raw, 1.0)
    X_train_n, X_val_n, X_test_n = X_train.copy(), X_val.copy(), X_test.copy()
    for X in (X_train_n, X_val_n, X_test_n):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span
    tr = split_components(X_train_n, cfg)
    va = split_components(X_val_n, cfg)
    te = split_components(X_test_n, cfg)

    models = {}
    for mode in ("additive", "multiplicative"):
        print(f"=== train DeepSequence ({mode} combine) ===")
        tf.keras.utils.set_random_seed(args.seed)
        models[mode] = _train_ds(
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
            component_combine=mode,
        )

    sku_test = enc(test_df)
    preds = {}
    for mode, model in models.items():
        pred = model.predict([*te, sku_test], batch_size=4096, verbose=0)
        preds[mode] = np.asarray(pred["final_forecast"]).reshape(-1)

    hol_block_names = cfg.holiday_block_names
    n_dist = len(cfg.holiday_names)
    onestep = {}
    dump_onestep = {
        "protocol": "one_step_test_additive_vs_multiplicative",
        "seed": args.seed,
        "formula": (
            "b_pre = softplus(e_T) * Π_{k in {S,H,R}} max(eps, 1 + alpha_k * e_k); "
            "then softplus magnitude Dense + gate p*b (unchanged)"
        ),
        "component_combine": ["additive", "multiplicative"],
        "skus": {},
        "summary": {},
    }
    iw_add_all, iw_mul_all = [], []
    for sku in plot_skus:
        m = test_df["id_var"].astype(str).to_numpy() == sku
        if not m.any():
            continue
        y = test_df.loc[m, "Quantity"].to_numpy(np.float64)
        add = preds["additive"][m].astype(np.float64)
        mul = preds["multiplicative"][m].astype(np.float64)
        hol_dates = []
        if use_hol_marks and hol_block_names:
            # Mark days with any binary holiday flag if present.
            hol_cols = [c for c in hol_block_names if c.startswith("is_")]
            if hol_cols:
                te_hol = h_te.loc[m]
                for i, row in te_hol.reset_index(drop=True).iterrows():
                    if any(float(row.get(c, 0) or 0) > 0.5 for c in hol_cols):
                        hol_dates.append(str(pd.to_datetime(test_df.loc[m, "ds"].iloc[i]))[:10])
        d = {
            "dates": [str(x)[:10] for x in pd.to_datetime(test_df.loc[m, "ds"])],
            "y": y.tolist(),
            "additive": add.tolist(),
            "multiplicative": mul.tolist(),
            "holiday_dates": hol_dates,
            "cv_additive": _series_cv(add),
            "cv_multiplicative": _series_cv(mul),
            "iwmae_additive": _iwmae(y, add),
            "iwmae_multiplicative": _iwmae(y, mul),
        }
        onestep[sku] = d
        dump_onestep["skus"][sku] = d
        iw_add_all.append(d["iwmae_additive"])
        iw_mul_all.append(d["iwmae_multiplicative"])

    dump_onestep["summary"] = {
        "mean_iwmae_additive": float(np.mean(iw_add_all)) if iw_add_all else None,
        "mean_iwmae_multiplicative": float(np.mean(iw_mul_all)) if iw_mul_all else None,
        "mean_cv_additive": float(np.mean([d["cv_additive"] for d in onestep.values()]))
        if onestep
        else None,
        "mean_cv_multiplicative": float(
            np.mean([d["cv_multiplicative"] for d in onestep.values()])
        )
        if onestep
        else None,
    }
    _plot_add_vs_mul_panel(
        onestep,
        "Daily one-step — additive vs multiplicative Level-2 combine",
        stem_os,
        holiday_marks=use_hol_marks,
    )
    (OUT / f"{stem_os}.json").write_text(json.dumps(dump_onestep, indent=2))
    print(
        "one-step IWMAE "
        f"add={dump_onestep['summary']['mean_iwmae_additive']} "
        f"mul={dump_onestep['summary']['mean_iwmae_multiplicative']} | "
        f"CV add={dump_onestep['summary']['mean_cv_additive']} "
        f"mul={dump_onestep['summary']['mean_cv_multiplicative']}"
    )

    # Short recursive path from a shared origin (same helpers as daily recursive plot).
    panel = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )
    hol = pd.concat([h_tr, h_va, h_te], ignore_index=True)
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
    print(f"daily combine-compare recursive origins: {len(origins)} H={H}")

    rolls = {}
    for mode, model in models.items():

        def _predict(X, sku, _model=model):
            parts = split_components(X, cfg)
            pred = _model.predict([*parts, sku], batch_size=512, verbose=0)
            return (
                np.asarray(pred["final_forecast"]).reshape(-1),
                np.asarray(pred["non_zero_probability"]).reshape(-1),
            )

        rolls[mode] = rollout_tabular(
            timelines,
            origins,
            sku_map,
            _predict,
            cfg.lag_periods,
            tmin_raw,
            span_raw,
            H,
        )

    h_short, h_long = 7, min(28, H)
    horiz = {}
    dump_h = {
        "protocol": "recursive_additive_vs_multiplicative",
        "seed": args.seed,
        "horizon": H,
        "skus": {},
    }
    for i, (sku, t_idx_o) in enumerate(origins):
        d = {
            "y": rolls["additive"]["y_true"][i].astype(np.float64).tolist(),
            "additive": rolls["additive"]["yhat"][i].astype(np.float64).tolist(),
            "multiplicative": rolls["multiplicative"]["yhat"][i].astype(np.float64).tolist(),
            "origin_idx": int(t_idx_o),
            "origin_date": str(pd.Timestamp(timelines[sku].dates[t_idx_o]))[:10],
            "cv_additive": _series_cv(rolls["additive"]["yhat"][i]),
            "cv_multiplicative": _series_cv(rolls["multiplicative"]["yhat"][i]),
        }
        horiz[sku] = d
        dump_h["skus"][sku] = d

    if horiz:
        plot_dict = {
            sku: {
                "y": d["y"],
                "ds": d["additive"],
                "baseline": d["multiplicative"],
            }
            for sku, d in horiz.items()
        }
        _plot_horizon_panel(
            plot_dict,
            "Daily recursive — additive vs multiplicative Level-2 combine",
            stem_rec,
            "Multiplicative",
            h_short,
            h_long,
        )
        (OUT / f"{stem_rec}.json").write_text(json.dumps(dump_h, indent=2))
    print("daily combine-compare done")


def _normalize_monthly_encoding(encoding: str) -> str:
    enc = str(encoding).lower().replace("+", "_").replace("-", "_")
    if enc in ("months_from_and_month_has", "month_has_and_months_from"):
        return "months_from_month_has"
    return enc


def _monthly_holiday_vector_for_date(date, cfg, sku_id):
    """Return ordered holiday feature vector for one month, or None if disabled."""
    if not cfg.holiday_names:
        return None
    meta = cfg.config.get("metadata", {}) or {}
    encoding = _normalize_monthly_encoding(meta.get("holiday_encoding", "none"))
    if encoding not in ("month_has", "months_from", "months_from_month_has"):
        return None
    mf_keys = [
        n.replace("months_from_", "", 1)
        for n in cfg.holiday_names
        if n.startswith("months_from_")
    ]
    mh_keys = [
        n.replace("month_has_", "", 1)
        for n in cfg.holiday_names
        if n.startswith("month_has_")
    ]
    keys = list(dict.fromkeys(mf_keys + mh_keys))
    default_country = str(
        meta.get("holiday_country_default", meta.get("holiday_country", "US"))
    )
    country = country_from_sku_id(sku_id, default=default_country)
    dates = pd.Series([pd.Timestamp(date)])
    distance_scope = str(
        meta.get("holiday_distance_scope", meta.get("distance_scope", "year"))
    )
    parts = []
    if encoding in ("months_from", "months_from_month_has"):
        parts.append(
            months_from_holiday_features(
                dates,
                holiday_keys=mf_keys or keys,
                country=country,
                distance_scope=distance_scope,
            )
        )
    if encoding in ("month_has", "months_from_month_has"):
        parts.append(
            month_has_holiday_features(
                dates, holiday_keys=mh_keys or keys, country=country
            )
        )
    built = (
        pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
        if len(parts) > 1
        else parts[0]
    )
    return built[cfg.holiday_names].to_numpy(np.float32).reshape(-1)


def _month_has_mark_dates(dates, sku_id, cfg):
    """Calendar months (YYYY-MM-DD) that contain any local holiday."""
    if not cfg.holiday_names:
        return []
    meta = cfg.config.get("metadata", {}) or {}
    encoding = _normalize_monthly_encoding(meta.get("holiday_encoding", "none"))
    if encoding not in ("month_has", "months_from_month_has"):
        return []
    keys = [
        n.replace("month_has_", "", 1)
        for n in cfg.holiday_names
        if n.startswith("month_has_")
    ]
    if not keys:
        return []
    default_country = str(
        meta.get("holiday_country_default", meta.get("holiday_country", "US"))
    )
    country = country_from_sku_id(sku_id, default=default_country)
    ds = pd.to_datetime(pd.Series(dates))
    built = month_has_holiday_features(ds, holiday_keys=keys, country=country)
    any_hol = built.to_numpy(np.float32).max(axis=1) > 0.5
    return [str(pd.Timestamp(d).normalize().date()) for d, flag in zip(ds, any_hol) if flag]


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

    cfg_path = Path(args.feature_config_monthly or (ROOT / "feature_config_monthly.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_feature_config(str(cfg_path))
    fig_prefix = str(args.fig_prefix_carparts or "fig_forecast_carparts")
    meta = cfg.config.get("metadata", {}) or {}
    use_country = _country_calendar_enabled(cfg)
    default_country = str(
        meta.get("holiday_country_default", meta.get("holiday_country", "US"))
    )
    print(
        f"feature_config={cfg_path.name} n_feat={cfg.total_features} "
        f"n_holiday={len(cfg.holiday_indices)} holiday_encoding={meta.get('holiday_encoding')} "
        f"holiday_calendar={meta.get('holiday_calendar', 'static')} "
        f"default_country={default_country}"
    )
    if use_country and cfg.holiday_names:
        enc_name = _normalize_monthly_encoding(
            meta.get("holiday_encoding", "month_has")
        )
        distance_scope = str(
            meta.get("holiday_distance_scope", meta.get("distance_scope", "year"))
        )
        print(
            f"rebuilding monthly holidays ({enc_name}, distance_scope={distance_scope}) "
            f"from country calendars (sku prefix / default={default_country})"
        )
        mf_keys = [
            n.replace("months_from_", "", 1)
            for n in cfg.holiday_names
            if n.startswith("months_from_")
        ]
        mh_keys = [
            n.replace("month_has_", "", 1)
            for n in cfg.holiday_names
            if n.startswith("month_has_")
        ]
        keys = list(dict.fromkeys(mf_keys + mh_keys))
        country_col = meta.get("holiday_country_column")
        rebuild_kw = dict(
            holiday_keys=keys,
            encoding=enc_name,
            sku_col="id_var",
            date_col="ds",
            default_country=default_country,
            distance_scope=distance_scope,
        )
        h_tr = build_country_month_holiday_features(
            train_df,
            country_col=country_col if country_col in train_df.columns else None,
            **rebuild_kw,
        )
        h_va = build_country_month_holiday_features(
            val_df,
            country_col=country_col if country_col in val_df.columns else None,
            **rebuild_kw,
        )
        h_te = build_country_month_holiday_features(
            test_df,
            country_col=country_col if country_col in test_df.columns else None,
            **rebuild_kw,
        )

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
        component_combine=args.component_combine,
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

    use_hol_marks = bool(cfg.holiday_names) or bool(args.holiday_markers)
    enc_name = _normalize_monthly_encoding(meta.get("holiday_encoding", "none"))
    distance_scope = str(
        meta.get("holiday_distance_scope", meta.get("distance_scope", "year"))
    )
    onestep = {}
    dump_onestep = {
        "protocol": "one_step_test",
        "seed": args.seed,
        "epochs": args.epochs,
        "feature_config": str(cfg_path.name),
        "holiday_encoding": meta.get("holiday_encoding"),
        "holiday_distance_scope": distance_scope,
        "holiday_calendar": meta.get("holiday_calendar", "static"),
        "holiday_country_default": default_country,
        "skus": {},
        "metrics": {},
    }
    hol_flag_all = np.zeros(len(test_df), dtype=np.float64)
    if cfg.holiday_names and enc_name in ("month_has", "months_from_month_has"):
        hol_cols = [c for c in Xte_df.columns if c.startswith("month_has_")]
        if hol_cols:
            hol_flag_all = Xte_df[hol_cols].to_numpy(np.float64).max(axis=1)

    for sku in plot_skus:
        m = test_df["id_var"].astype(str).to_numpy() == sku
        if not m.any():
            continue
        dates = [str(x)[:10] for x in pd.to_datetime(test_df.loc[m, "ds"])]
        y_sku = test_df.loc[m, "Quantity"].to_numpy(np.float64)
        ds_sku = yhat_ds[m].astype(np.float64)
        base_sku = yhat_tsb[m].astype(np.float64)
        hol_dates = _month_has_mark_dates(dates, sku, cfg) if use_hol_marks else []
        hol_flag = hol_flag_all[m]
        corr = float("nan")
        if hol_flag.std() > 1e-12 and ds_sku.std() > 1e-12:
            corr = float(np.corrcoef(ds_sku, hol_flag)[0, 1])
        d = {
            "dates": dates,
            "y": y_sku.tolist(),
            "ds": ds_sku.tolist(),
            "baseline": base_sku.tolist(),
            "holiday_dates": hol_dates,
            "iwmae_ds": _iwmae(y_sku, ds_sku),
            "iwmae_baseline": _iwmae(y_sku, base_sku),
            "corr_yhat_month_has_any": corr,
        }
        onestep[sku] = d
        dump_onestep["skus"][sku] = d
        dump_onestep["metrics"][sku] = {
            "iwmae_ds": d["iwmae_ds"],
            "iwmae_baseline": d["iwmae_baseline"],
            "corr_yhat_month_has_any": corr,
        }

    title_os = "Car Parts (monthly) — one-step forecasts (test window)"
    if use_country and cfg.holiday_names:
        title_os = (
            f"Car Parts monthly — country {enc_name} "
            f"(scope={distance_scope}; default {default_country}; no country on T#### ids)"
        )
    elif cfg.holiday_names:
        title_os = f"Car Parts monthly — one-step with {enc_name} holidays"
    _plot_onestep_panel(
        onestep,
        title_os,
        f"{fig_prefix}_onestep",
        "TSB",
        holiday_marks=use_hol_marks,
    )
    (OUT / f"{fig_prefix}_onestep.json").write_text(json.dumps(dump_onestep, indent=2))

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
        "feature_config": str(cfg_path.name),
        "holiday_encoding": meta.get("holiday_encoding"),
        "holiday_calendar": meta.get("holiday_calendar", "static"),
        "holiday_country_default": default_country,
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
        yhat_ds_h = np.zeros(H, np.float64)
        yhat_tsb_h = np.zeros(H, np.float64)
        y_hist_tsb = list(ys[: origin_i + 1].astype(float))
        for h in range(H):
            date = pd.Timestamp(dates[first_test + h])
            hol_vec = _monthly_holiday_vector_for_date(date, cfg, sku)
            feat = assemble_monthly_row(
                date, st, lag_periods, tmin_raw, span_raw, holiday_values=hol_vec
            )
            Xrow = feat.reshape(1, -1).astype(np.float32)
            parts = split_components(Xrow, cfg)
            sk = np.array([[sku_map[sku]]], np.int32)
            pred = ds_model.predict([*parts, sk], verbose=0)
            yh = float(np.asarray(pred["final_forecast"]).reshape(-1)[0])
            yhat_ds_h[h] = max(yh, 0.0)
            st.update(date, yhat_ds_h[h])
            tsb = croston_variants(np.asarray(y_hist_tsb, float))["tsb"]
            yhat_tsb_h[h] = tsb
            y_hist_tsb.append(tsb)
        d = {
            "y": y_true.tolist(),
            "ds": yhat_ds_h.tolist(),
            "baseline": yhat_tsb_h.tolist(),
            "origin_date": str(dates[origin_i])[:10],
            "test_start": str(dates[first_test])[:10],
            "iwmae_ds": _iwmae(y_true, yhat_ds_h),
            "iwmae_baseline": _iwmae(y_true, yhat_tsb_h),
        }
        horiz[sku] = d
        dump_h["skus"][sku] = d

    if horiz:
        title_rec = "Car Parts recursive forecasts from pre-test origin (DS vs TSB)"
        if use_country and cfg.holiday_names:
            title_rec = (
                f"Car Parts recursive — country {enc_name} "
                f"(scope={distance_scope}; default {default_country})"
            )
        elif cfg.holiday_names:
            title_rec = f"Car Parts recursive — {enc_name} holidays"
        _plot_horizon_panel(
            horiz,
            title_rec,
            f"{fig_prefix}_recursive",
            "TSB",
            2,
            H,
        )
        (OUT / f"{fig_prefix}_recursive.json").write_text(json.dumps(dump_h, indent=2))
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
    p.add_argument("--only", choices=("daily", "carparts", "both", "daily_combine"), default="both")
    p.add_argument(
        "--feature_config_daily",
        default=None,
        help="Override daily feature YAML (e.g. feature_config_daily_country_holiday.yaml).",
    )
    p.add_argument(
        "--fig_prefix_daily",
        default="fig_forecast_daily",
        help="Stem prefix for daily figs (e.g. fig_forecast_daily_binary_hol).",
    )
    p.add_argument(
        "--feature_config_monthly",
        default=None,
        help=(
            "Override monthly feature YAML "
            "(e.g. feature_config_monthly_country_holiday.yaml). "
            "Default locked bake-off: feature_config_monthly.yaml."
        ),
    )
    p.add_argument(
        "--fig_prefix_carparts",
        default="fig_forecast_carparts",
        help="Stem prefix for carparts figs (e.g. fig_forecast_carparts_country_hol).",
    )
    p.add_argument(
        "--holiday_markers",
        type=int,
        default=0,
        help="Force holiday axvlines on one-step plots (1/0). Auto-on when binaries/month_has present.",
    )
    p.add_argument(
        "--component_combine",
        choices=("additive", "multiplicative"),
        default="additive",
        help="Level-2 expert combine for --only daily/carparts (default additive).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    if args.only == "daily_combine":
        run_daily_combine_compare(args)
    else:
        if args.only in ("daily", "both"):
            run_daily(args)
        if args.only in ("carparts", "both"):
            run_carparts(args)
    print(f"all done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
