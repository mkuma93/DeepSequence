#!/usr/bin/env python3
"""Weekly-grain multi-horizon bake-off (ISO Monday panel).

Direct-MH DeepSequence + LightGBM + classical TSB (Croston/SBA optional).
Uses ``feature_config_weekly.yaml`` and holiday CSVs when present.

Protocol: one origin per SKU at the first test week; forecast next H weeks.
Report horizons default to 1, 4, 8 (≈ week / month / 2-month).

Example (locked 800)::

  TF_USE_LEGACY_KERAS=1 .venv-test/bin/python examples/eval_weekly_mh.py \\
    --data_dir ab_runs/weekly/panel_locked800 \\
    --feature_config feature_config_weekly.yaml \\
    --sku_list ab_runs/recompare/sku_list_daily_data42.json \\
    --max_skus 800 --horizon 8 --report_horizons 1,4,8 \\
    --models deepsequence,tsb,lightgbm --epochs 15 --seed 42 \\
    --out_json ab_runs/weekly/weekly_mh8_locked800_s42.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from classical_intermittent import croston_variants
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from eval_helpers import (
    add_panel_seed_args,
    class_balance_pos_weight,
    cummae_from_rollout,
    filter_aligned,
    kpi_block,
    resolve_eval_seeds,
    resolve_sku_zero_rates,
    select_eval_skus,
    split_components,
    train_mase_scale,
    train_volume_terciles,
)
from feature_config_loader import load_feature_config
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss

ALL_MODELS = (
    "deepsequence",
    "lightgbm",
    "croston",
    "sba",
    "tsb",
)


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
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument(
        "--report_horizons",
        default="1,4,8",
        help="Comma-separated 1-indexed horizons to report.",
    )
    add_panel_seed_args(p)
    p.add_argument("--mase_season", type=int, default=4)
    p.add_argument("--models", default="deepsequence,tsb,lightgbm")
    p.add_argument(
        "--out_json",
        default=str(ROOT / "ab_runs/weekly/weekly_mh_s42.json"),
    )
    p.add_argument("--output_activation", default=None)
    p.add_argument("--trend_monotonic", type=int, default=None)
    p.add_argument("--holiday_monotonic", type=int, default=None)
    p.add_argument("--regressor_monotonic", type=int, default=None)
    p.add_argument("--context_aware_component_mixer", type=int, default=None)
    p.add_argument("--context_film_seasonal_holiday", type=int, default=None)
    p.add_argument("--use_cross_layers", type=int, default=None)
    p.add_argument(
        "--no_sku",
        action="store_true",
        help="Disable SKU embedding (Car Parts-style shared trunk).",
    )
    return p.parse_args()


def _ds_builder_kwargs(args) -> dict:
    sig = inspect.signature(build_hierarchical_model_lightweight)
    keys = (
        "output_activation",
        "trend_monotonic",
        "holiday_monotonic",
        "regressor_monotonic",
        "context_aware_component_mixer",
        "context_film_seasonal_holiday",
        "use_cross_layers",
    )
    defaults = {k: sig.parameters[k].default for k in keys}
    out = dict(defaults)
    if args.output_activation is not None:
        out["output_activation"] = args.output_activation
    for flag in keys:
        if flag == "output_activation":
            continue
        val = getattr(args, flag)
        if val is not None:
            out[flag] = bool(val)
    return out


def _parse_report_horizons(raw: str, horizon: int) -> list[int]:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        h = int(part)
        if h < 1 or h > int(horizon):
            raise SystemExit(f"report horizon {h} outside 1..{horizon}")
        out.append(h)
    if not out:
        raise SystemExit("--report_horizons empty")
    return out


def _load_holiday(data_dir: Path, split: str, n_rows: int) -> pd.DataFrame:
    path = data_dir / f"holiday_features_{split}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(index=range(n_rows))


def build_mh_xy(X, y, skus, horizon: int):
    H = int(horizon)
    xs, ys, ss = [], [], []
    skus = np.asarray(skus)
    y = np.asarray(y, np.float32)
    for sku in np.unique(skus):
        idx = np.where(skus == sku)[0]
        if len(idx) < H:
            continue
        for i in range(len(idx) - H + 1):
            sl = idx[i : i + H]
            xs.append(X[sl[0]])
            ys.append(y[sl])
            ss.append(sku)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(ss)


def mh_metrics(y_true, yhat, p, report_horizons=None, mase_scale=None):
    y_true = np.asarray(y_true, np.float32)
    yhat = np.maximum(np.asarray(yhat, np.float32), 0.0)
    p = None if p is None else np.asarray(p, np.float32)
    H = y_true.shape[1]
    if report_horizons is None:
        report_horizons = list(range(1, H + 1))
    out = {
        "mean_1_to_H": kpi_block(
            y_true.reshape(-1),
            yhat.reshape(-1),
            None if p is None else p.reshape(-1),
            mase_scale=mase_scale,
        ),
        "by_horizon": {},
        "by_horizon_cum": {},
        "flatness": {},
    }
    for h in report_horizons:
        if 1 <= h <= H:
            col = h - 1
            out["by_horizon"][str(h)] = kpi_block(
                y_true[:, col],
                yhat[:, col],
                None if p is None else p[:, col],
                mase_scale=mase_scale,
            )
    cum = cummae_from_rollout(
        y_true, yhat, p, report_horizons=report_horizons, mase_scale=mase_scale
    )
    out["by_horizon_cum"] = cum["by_horizon"]

    # Flat mean-rate diagnostic (planning-rate vs spike-tracking)
    flat = {}
    for h in report_horizons:
        if not (1 <= h <= H):
            continue
        col = h - 1
        yt = y_true[:, col].astype(np.float64)
        yh = yhat[:, col].astype(np.float64)
        mean_y = float(yt.mean()) if len(yt) else 0.0
        mean_h = float(yh.mean()) if len(yh) else 0.0
        std_h = float(yh.std()) if len(yh) else 0.0
        # fraction of forecasts within 10% of panel mean forecast
        near = float(np.mean(np.abs(yh - mean_h) <= 0.10 * max(mean_h, 1e-6))) if len(yh) else None
        corr = None
        if len(yt) > 2 and yt.std() > 0 and std_h > 0:
            corr = float(np.corrcoef(yt, yh)[0, 1])
        flat[str(h)] = {
            "mean_actual": mean_y,
            "mean_yhat": mean_h,
            "std_yhat": std_h,
            "cv_yhat": std_h / max(mean_h, 1e-6),
            "frac_near_mean_yhat": near,
            "corr_y_yhat": corr,
            "n_unique_rounded": int(len(np.unique(np.round(yh)))),
        }
    out["flatness"] = flat
    return out


def classical_recursive(histories, horizon, alpha=0.1):
    H = int(horizon)
    n = len(histories)
    names = ("croston", "sba", "tsb")
    out = {k: np.zeros((n, H), np.float32) for k in names}
    for i, y0 in enumerate(histories):
        for k in names:
            y = list(np.asarray(y0, np.float64))
            for h in range(H):
                preds = croston_variants(np.asarray(y), alpha=alpha)
                out[k][i, h] = preds[k]
                y.append(preds[k])
    return out


def train_gated(
    base,
    tr,
    va,
    y_tr,
    y_va,
    sku_tr,
    sku_va,
    zero_rate,
    avg_nz,
    pos_weight,
    epochs,
    batch_size,
    horizon_decay=1.0,
):
    _ = base(
        [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
        training=False,
    )
    model = AdaptiveWeightedModel(
        base_model=base,
        bce_loss_fn=WeightedBCELoss(weight_nonzero=pos_weight, weight_zero=1.0),
        mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        zero_rate=zero_rate,
        avg_nonzero_demand=avg_nz,
        pos_weight=pos_weight,
        loss_recipe="three_term",
        alpha_bce=0.2,
        w_gated=1.0,
        w_mag=1.0,
        use_fixed_weights=True,
        horizon_decay=horizon_decay,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
    model.fit(
        [*tr, sku_tr],
        {"final_forecast": y_tr, "base_forecast": y_tr},
        validation_data=([*va, sku_va], {"final_forecast": y_va, "base_forecast": y_va}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            )
        ],
        verbose=2,
    )
    return model


def main():
    args = parse_args()
    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    unknown = selected - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"Unknown models: {sorted(unknown)}")
    report_horizons = _parse_report_horizons(args.report_horizons, args.horizon)

    data_seed, train_seed = resolve_eval_seeds(
        args.seed, args.data_seed, args.train_seed
    )
    tf.keras.utils.set_random_seed(train_seed)
    H = int(args.horizon)
    data_dir = Path(args.data_dir)

    print("Loading weekly panel...")
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    h_tr = _load_holiday(data_dir, "train", len(train_df))
    h_va = _load_holiday(data_dir, "val", len(val_df))
    h_te = _load_holiday(data_dir, "test", len(test_df))

    # Universe = all splits: weekly prepare can leave a few locked SKUs with
    # test-only history (no rows before the train cut).
    universe = (
        pd.concat([train_df["id_var"], val_df["id_var"], test_df["id_var"]], ignore_index=True)
        .astype(str)
        .unique()
    )
    chosen_list = select_eval_skus(
        universe,
        max_skus=args.max_skus,
        data_seed=data_seed,
        sku_list_path=args.sku_list,
        save_sku_list_path=args.save_sku_list,
    )
    chosen = set(chosen_list)
    print(
        f"Panel lock: data_seed={data_seed} train_seed={train_seed} "
        f"n_skus={len(chosen)}"
        + (f" sku_list={args.sku_list}" if args.sku_list else "")
    )
    train_df, h_tr = filter_aligned(train_df, h_tr, chosen)
    val_df, h_va = filter_aligned(val_df, h_va, chosen)
    test_df, h_te = filter_aligned(test_df, h_te, chosen)
    for df in (train_df, val_df, test_df):
        df.sort_values(["id_var", "ds"], kind="mergesort", inplace=True)
        df.reset_index(drop=True, inplace=True)

    volume_map, volume_stats = train_volume_terciles(train_df)
    mase_scale = train_mase_scale(train_df, season=args.mase_season)
    # Include locked SKUs that appear only in val/test (no train weeks).
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
    use_sku = not bool(args.no_sku)

    def enc(df):
        return (
            df["id_var"].astype(str).map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)
        )

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val = enc(train_df), enc(val_df)
    zero_rate = float((y_train == 0).mean())
    avg_nz = float(y_train[y_train > 0].mean()) if (y_train > 0).any() else 1.0
    pos_weight = class_balance_pos_weight(y_train)
    print(
        f"n_skus={n_skus} zero_rate={zero_rate:.3f} H={H} "
        f"pos_weight={pos_weight:.3f} use_sku={use_sku}"
    )

    cfg = load_feature_config(args.feature_config)
    print(f"Features v{cfg.config['metadata'].get('version')} n={cfg.total_features}")
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

    sk_te = test_df["id_var"].astype(str).to_numpy()
    X_origin, y_true_mh, sk_origin = build_mh_xy(X_test, y_test, sk_te, H)
    keep, seen = [], set()
    for i, s in enumerate(sk_origin):
        if s in seen:
            continue
        seen.add(s)
        keep.append(i)
    keep = np.asarray(keep, np.int64)
    X_origin, y_true_mh, sk_origin = X_origin[keep], y_true_mh[keep], sk_origin[keep]
    sku_origin = np.array([sku_map[str(s)] for s in sk_origin], np.int32).reshape(-1, 1)
    print(f"origins={len(sk_origin)} (one per series at test start)")

    hist_y = []
    for sku in sk_origin:
        tr = train_df[train_df["id_var"].astype(str) == sku]
        va = val_df[val_df["id_var"].astype(str) == sku]
        hist_y.append(
            np.concatenate(
                [tr["Quantity"].to_numpy(np.float64), va["Quantity"].to_numpy(np.float64)]
            )
        )

    ds_stack = _ds_builder_kwargs(args)
    results = {
        "config": {
            "dataset": "weekly_aggregate_locked",
            "protocol": "fixed origin = first test week; forecast H weeks (direct MH)",
            "week_rule": "ISO Monday-start",
            "horizon": H,
            "report_horizons": report_horizons,
            "n_skus": n_skus,
            "seed": args.seed,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "sku_list": args.sku_list,
            "zero_rate_train": zero_rate,
            "feature_config": str(args.feature_config),
            "feature_version": cfg.config["metadata"].get("version"),
            "volume_stats": volume_stats,
            "models": sorted(selected),
            "use_sku_deepsequence": use_sku,
            "ds_stack": ds_stack,
            "mase_season": args.mase_season,
        },
        "models": {},
    }

    classical = selected & {"croston", "sba", "tsb"}
    if classical:
        print("\n=== Classical recursive MH ===")
        t0 = time.time()
        preds = classical_recursive(hist_y, H)
        dt = time.time() - t0
        for name in ("croston", "sba", "tsb"):
            if name not in selected:
                continue
            yhat = preds[name]
            p = np.clip(1.0 - np.exp(-yhat), 0, 1)
            metrics = mh_metrics(
                y_true_mh, yhat, p, report_horizons=report_horizons, mase_scale=mase_scale
            )
            results["models"][name] = {
                "method": "recursive",
                "train_seconds": dt / max(len(classical), 1),
                **metrics,
            }

    if "deepsequence" in selected:
        print("\n=== DeepSequence direct MH ===")
        sk_tr = train_df["id_var"].astype(str).to_numpy()
        sk_va = val_df["id_var"].astype(str).to_numpy()
        Xtr_mh, ytr_mh, sktr_mh = build_mh_xy(X_train, y_train, sk_tr, H)
        Xva_mh, yva_mh, skva_mh = build_mh_xy(X_val, y_val, sk_va, H)
        if len(Xva_mh) == 0:
            Xva_mh, yva_mh, skva_mh = Xtr_mh[-n_skus:], ytr_mh[-n_skus:], sktr_mh[-n_skus:]
        sktr = np.array([sku_map[str(s)] for s in sktr_mh], np.int32).reshape(-1, 1)
        skva = np.array([sku_map[str(s)] for s in skva_mh], np.int32).reshape(-1, 1)
        tr = split_components(Xtr_mh, cfg)
        va = split_components(Xva_mh, cfg)
        te = split_components(X_origin, cfg)
        print(f"MH windows train/val={len(ytr_mh)}/{len(yva_mh)}")
        _, sku_rates = resolve_sku_zero_rates(y_train, sku_train, n_skus=n_skus)
        _ = sku_rates  # rates used implicitly via panel zero_rate / BCE
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
            use_sku=use_sku,
            horizon=H,
            **ds_stack,
        )
        t0 = time.time()
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
            args.epochs,
            args.batch_size,
            horizon_decay=0.95,
        )
        pred = model.predict([*te, sku_origin], batch_size=2048, verbose=0)
        yhat = np.asarray(pred["final_forecast"], np.float32)
        p = np.asarray(pred["non_zero_probability"], np.float32)
        if yhat.ndim == 1:
            yhat = yhat.reshape(-1, H)
        if p.ndim == 1:
            p = p.reshape(-1, H)
        metrics = mh_metrics(
            y_true_mh, yhat, p, report_horizons=report_horizons, mase_scale=mase_scale
        )
        results["models"]["deepsequence"] = {
            "method": "direct_mh",
            "use_sku": use_sku,
            "ds_stack": ds_stack,
            "train_seconds": time.time() - t0,
            **metrics,
        }

    if "lightgbm" in selected:
        print("\n=== LightGBM multi-output MH ===")
        import lightgbm as lgb

        sk_tr = train_df["id_var"].astype(str).to_numpy()
        Xtr_mh, ytr_mh, sktr_mh = build_mh_xy(X_train, y_train, sk_tr, H)
        sktr = np.array([sku_map[str(s)] for s in sktr_mh], np.float32).reshape(-1, 1)
        Xtr = np.concatenate([Xtr_mh, sktr], axis=1)
        Xte = np.concatenate([X_origin, sku_origin.astype(np.float32)], axis=1)
        t0 = time.time()
        model = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=train_seed,
                n_jobs=-1,
                verbosity=-1,
            )
        )
        model.fit(Xtr, ytr_mh)
        yhat = np.maximum(model.predict(Xte), 0.0).astype(np.float32)
        p = np.clip(1.0 - np.exp(-yhat), 0, 1)
        metrics = mh_metrics(
            y_true_mh, yhat, p, report_horizons=report_horizons, mase_scale=mase_scale
        )
        results["models"]["lightgbm"] = {
            "method": "multi_output",
            "train_seconds": time.time() - t0,
            **metrics,
        }

    comparison = []
    comparison_cum = []
    for model, payload in results["models"].items():
        o = payload["mean_1_to_H"]
        by_h = payload.get("by_horizon", {})
        by_c = payload.get("by_horizon_cum", {})
        row = {
            "model": model,
            "method": payload.get("method"),
            "iwmae_rounded": o.get("iwmae_rounded"),
            "mae_rounded": o.get("mae_all_rounded"),
            "mae_nonzero": o.get("mae_nonzero"),
            "occ_f1": o.get("occ_f1"),
            "bias": o.get("bias"),
            "mean_final": o.get("mean_final"),
            "std_yhat_h1": payload.get("flatness", {}).get("1", {}).get("std_yhat"),
            "cv_yhat_h1": payload.get("flatness", {}).get("1", {}).get("cv_yhat"),
            "corr_h1": payload.get("flatness", {}).get("1", {}).get("corr_y_yhat"),
        }
        for h in report_horizons:
            row[f"h{h}_iwmae"] = by_h.get(str(h), {}).get("iwmae_rounded")
            row[f"h{h}_cummae"] = by_c.get(str(h), {}).get("cummae_rounded")
            row[f"h{h}_cum_iwmae"] = by_c.get(str(h), {}).get("cum_iwmae_rounded")
        comparison.append(row)
        cum_row = {"model": model}
        for h in report_horizons:
            cum_row[f"h{h}_cummae"] = by_c.get(str(h), {}).get("cummae_rounded")
            cum_row[f"h{h}_cum_iwmae"] = by_c.get(str(h), {}).get("cum_iwmae_rounded")
        comparison_cum.append(cum_row)

    comparison = sorted(
        comparison,
        key=lambda r: (
            r["iwmae_rounded"] is None,
            r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
        ),
    )
    h_long = report_horizons[-1]
    comparison_cum = sorted(
        comparison_cum,
        key=lambda r: (
            r.get(f"h{h_long}_cummae") is None,
            r.get(f"h{h_long}_cummae") if r.get(f"h{h_long}_cummae") is not None else 1e9,
        ),
    )
    results["comparison"] = comparison
    results["comparison_cum"] = comparison_cum
    results["mase_scale"] = mase_scale

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("\n" + "=" * 72)
    print(f"WEEKLY MH BAKE-OFF  H={H}  (primary: mean_1_to_H iwmae_rounded)")
    print("=" * 72)
    for row in comparison:
        hs = " ".join(
            f"h{h}={row.get(f'h{h}_iwmae'):.3f}" for h in report_horizons if row.get(f"h{h}_iwmae") is not None
        )
        print(
            f"  {row['model']:22s} {row['method']:16s} "
            f"mean={row['iwmae_rounded']:.3f} {hs} "
            f"bias={row['bias']:+.3f} cv_h1={row.get('cv_yhat_h1')}"
        )
    print("\nCumMAE:")
    for row in comparison_cum:
        hs = " ".join(
            f"h{h}={row.get(f'h{h}_cummae'):.3f}"
            for h in report_horizons
            if row.get(f"h{h}_cummae") is not None
        )
        print(f"  {row['model']:22s} {hs}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
