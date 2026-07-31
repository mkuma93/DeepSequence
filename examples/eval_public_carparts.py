#!/usr/bin/env python3
"""
Public intermittent bake-off on Monash Car Parts (monthly).

Models: DeepSequence, LightGBM, DeepAR-lite, TST, TFT-lite,
        Croston, SBA, TSB.

Primary metric: IWMAE (same suite as proprietary v1.6 bake-off).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from classical_intermittent import predict_classical_on_panel
from eval_helpers import (
    apply_iwmae_gate,
    build_deepar,
    build_tft,
    build_transformer,
    calibrate_iwmae_gate,
    class_balance_pos_weight,
    add_panel_seed_args,
    filter_aligned,
    kpi_block,
    resolve_eval_seeds,
    select_eval_skus,
    split_components,
    strata_report,
    train_mase_scale,
    train_volume_terciles,
)

ALL_MODELS = (
    "deepsequence",
    "lightgbm",
    "deepar_lite",
    "temporal_transformer",
    "tft_lite",
    "croston",
    "sba",
    "tsb",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default=str(ROOT / "public_data/car_parts/panel"),
    )
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=12)
    add_panel_seed_args(p)
    p.add_argument("--mase_season", type=int, default=12)
    p.add_argument("--models", default=",".join(ALL_MODELS))
    p.add_argument(
        "--feature_config",
        default=str(ROOT / "feature_config_monthly.yaml"),
        help="Feature YAML (monthly contract by default for Car Parts)",
    )
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_public_carparts_v16.json"),
    )
    p.add_argument(
        "--no_sku",
        action="store_true",
        help="Disable SKU embedding / ID personalization (shared trunk only).",
    )
    p.add_argument(
        "--no_calib",
        action="store_true",
        help="Skip validation IWMAE gate calibration.",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon. 1=one-step; H>1=direct multi-horizon head.",
    )
    return p.parse_args()


def build_mh_xy(X, y, skus, horizon: int):
    """Per-series sliding windows: features at t -> targets y[t:t+H]."""
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
    return (
        np.asarray(xs, np.float32),
        np.asarray(ys, np.float32),
        np.asarray(ss),
    )


def mh_horizon_report(y_true, yhat, p, skus, volume_map, mase_scale, report_horizons=None):
    """Per-horizon + pooled IWMAE for direct MH predictions."""
    y_true = np.asarray(y_true, np.float32)
    yhat = np.asarray(yhat, np.float32)
    p = None if p is None else np.asarray(p, np.float32)
    H = y_true.shape[1]
    if report_horizons is None:
        report_horizons = list(range(1, H + 1))
    out = {
        "mean_1_to_H": {
            "overall": kpi_block(
                y_true.reshape(-1),
                yhat.reshape(-1),
                None if p is None else p.reshape(-1),
                mase_scale=mase_scale,
            )
        },
        "by_horizon": {},
    }
    for h in report_horizons:
        if h < 1 or h > H:
            continue
        col = h - 1
        block = {
            "overall": kpi_block(
                y_true[:, col],
                yhat[:, col],
                None if p is None else p[:, col],
                mase_scale=mase_scale,
            ),
            "strata": strata_report(
                y_true[:, col],
                yhat[:, col],
                None if p is None else p[:, col],
                skus,
                volume_map,
                mase_scale=mase_scale,
            ),
        }
        out["by_horizon"][str(h)] = block
    return out


def main():
    args = parse_args()
    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    unknown = selected - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"Unknown models: {sorted(unknown)}")

    data_seed, train_seed = resolve_eval_seeds(
        args.seed, args.data_seed, args.train_seed
    )
    tf.keras.utils.set_random_seed(train_seed)
    data_dir = Path(args.data_dir)
    if not (data_dir / "train_split.csv").exists():
        raise SystemExit(
            f"Missing panel in {data_dir}. Run:\n"
            "  python examples/public_data/prepare_carparts.py"
        )

    print("Loading Car Parts panel...")
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    # Monthly contract disables holidays; keep empty frames for filter_aligned length checks
    h_tr = pd.DataFrame(index=range(len(train_df)))
    h_va = pd.DataFrame(index=range(len(val_df)))
    h_te = pd.DataFrame(index=range(len(test_df)))

    chosen_list = select_eval_skus(
        train_df["id_var"].unique(),
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
    train_df = train_df.sort_values(["id_var", "ds"], kind="mergesort").reset_index(drop=True)
    val_df = val_df.sort_values(["id_var", "ds"], kind="mergesort").reset_index(drop=True)
    test_df = test_df.sort_values(["id_var", "ds"], kind="mergesort").reset_index(drop=True)
    h_tr = h_tr.iloc[: len(train_df)].reset_index(drop=True)
    h_va = h_va.iloc[: len(val_df)].reset_index(drop=True)
    h_te = h_te.iloc[: len(test_df)].reset_index(drop=True)

    volume_map, volume_stats = train_volume_terciles(train_df)
    mase_scale = train_mase_scale(train_df, season=args.mase_season)
    print(f"n_skus={len(chosen)} zero_rate={(train_df.Quantity==0).mean():.3f} mase_scale={mase_scale}")

    cats = pd.Categorical(train_df["id_var"].astype(str))
    sku_map = {str(k): i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return (
            df["id_var"].astype(str).map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)
        )

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val, sku_test = enc(train_df), enc(val_df), enc(test_df)
    sku_test_raw = test_df["id_var"].astype(str).to_numpy()
    zero_rate = float((y_train == 0).mean())

    cfg = load_feature_config(args.feature_config)
    print(
        f"Building causal features from {cfg.config_path} "
        f"(v{cfg.config['metadata'].get('version')}, n={cfg.total_features})..."
    )
    Xtr_df, states = cfg.create_features(train_df, None, return_states=True)
    Xva_df, states = cfg.create_features(val_df, None, prior_states=states, return_states=True)
    Xte_df, _ = cfg.create_features(test_df, None, prior_states=states, return_states=True)
    feature_names = list(Xtr_df.columns)
    X_train = Xtr_df.to_numpy(np.float32, copy=True)
    X_val = Xva_df.to_numpy(np.float32, copy=True)
    X_test = Xte_df.to_numpy(np.float32, copy=True)
    t_idx = cfg.trend_indices[0]
    tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    results = {
        "config": {
            "dataset": "Monash Car Parts (Zenodo 4656021)",
            "frequency": "monthly",
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "seed": args.seed,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "sku_list": args.sku_list,
            "lookback": args.lookback,
            "mase_season": args.mase_season,
            "zero_rate": zero_rate,
            "feature_config": str(args.feature_config),
            "feature_version": cfg.config["metadata"].get("version"),
            "n_tabular_features": len(feature_names),
            "feature_names": feature_names,
            "volume_stats": volume_stats,
            "models": sorted(selected),
            "note": (
                "Monthly v1.7: thin calendar (trend+Fourier); lags 1/2/12; "
                "classic intermittent (no rolling rates); no holidays; "
                "data-driven pos_weight; val IWMAE gate calibration."
            ),
            "pos_weight_policy": "class_balance_from_train",
            "use_sku": (not args.no_sku),
            "gate_calibration": (not args.no_calib),
            "horizon": int(args.horizon),
        },
        "models": {},
    }

    tr, va, te = (
        split_components(X_train, cfg),
        split_components(X_val, cfg),
        split_components(X_test, cfg),
    )

    # --- Classical intermittent ---
    classical = selected & {"croston", "sba", "tsb"}
    if classical:
        print("\n=== Croston / SBA / TSB ===")
        t0 = time.time()
        preds = predict_classical_on_panel(train_df, val_df, test_df, alpha=0.1)
        dt = time.time() - t0
        for name in ("croston", "sba", "tsb"):
            if name not in selected:
                continue
            yhat = preds[name]
            p = np.clip(1.0 - np.exp(-yhat), 0, 1)
            results["models"][name] = {
                "train_seconds": dt / max(len(classical), 1),
                "overall": kpi_block(y_test, yhat, p, mase_scale=mase_scale),
                "strata": strata_report(
                    y_test, yhat, p, sku_test_raw, volume_map, mase_scale=mase_scale
                ),
            }

    if "deepsequence" in selected:
        horizon = int(args.horizon)
        print(f"\n=== DeepSequence (horizon={horizon}) ===")
        pos_weight = class_balance_pos_weight(y_train)
        print(f"pos_weight={pos_weight:.4f} (train zeros={zero_rate:.3f})")

        if horizon <= 1:
            Xtr_fit, ytr_fit, sktr_fit = X_train, y_train.reshape(-1, 1), sku_train
            Xva_fit, yva_fit, skva_fit = X_val, y_val.reshape(-1, 1), sku_val
            tr_fit, va_fit = tr, va
        else:
            sk_tr_raw = train_df["id_var"].astype(str).to_numpy()
            sk_va_raw = val_df["id_var"].astype(str).to_numpy()
            # Train MH windows on train; validate on train+val chronology so val
            # origins have enough history when H matches the 6-month val block.
            X_tv = np.concatenate([X_train, X_val], axis=0)
            y_tv = np.concatenate([y_train, y_val], axis=0)
            sk_tv = np.concatenate([sk_tr_raw, sk_va_raw], axis=0)
            # Keep chronological within sku: already sorted splits concatenated
            # rebuild by sku to avoid train/val boundary disorder across skus
            order_idx = []
            for sku in train_df["id_var"].astype(str).unique():
                order_idx.extend(np.where(sk_tv == sku)[0].tolist())
            order_idx = np.asarray(order_idx, dtype=np.int64)
            X_tv, y_tv, sk_tv = X_tv[order_idx], y_tv[order_idx], sk_tv[order_idx]

            Xtr_mh, ytr_mh, sktr_mh = build_mh_xy(X_train, y_train, sk_tr_raw, horizon)
            # Val windows: origins whose full H-span lies in val months only is
            # rare; use last-H windows ending in val from combined timeline.
            Xva_mh, yva_mh, skva_mh = [], [], []
            n_tr_per = train_df.groupby(train_df["id_var"].astype(str)).size()
            for sku in np.unique(sk_tv):
                idx = np.where(sk_tv == sku)[0]
                n_tr = int(n_tr_per.get(sku, 0))
                # origins i where i+H-1 >= n_tr (window touches val) and i+H <= len
                for i in range(max(0, n_tr - horizon + 1), len(idx) - horizon + 1):
                    sl = idx[i : i + horizon]
                    Xva_mh.append(X_tv[sl[0]])
                    yva_mh.append(y_tv[sl])
                    skva_mh.append(sku)
            Xva_mh = np.asarray(Xva_mh, np.float32)
            yva_mh = np.asarray(yva_mh, np.float32)
            skva_mh = np.asarray(skva_mh)
            if len(Xva_mh) == 0:
                # fallback: within-val windows
                Xva_mh, yva_mh, skva_mh = build_mh_xy(X_val, y_val, sk_va_raw, horizon)

            sktr_fit = np.array([sku_map[str(s)] for s in sktr_mh], np.int32).reshape(-1, 1)
            skva_fit = np.array([sku_map[str(s)] for s in skva_mh], np.int32).reshape(-1, 1)
            Xtr_fit, ytr_fit = Xtr_mh, ytr_mh
            Xva_fit, yva_fit = Xva_mh, yva_mh
            tr_fit = split_components(Xtr_fit, cfg)
            va_fit = split_components(Xva_fit, cfg)
            print(f"MH windows train/val={len(ytr_fit)}/{len(yva_fit)} H={horizon}")

        base = build_hierarchical_model_lightweight(
            n_temporal_features=len(cfg.trend_indices),
            n_fourier_features=len(cfg.seasonal_indices),
            n_holiday_features=max(len(cfg.holiday_indices), 0),
            n_lag_features=len(cfg.regressor_indices),
            n_skus=n_skus,
            hidden_dim=48,
            sku_embedding_dim=4,
            dropout_rate=0.23,
            use_cross_layers=True,
            use_intermittent=True,
            n_changepoints=15,
            use_sku=(not args.no_sku),
            horizon=horizon,
        )
        _ = base(
            [
                *(np.zeros((1, x.shape[1]), np.float32) for x in tr_fit),
                np.zeros((1, 1), np.int32),
            ],
            training=False,
        )
        ds_model = AdaptiveWeightedModel(
            base_model=base,
            bce_loss_fn=WeightedBCELoss(weight_nonzero=pos_weight, weight_zero=1.0),
            mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            zero_rate=zero_rate,
            avg_nonzero_demand=float(y_train[y_train > 0].mean()) if (y_train > 0).any() else 1.0,
            pos_weight=pos_weight,
            loss_recipe="three_term",
            alpha_bce=0.2,
            w_gated=1.0,
            w_mag=1.0,
            use_fixed_weights=True,
            horizon_decay=0.95 if horizon > 1 else 1.0,
        )
        ds_model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
        ytr = {"final_forecast": ytr_fit, "base_forecast": ytr_fit}
        yva = {"final_forecast": yva_fit, "base_forecast": yva_fit}
        t0 = time.time()
        ds_model.fit(
            [*tr_fit, sktr_fit],
            ytr,
            validation_data=([*va_fit, skva_fit], yva),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
                )
            ],
            verbose=2,
        )
        train_seconds = time.time() - t0

        if horizon <= 1:
            pred = ds_model.predict([*te, sku_test], batch_size=4096, verbose=0)
            yhat = np.asarray(pred["final_forecast"]).reshape(-1)
            p = np.asarray(pred["non_zero_probability"]).reshape(-1)
            calib = {"scale": 1.0, "threshold": 0.0, "iwmae_rounded": None}
            if not args.no_calib:
                pred_va = ds_model.predict([*va, sku_val], batch_size=4096, verbose=0)
                yhat_va = np.asarray(pred_va["final_forecast"]).reshape(-1)
                p_va = np.asarray(pred_va["non_zero_probability"]).reshape(-1)
                calib = calibrate_iwmae_gate(y_val, yhat_va, p_va)
                print(
                    f"val IWMAE gate calib: scale={calib['scale']:.3f} "
                    f"thr={calib['threshold']:.3f} iwmae={calib['iwmae_rounded']:.3f}"
                )
                yhat = apply_iwmae_gate(
                    yhat, p, scale=calib["scale"], threshold=calib["threshold"]
                )
            results["models"]["deepsequence"] = {
                "train_seconds": train_seconds,
                "pos_weight": pos_weight,
                "use_sku": (not args.no_sku),
                "horizon": 1,
                "calibration": calib,
                "overall": kpi_block(y_test, yhat, p, mase_scale=mase_scale),
                "strata": strata_report(
                    y_test, yhat, p, sku_test_raw, volume_map, mase_scale=mase_scale
                ),
            }
        else:
            # One origin per series at first test month -> next H test months
            Xte_mh, yte_mh, skte_mh = build_mh_xy(
                X_test, y_test, test_df["id_var"].astype(str).to_numpy(), horizon
            )
            # Prefer the first window only (origin = first test month)
            keep = []
            seen = set()
            for i, s in enumerate(skte_mh):
                if s in seen:
                    continue
                seen.add(s)
                keep.append(i)
            keep = np.asarray(keep, dtype=np.int64)
            Xte_mh, yte_mh, skte_mh = Xte_mh[keep], yte_mh[keep], skte_mh[keep]
            sku_te_mh = np.array([sku_map[str(s)] for s in skte_mh], np.int32).reshape(-1, 1)
            te_mh = split_components(Xte_mh, cfg)
            pred = ds_model.predict([*te_mh, sku_te_mh], batch_size=2048, verbose=0)
            yhat = np.maximum(np.asarray(pred["final_forecast"], np.float32), 0.0)
            p = np.asarray(pred["non_zero_probability"], np.float32)
            if yhat.ndim == 1:
                yhat = yhat.reshape(-1, horizon)
            if p.ndim == 1:
                p = p.reshape(-1, horizon)
            calib = {"scale": 1.0, "threshold": 0.0, "iwmae_rounded": None}
            if not args.no_calib:
                pred_va = ds_model.predict([*va_fit, skva_fit], batch_size=2048, verbose=0)
                yhat_va = np.maximum(np.asarray(pred_va["final_forecast"], np.float32), 0.0)
                p_va = np.asarray(pred_va["non_zero_probability"], np.float32)
                calib = calibrate_iwmae_gate(
                    yva_fit.reshape(-1),
                    yhat_va.reshape(-1),
                    p_va.reshape(-1),
                )
                print(
                    f"val IWMAE gate calib: scale={calib['scale']:.3f} "
                    f"thr={calib['threshold']:.3f} iwmae={calib['iwmae_rounded']:.3f}"
                )
                yhat = apply_iwmae_gate(
                    yhat.reshape(-1),
                    p.reshape(-1),
                    scale=calib["scale"],
                    threshold=calib["threshold"],
                ).reshape(yhat.shape)
            mh = mh_horizon_report(
                yte_mh, yhat, p, skte_mh, volume_map, mase_scale
            )
            results["models"]["deepsequence"] = {
                "train_seconds": train_seconds,
                "pos_weight": pos_weight,
                "use_sku": (not args.no_sku),
                "horizon": horizon,
                "calibration": calib,
                "overall": mh["mean_1_to_H"]["overall"],
                "strata": mh["by_horizon"].get(str(horizon), {}).get("strata"),
                "by_horizon": {
                    h: mh["by_horizon"][h]["overall"] for h in mh["by_horizon"]
                },
                "n_origins": int(len(yte_mh)),
            }
            print("MH by horizon (iwmae_rounded):")
            for h, block in results["models"]["deepsequence"]["by_horizon"].items():
                print(f"  h={h}: iwmae={block['iwmae_rounded']:.3f}")
            print(
                f"  mean_1_to_H: iwmae={results['models']['deepsequence']['overall']['iwmae_rounded']:.3f}"
            )

    if "lightgbm" in selected:
        print("\n=== LightGBM ===")
        import lightgbm as lgb

        Xlgb_tr = np.concatenate([X_train, sku_train.astype(np.float32)], axis=1)
        Xlgb_va = np.concatenate([X_val, sku_val.astype(np.float32)], axis=1)
        Xlgb_te = np.concatenate([X_test, sku_test.astype(np.float32)], axis=1)
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            n_jobs=-1,
        )
        t0 = time.time()
        model.fit(
            Xlgb_tr,
            y_train,
            eval_set=[(Xlgb_va, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        yhat = np.maximum(model.predict(Xlgb_te), 0.0)
        p = np.clip(1.0 - np.exp(-yhat), 0, 1)
        results["models"]["lightgbm"] = {
            "train_seconds": time.time() - t0,
            "overall": kpi_block(y_test, yhat, p, mase_scale=mase_scale),
            "strata": strata_report(
                y_test, yhat, p, sku_test_raw, volume_map, mase_scale=mase_scale
            ),
        }

    need_seq = selected & {"deepar_lite", "temporal_transformer", "tft_lite"}
    if need_seq:
        print("\nBuilding sequence windows...")

        def build_windows_with_test():
            frames = []
            for df, X, split in (
                (train_df, X_train, "train"),
                (val_df, X_val, "val"),
                (test_df, X_test, "test"),
            ):
                tmp = df.copy()
                tmp["_split"] = split
                tmp["_X"] = list(X)
                frames.append(tmp)
            full = pd.concat(frames, ignore_index=True)
            full["id_var"] = full["id_var"].astype(str)
            Xs, ys, sks, sps = [], [], [], []
            for sku, g in full.groupby("id_var", sort=False):
                g = g.sort_values("ds", kind="mergesort")
                qty = g["Quantity"].to_numpy(np.float32)
                feats = np.stack(g["_X"].to_numpy())
                splits_g = g["_split"].to_numpy()
                n = len(g)
                for t in range(args.lookback, n):
                    hist_q = qty[t - args.lookback : t]
                    hist_x = feats[t - args.lookback : t]
                    win = np.concatenate(
                        [hist_q.reshape(args.lookback, 1), hist_x], axis=1
                    )
                    Xs.append(win)
                    ys.append(qty[t])
                    sks.append(sku)
                    sps.append(splits_g[t])
            return (
                np.asarray(Xs, np.float32),
                np.asarray(ys, np.float32),
                np.asarray(sks),
                np.asarray(sps),
                1 + X_train.shape[1],
            )

        Xseq, yseq, sku_seq_raw, split_seq, n_channels = build_windows_with_test()
        sku_seq = np.array([sku_map[str(s)] for s in sku_seq_raw], np.int32).reshape(-1, 1)
        m_tr = split_seq == "train"
        m_va = split_seq == "val"
        m_te = split_seq == "test"
        print(f"windows train/val/test={m_tr.sum()}/{m_va.sum()}/{m_te.sum()} ch={n_channels}")

        builders = {
            "deepar_lite": build_deepar,
            "temporal_transformer": build_transformer,
            "tft_lite": build_tft,
        }
        for name, builder in builders.items():
            if name not in selected:
                continue
            print(f"\n=== {name} ===")
            model = builder(args.lookback, n_skus, n_channels=n_channels)
            pos_weight = class_balance_pos_weight(y_train)
            wrap = AdaptiveWeightedModel(
                base_model=model,
                bce_loss_fn=WeightedBCELoss(weight_nonzero=pos_weight, weight_zero=1.0),
                mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
                zero_rate=zero_rate,
                avg_nonzero_demand=float(y_train[y_train > 0].mean())
                if (y_train > 0).any()
                else 1.0,
                pos_weight=pos_weight,
                loss_recipe="three_term",
                alpha_bce=0.2,
                w_gated=1.0,
                w_mag=1.0,
                use_fixed_weights=True,
            )
            wrap.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
            ytr = {
                "final_forecast": yseq[m_tr].reshape(-1, 1),
                "base_forecast": yseq[m_tr].reshape(-1, 1),
            }
            yva = {
                "final_forecast": yseq[m_va].reshape(-1, 1),
                "base_forecast": yseq[m_va].reshape(-1, 1),
            }
            t0 = time.time()
            wrap.fit(
                [Xseq[m_tr], sku_seq[m_tr]],
                ytr,
                validation_data=([Xseq[m_va], sku_seq[m_va]], yva),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
                    )
                ],
                verbose=2,
            )
            train_seconds = time.time() - t0
            pred_va = wrap.predict([Xseq[m_va], sku_seq[m_va]], batch_size=2048, verbose=0)
            yhat_va = np.asarray(pred_va["final_forecast"]).reshape(-1)
            p_va = np.asarray(pred_va["non_zero_probability"]).reshape(-1)
            calib = calibrate_iwmae_gate(yseq[m_va], yhat_va, p_va)
            print(
                f"val IWMAE gate calib: scale={calib['scale']:.3f} "
                f"thr={calib['threshold']:.3f} iwmae={calib['iwmae_rounded']:.3f}"
            )
            pred = wrap.predict([Xseq[m_te], sku_seq[m_te]], batch_size=2048, verbose=0)
            yhat_raw = np.asarray(pred["final_forecast"]).reshape(-1)
            p = np.asarray(pred["non_zero_probability"]).reshape(-1)
            yhat = apply_iwmae_gate(
                yhat_raw, p, scale=calib["scale"], threshold=calib["threshold"]
            )
            y_te = yseq[m_te]
            sku_te = sku_seq_raw[m_te]
            results["models"][name] = {
                "train_seconds": train_seconds,
                "pos_weight": pos_weight,
                "calibration": calib,
                "overall": kpi_block(y_te, yhat, p, mase_scale=mase_scale),
                "strata": strata_report(
                    y_te, yhat, p, sku_te, volume_map, mase_scale=mase_scale
                ),
            }

    # Comparison table
    comparison = []
    for model, payload in results["models"].items():
        o = payload["overall"]
        comparison.append(
            {
                "model": model,
                "iwmae_rounded": o.get("iwmae_rounded"),
                "mae_rounded": o.get("mae_all_rounded"),
                "mae_nonzero": o.get("mae_nonzero"),
                "mase_rounded": o.get("mase_rounded"),
                "occ_f1": o.get("occ_f1"),
                "underforecast_rate_nonzero": o.get("underforecast_rate_nonzero"),
                "aucroc": o.get("aucroc"),
                "bias": o.get("bias"),
            }
        )
    comparison = sorted(
        comparison,
        key=lambda r: (
            r["iwmae_rounded"] is None,
            r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
        ),
    )
    results["comparison"] = comparison
    results["mase_scale"] = mase_scale

    out = Path(args.out_json)
    out.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("PUBLIC CAR PARTS BAKE-OFF (primary: iwmae_rounded)")
    print("=" * 70)
    for row in comparison:
        print(
            f"  {row['model']:24s} iwmae={row['iwmae_rounded']:.3f} "
            f"mae={row['mae_rounded']:.3f} nz={row['mae_nonzero']:.3f} "
            f"occ_f1={row['occ_f1']:.3f} under={row['underforecast_rate_nonzero']:.3f} "
            f"bias={row['bias']:.3f}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
