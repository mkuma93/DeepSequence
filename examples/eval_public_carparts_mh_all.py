#!/usr/bin/env python3
"""
Car Parts multi-horizon bake-off (H=6): all models, same origin protocol.

Origin: last month of val (stand before test). Forecast next H test months.

  - deepsequence: direct MH head (use_sku=False)
  - lightgbm: multi-output H targets from same MH windows
  - croston/sba/tsb: recursive classical
  - deepar/tst/tft: 1-step models, recursive qty rollout with monthly
    causal feature rebuild (no test leakage)
"""

from __future__ import annotations

import argparse
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
from deepsequence_hierarchical_attention.intermittent_features import (
    empty_state,
)
from eval_helpers import (
    add_panel_seed_args,
    build_deepar,
    build_tft,
    build_transformer,
    class_balance_pos_weight,
    filter_aligned,
    kpi_block,
    resolve_eval_seeds,
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
    "deepar_lite",
    "temporal_transformer",
    "tft_lite",
    "croston",
    "sba",
    "tsb",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(ROOT / "public_data/car_parts/panel"))
    p.add_argument("--feature_config", default=str(ROOT / "feature_config_monthly.yaml"))
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=12)
    p.add_argument("--horizon", type=int, default=6)
    add_panel_seed_args(p)
    p.add_argument("--mase_season", type=int, default=12)
    p.add_argument("--models", default=",".join(ALL_MODELS))
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_public_carparts_mh_all.json"),
    )
    return p.parse_args()


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
    return out


def assemble_monthly_row(date, state, lag_periods, tmin, span):
    date = pd.Timestamp(date).normalize()
    mi = float(date.year * 12 + date.month)
    month = float(date.month)
    feats = state.features_at(date, lags=lag_periods, gap_unit="months")
    row = [
        (mi - tmin) / span,
        np.sin(2 * np.pi * mi / 3.0),
        np.cos(2 * np.pi * mi / 3.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
    ]
    for lag in lag_periods:
        row.append(float(feats[f"lag_{lag}"]))
    row.extend(
        [
            float(feats["months_since_last_sale"]),
            float(feats["last_sale_quantity"]),
            float(feats["lifetime_cumsum"]),
        ]
    )
    return np.asarray(row, np.float32)


def classical_recursive(histories, horizon, alpha=0.1):
    """histories: list of 1d y arrays (true history before origin)."""
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

    data_seed, train_seed = resolve_eval_seeds(
        args.seed, args.data_seed, args.train_seed
    )
    tf.keras.utils.set_random_seed(train_seed)
    H = int(args.horizon)
    data_dir = Path(args.data_dir)

    print("Loading Car Parts panel...")
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
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
    for df in (train_df, val_df, test_df):
        df.sort_values(["id_var", "ds"], kind="mergesort", inplace=True)
        df.reset_index(drop=True, inplace=True)

    volume_map, volume_stats = train_volume_terciles(train_df)
    mase_scale = train_mase_scale(train_df, season=args.mase_season)
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
    zero_rate = float((y_train == 0).mean())
    avg_nz = float(y_train[y_train > 0].mean()) if (y_train > 0).any() else 1.0
    pos_weight = class_balance_pos_weight(y_train)
    print(
        f"n_skus={n_skus} zero_rate={zero_rate:.3f} H={H} pos_weight={pos_weight:.3f}"
    )

    cfg = load_feature_config(args.feature_config)
    lag_periods = cfg.lag_periods
    print(f"Features v{cfg.config['metadata'].get('version')} n={cfg.total_features}")
    Xtr_df, states = cfg.create_features(train_df, None, return_states=True)
    Xva_df, states = cfg.create_features(val_df, None, prior_states=states, return_states=True)
    Xte_df, _ = cfg.create_features(test_df, None, prior_states=states, return_states=True)
    X_train = Xtr_df.to_numpy(np.float32, copy=True)
    X_val = Xva_df.to_numpy(np.float32, copy=True)
    X_test = Xte_df.to_numpy(np.float32, copy=True)
    t_idx = cfg.trend_indices[0]
    tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    # --- Fixed test origins: first test month per sku ---
    sk_te = test_df["id_var"].astype(str).to_numpy()
    X_origin, y_true_mh, sk_origin = build_mh_xy(X_test, y_test, sk_te, H)
    # first window only
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

    # True history before test for classical / recursive state
    hist_y = []
    hist_dates = []
    test_dates = []
    for sku in sk_origin:
        tr = train_df[train_df["id_var"].astype(str) == sku]
        va = val_df[val_df["id_var"].astype(str) == sku]
        te = test_df[test_df["id_var"].astype(str) == sku].sort_values("ds")
        hist_y.append(
            np.concatenate(
                [tr["Quantity"].to_numpy(np.float64), va["Quantity"].to_numpy(np.float64)]
            )
        )
        hist_dates.append(
            pd.to_datetime(
                pd.concat([tr["ds"], va["ds"]], ignore_index=True)
            ).to_numpy()
        )
        test_dates.append(pd.to_datetime(te["ds"]).to_numpy()[:H])

    results = {
        "config": {
            "dataset": "Monash Car Parts",
            "protocol": "fixed origin = first test month; forecast H test months",
            "horizon": H,
            "n_skus": n_skus,
            "seed": args.seed,
            "data_seed": data_seed,
            "train_seed": train_seed,
            "sku_list": args.sku_list,
            "use_sku_deepsequence": False,
            "feature_version": cfg.config["metadata"].get("version"),
            "volume_stats": volume_stats,
            "models": sorted(selected),
        },
        "models": {},
    }

    # -------- Classical recursive --------
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
            metrics = mh_metrics(y_true_mh, yhat, p, mase_scale=mase_scale)
            results["models"][name] = {
                "method": "recursive",
                "train_seconds": dt / max(len(classical), 1),
                **metrics,
            }

    # -------- DeepSequence direct MH --------
    if "deepsequence" in selected:
        print("\n=== DeepSequence direct MH ===")
        sk_tr = train_df["id_var"].astype(str).to_numpy()
        sk_va = val_df["id_var"].astype(str).to_numpy()
        Xtr_mh, ytr_mh, sktr_mh = build_mh_xy(X_train, y_train, sk_tr, H)
        Xva_mh, yva_mh, skva_mh = build_mh_xy(X_val, y_val, sk_va, H)
        if len(Xva_mh) == 0:
            # rare; use combined boundary windows
            Xva_mh, yva_mh, skva_mh = Xtr_mh[-n_skus:], ytr_mh[-n_skus:], sktr_mh[-n_skus:]
        sktr = np.array([sku_map[str(s)] for s in sktr_mh], np.int32).reshape(-1, 1)
        skva = np.array([sku_map[str(s)] for s in skva_mh], np.int32).reshape(-1, 1)
        tr = split_components(Xtr_mh, cfg)
        va = split_components(Xva_mh, cfg)
        te = split_components(X_origin, cfg)
        print(f"MH windows train/val={len(ytr_mh)}/{len(yva_mh)}")
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
            use_sku=False,
            horizon=H,
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
        metrics = mh_metrics(y_true_mh, yhat, p, mase_scale=mase_scale)
        results["models"]["deepsequence"] = {
            "method": "direct_mh",
            "use_sku": False,
            "train_seconds": time.time() - t0,
            **metrics,
        }

    # -------- LightGBM multi-output --------
    if "lightgbm" in selected:
        print("\n=== LightGBM multi-output MH ===")
        import lightgbm as lgb

        sk_tr = train_df["id_var"].astype(str).to_numpy()
        sk_va = val_df["id_var"].astype(str).to_numpy()
        Xtr_mh, ytr_mh, sktr_mh = build_mh_xy(X_train, y_train, sk_tr, H)
        Xva_mh, yva_mh, skva_mh = build_mh_xy(X_val, y_val, sk_va, H)
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
                random_state=args.seed,
                n_jobs=-1,
                verbosity=-1,
            )
        )
        model.fit(Xtr, ytr_mh)
        yhat = np.maximum(model.predict(Xte), 0.0).astype(np.float32)
        p = np.clip(1.0 - np.exp(-yhat), 0, 1)
        metrics = mh_metrics(y_true_mh, yhat, p, mase_scale=mase_scale)
        results["models"]["lightgbm"] = {
            "method": "multi_output",
            "train_seconds": time.time() - t0,
            **metrics,
        }

    # -------- Sequence models: 1-step train + recursive feature rebuild --------
    need_seq = selected & {"deepar_lite", "temporal_transformer", "tft_lite"}
    if need_seq:
        print("\nBuilding 1-step sequence windows (train/val)...")
        frames = []
        for df, X, split in (
            (train_df, X_train, "train"),
            (val_df, X_val, "val"),
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
            for t in range(args.lookback, len(g)):
                hist_q = qty[t - args.lookback : t]
                hist_x = feats[t - args.lookback : t]
                win = np.concatenate(
                    [hist_q.reshape(args.lookback, 1), hist_x], axis=1
                )
                Xs.append(win)
                ys.append(qty[t])
                sks.append(sku)
                sps.append(splits_g[t])
        Xseq = np.asarray(Xs, np.float32)
        yseq = np.asarray(ys, np.float32).reshape(-1, 1)
        sku_seq = np.array([sku_map[str(s)] for s in sks], np.int32).reshape(-1, 1)
        split_seq = np.asarray(sps)
        m_tr = split_seq == "train"
        m_va = split_seq == "val"
        n_channels = 1 + X_train.shape[1]
        print(f"seq windows train/val={m_tr.sum()}/{m_va.sum()} ch={n_channels}")

        builders = {
            "deepar_lite": build_deepar,
            "temporal_transformer": build_transformer,
            "tft_lite": build_tft,
        }
        # Precompute raw (unnormalized) month_index scale for assemble
        # tmin/span already on normalized month_index scale used in X;
        # assemble_monthly_row uses same tmin/span on raw month index —
        # recover raw from train dates.
        raw_mi = (
            pd.to_datetime(train_df["ds"]).dt.year * 12
            + pd.to_datetime(train_df["ds"]).dt.month
        ).to_numpy(np.float64)
        raw_tmin, raw_tmax = float(raw_mi.min()), float(raw_mi.max())
        raw_span = max(raw_tmax - raw_tmin, 1.0)

        for name, builder in builders.items():
            if name not in selected:
                continue
            print(f"\n=== {name} 1-step train + recursive MH ===")
            base = builder(args.lookback, n_skus, n_channels=n_channels)
            # seq models expect [history, sku]
            # AdaptiveWeightedModel wraps keras model with those inputs
            t0 = time.time()
            wrap = AdaptiveWeightedModel(
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
            )
            wrap.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
            wrap.fit(
                [Xseq[m_tr], sku_seq[m_tr]],
                {"final_forecast": yseq[m_tr], "base_forecast": yseq[m_tr]},
                validation_data=(
                    [Xseq[m_va], sku_seq[m_va]],
                    {"final_forecast": yseq[m_va], "base_forecast": yseq[m_va]},
                ),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
                    )
                ],
                verbose=2,
            )
            train_s = time.time() - t0

            # Recursive rollout from origin
            n_o = len(sk_origin)
            yhat = np.zeros((n_o, H), np.float32)
            p_mat = np.zeros((n_o, H), np.float32)
            # init states + qty lookback from hist
            states = []
            qty_bufs = []
            for i, sku in enumerate(sk_origin):
                st = empty_state(max_lag=max(lag_periods), rate_window=12)
                for d, q in zip(hist_dates[i], hist_y[i]):
                    st.update(pd.Timestamp(d), float(q))
                states.append(st)
                q = list(hist_y[i][-args.lookback :])
                while len(q) < args.lookback:
                    q = [0.0] + q
                qty_bufs.append(q[-args.lookback :])

            for h in range(H):
                Xb = np.zeros((n_o, args.lookback, n_channels), np.float32)
                skb = sku_origin.copy()
                for i in range(n_o):
                    date = pd.Timestamp(test_dates[i][h])
                    # feature row at this date from current state (no leakage)
                    feat = assemble_monthly_row(
                        date, states[i], lag_periods, raw_tmin, raw_span
                    )
                    # lookback window: past qty + past feats approximated by
                    # repeating current feat for non-qty channels (qty exact)
                    for t in range(args.lookback):
                        Xb[i, t, 0] = qty_bufs[i][t]
                        Xb[i, t, 1:] = feat
                pred = wrap.predict([Xb, skb], batch_size=2048, verbose=0)
                yh = np.asarray(pred["final_forecast"]).reshape(-1)
                pp = np.asarray(pred["non_zero_probability"]).reshape(-1)
                yhat[:, h] = np.maximum(yh, 0.0)
                p_mat[:, h] = pp
                for i in range(n_o):
                    q = float(yhat[i, h])
                    date = pd.Timestamp(test_dates[i][h])
                    states[i].update(date, q)
                    qty_bufs[i] = qty_bufs[i][1:] + [q]

            metrics = mh_metrics(y_true_mh, yhat, p_mat, mase_scale=mase_scale)
            results["models"][name] = {
                "method": "recursive_1step",
                "train_seconds": train_s,
                **metrics,
            }

    # -------- Leaderboard --------
    comparison = []
    for model, payload in results["models"].items():
        o = payload["mean_1_to_H"]
        comparison.append(
            {
                "model": model,
                "method": payload.get("method"),
                "iwmae_rounded": o.get("iwmae_rounded"),
                "mae_rounded": o.get("mae_all_rounded"),
                "mae_nonzero": o.get("mae_nonzero"),
                "occ_f1": o.get("occ_f1"),
                "bias": o.get("bias"),
                "h1_iwmae": payload.get("by_horizon", {}).get("1", {}).get("iwmae_rounded"),
                "h6_iwmae": payload.get("by_horizon", {}).get(str(H), {}).get("iwmae_rounded"),
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
    print("\n" + "=" * 72)
    print(f"CAR PARTS MH BAKE-OFF  H={H}  (primary: mean_1_to_H iwmae_rounded)")
    print("=" * 72)
    for row in comparison:
        print(
            f"  {row['model']:22s} {row['method']:16s} "
            f"mean={row['iwmae_rounded']:.3f} h1={row['h1_iwmae']:.3f} "
            f"h{H}={row['h6_iwmae']:.3f} bias={row['bias']:+.3f}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
