#!/usr/bin/env python3
"""
Same-slice baseline comparison (800 SKUs, seed=42):

  1. DeepSequence hierarchical + three_term loss
  2. LightGBM (same causal tabular features)
  3. DeepAR-lite (LSTM + intermittent gate)
  4. Temporal Transformer (encoder + intermittent gate)

Sequence models use lookback windows of past Quantity + covariates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.losses import three_term_loss_config
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
    )
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_baseline_compare.json"),
    )
    return p.parse_args()


def filter_aligned(df, holidays, sku_set):
    mask = df["id_var"].isin(sku_set).to_numpy()
    return df.loc[mask].reset_index(drop=True), holidays.loc[mask].reset_index(drop=True)


def split_components(X, cfg):
    return (
        X[:, cfg.trend_indices].astype(np.float32),
        X[:, cfg.seasonal_indices].astype(np.float32),
        X[:, cfg.holiday_indices].astype(np.float32),
        X[:, cfg.regressor_indices].astype(np.float32),
    )


def eval_arrays(y, yhat, p=None, zero_rate=0.9):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    yhat = np.maximum(yhat, 0.0)
    yhat_round = np.rint(yhat)  # nearest integer forecast
    nz = y > 0
    y_bin = (y > 0).astype(np.float64)

    def _mae_block(pred):
        return {
            "mae_all": float(mean_absolute_error(y, pred)),
            "mae_nonzero": float(mean_absolute_error(y[nz], pred[nz])) if nz.any() else None,
            "mean_final": float(pred.mean()),
        }

    out = {
        **_mae_block(yhat),
        "mean_actual": float(y.mean()),
        "predict_zero_mae": float(mean_absolute_error(y, np.zeros_like(y))),
        "rounded": {
            **_mae_block(yhat_round),
            "mae_all_delta_vs_continuous": float(
                mean_absolute_error(y, yhat) - mean_absolute_error(y, yhat_round)
            ),
        },
    }
    if p is not None:
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        out["mean_p"] = float(p.mean())
        out["aucroc"] = float(roc_auc_score(y_bin, p))
        out["aucpr"] = float(average_precision_score(y_bin, p))
        thr = max(0.05, 1.0 - zero_rate)
        pred = (p >= thr).astype(np.float64)
        out["mae_gated_prior"] = float(mean_absolute_error(y, yhat * pred))
        out["mae_gated_prior_rounded"] = float(
            mean_absolute_error(y, yhat_round * pred)
        )
        # best-F1 gated MAE
        best = None
        best_r = None
        for t in np.linspace(0.05, 0.95, 19):
            g = (p >= t).astype(np.float64)
            tp = ((g == 1) & (y_bin == 1)).sum()
            fp = ((g == 1) & (y_bin == 0)).sum()
            fn = ((g == 0) & (y_bin == 1)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            mae_g = float(mean_absolute_error(y, yhat * g))
            mae_gr = float(mean_absolute_error(y, yhat_round * g))
            if best is None or f1 > best["f1"]:
                best = {"threshold": float(t), "f1": float(f1), "mae_gated": mae_g}
            if best_r is None or f1 > best_r["f1"]:
                best_r = {
                    "threshold": float(t),
                    "f1": float(f1),
                    "mae_gated": mae_gr,
                }
        out["best_f1"] = best
        out["best_f1_rounded"] = best_r
    else:
        # proxy probability from magnitude for ranking metrics only
        p_proxy = np.clip(yhat / (yhat.mean() + 1e-6) * (1 - zero_rate), 0, 1)
        try:
            out["aucroc_proxy"] = float(roc_auc_score(y_bin, p_proxy))
            out["aucpr_proxy"] = float(average_precision_score(y_bin, p_proxy))
        except ValueError:
            out["aucroc_proxy"] = None
            out["aucpr_proxy"] = None
    return out


# ---------------------------------------------------------------------------
# Sequence window builder
# ---------------------------------------------------------------------------

def build_sequence_tables(train_df, val_df, test_df, lookback: int):
    """
    Build next-step windows: X[t-L:t] history of y + calendar covars at each step,
    target y[t]. Val/test windows may use prior-split history for warm start.
    """
    frames = []
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d = df[["id_var", "ds", "Quantity"]].copy()
        d["split"] = split
        frames.append(d)
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["id_var", "ds"], kind="mergesort").reset_index(drop=True)

    # calendar covariates (normalized roughly)
    ds = pd.to_datetime(all_df["ds"])
    all_df["dow"] = ds.dt.dayofweek.to_numpy(np.float32) / 6.0
    all_df["month"] = (ds.dt.month.to_numpy(np.float32) - 1.0) / 11.0
    all_df["doy"] = (ds.dt.dayofyear.to_numpy(np.float32) - 1.0) / 365.0

    xs, ys, skus, splits = [], [], [], []
    for sku, g in all_df.groupby("id_var", sort=False):
        y = g["Quantity"].to_numpy(np.float32)
        cov = g[["dow", "month", "doy"]].to_numpy(np.float32)
        sp = g["split"].to_numpy()
        n = len(g)
        if n <= lookback:
            continue
        # channel: [y, dow, month, doy]
        seq = np.concatenate([y.reshape(-1, 1), cov], axis=1)
        for t in range(lookback, n):
            xs.append(seq[t - lookback : t])
            ys.append(y[t])
            skus.append(sku)
            splits.append(sp[t])

    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    skus = np.asarray(skus)
    splits = np.asarray(splits)
    return X, y, skus, splits


def build_deepar(lookback, n_skus, n_channels=4, hidden=64):
    hist = tf.keras.Input(shape=(lookback, n_channels), name="history")
    sku = tf.keras.Input(shape=(1,), dtype=tf.int32, name="sku_id")
    emb = tf.keras.layers.Embedding(n_skus, 8)(sku)
    emb = tf.keras.layers.Flatten()(emb)
    emb_t = tf.keras.layers.RepeatVector(lookback)(emb)
    x = tf.keras.layers.Concatenate(axis=-1)([hist, emb_t])
    h = tf.keras.layers.LSTM(hidden)(x)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    base = tf.keras.layers.Dense(1, activation="softplus", name="base_forecast")(h)
    p = tf.keras.layers.Dense(1, activation="sigmoid", name="non_zero_probability")(h)
    final = tf.keras.layers.Multiply(name="final_forecast")([base, p])
    return tf.keras.Model(
        [hist, sku],
        {"final_forecast": final, "non_zero_probability": p, "base_forecast": base},
        name="deepar_lite",
    )


def build_transformer(lookback, n_skus, n_channels=4, d_model=64, n_heads=4):
    hist = tf.keras.Input(shape=(lookback, n_channels), name="history")
    sku = tf.keras.Input(shape=(1,), dtype=tf.int32, name="sku_id")
    emb = tf.keras.layers.Embedding(n_skus, 8)(sku)
    emb = tf.keras.layers.Flatten()(emb)
    emb_t = tf.keras.layers.RepeatVector(lookback)(emb)
    x = tf.keras.layers.Concatenate(axis=-1)([hist, emb_t])
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=max(1, d_model // n_heads)
    )(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)
    ff = tf.keras.layers.Dense(d_model * 2, activation="relu")(x)
    ff = tf.keras.layers.Dense(d_model)(ff)
    x = tf.keras.layers.LayerNormalization()(x + ff)
    h = tf.keras.layers.GlobalAveragePooling1D()(x)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    base = tf.keras.layers.Dense(1, activation="softplus", name="base_forecast")(h)
    p = tf.keras.layers.Dense(1, activation="sigmoid", name="non_zero_probability")(h)
    final = tf.keras.layers.Multiply(name="final_forecast")([base, p])
    return tf.keras.Model(
        [hist, sku],
        {"final_forecast": final, "non_zero_probability": p, "base_forecast": base},
        name="temporal_transformer",
    )


def train_seq_model(model, Xtr, ytr, sku_tr, Xva, yva, sku_va, zero_rate, args, label):
    """Train sequence model with three_term-style compile losses."""
    cfg = three_term_loss_config(zero_rate, alpha_bce=0.2, w_gated=1.0, w_mag=1.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate if hasattr(args, "learning_rate") else 0.0025),
        loss=cfg["losses"],
        loss_weights=cfg["weights"],
    )
    ytr_d = {
        "final_forecast": ytr.reshape(-1, 1),
        "base_forecast": ytr.reshape(-1, 1),
        "non_zero_probability": (ytr > 0).astype(np.float32).reshape(-1, 1),
    }
    yva_d = {
        "final_forecast": yva.reshape(-1, 1),
        "base_forecast": yva.reshape(-1, 1),
        "non_zero_probability": (yva > 0).astype(np.float32).reshape(-1, 1),
    }
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]
    print(f"\n=== {label} ===")
    t0 = time.time()
    model.fit(
        [Xtr, sku_tr],
        ytr_d,
        validation_data=([Xva, sku_va], yva_d),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=2,
    )
    return time.time() - t0


def predict_seq(model, X, sku):
    pred = model.predict([X, sku], batch_size=4096, verbose=0)
    return (
        np.asarray(pred["final_forecast"]).reshape(-1),
        np.asarray(pred["non_zero_probability"]).reshape(-1),
    )


def main():
    args = parse_args()
    # attach lr for seq trainers
    args.learning_rate = 0.0025
    tf.keras.utils.set_random_seed(args.seed)
    data_dir = Path(args.data_dir)

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    h_tr = pd.read_csv(data_dir / "holiday_features_train.csv")
    h_va = pd.read_csv(data_dir / "holiday_features_val.csv")
    h_te = pd.read_csv(data_dir / "holiday_features_test.csv")

    rng = np.random.default_rng(args.seed)
    chosen = set(
        rng.choice(
            train_df["id_var"].unique(),
            size=min(args.max_skus, train_df["id_var"].nunique()),
            replace=False,
        )
    )
    train_df, h_tr = filter_aligned(train_df, h_tr, chosen)
    val_df, h_va = filter_aligned(val_df, h_va, chosen)
    test_df, h_te = filter_aligned(test_df, h_te, chosen)

    cats = pd.Categorical(train_df["id_var"])
    sku_map = {k: i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return df["id_var"].map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val, sku_test = enc(train_df), enc(val_df), enc(test_df)
    zero_rate = float((y_train == 0).mean())
    print(f"SKUs={n_skus} zero_rate={zero_rate:.3f} rows={len(y_train)}/{len(y_val)}/{len(y_test)}")

    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "seed": args.seed,
            "epochs": args.epochs,
            "lookback": args.lookback,
            "zero_rate": zero_rate,
            "note": "Same 800-SKU causal slice for all models",
        },
        "models": {},
    }

    # ------------------------------------------------------------------
    # Shared causal tabular features
    # ------------------------------------------------------------------
    cfg = load_feature_config()
    print("Building causal features...")
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, states = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    t_idx = cfg.trend_indices[0]
    tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    # ------------------------------------------------------------------
    # 1) DeepSequence three_term
    # ------------------------------------------------------------------
    tr, va, te = (
        split_components(X_train, cfg),
        split_components(X_val, cfg),
        split_components(X_test, cfg),
    )
    base = build_hierarchical_model_lightweight(
        n_temporal_features=len(cfg.trend_indices),
        n_fourier_features=len(cfg.seasonal_indices),
        n_holiday_features=len(cfg.holiday_indices),
        n_lag_features=len(cfg.regressor_indices),
        n_skus=n_skus,
        hidden_dim=48,
        sku_embedding_dim=4,
        dropout_rate=0.23,
        use_cross_layers=True,
        use_intermittent=True,
        n_changepoints=15,
    )
    _ = base(
        [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
        training=False,
    )
    pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))
    ds_model = AdaptiveWeightedModel(
        base_model=base,
        bce_loss_fn=WeightedBCELoss(weight_nonzero=pos_weight, weight_zero=1.0),
        mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        zero_rate=zero_rate,
        avg_nonzero_demand=float(y_train[y_train > 0].mean()),
        pos_weight=pos_weight,
        loss_recipe="three_term",
        alpha_bce=0.2,
        w_gated=1.0,
        w_mag=1.0,
        use_fixed_weights=True,
    )
    ds_model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
    ytr = {"final_forecast": y_train.reshape(-1, 1), "base_forecast": y_train.reshape(-1, 1)}
    yva = {"final_forecast": y_val.reshape(-1, 1), "base_forecast": y_val.reshape(-1, 1)}
    print("\n=== DeepSequence three_term ===")
    t0 = time.time()
    ds_model.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
            ),
        ],
        verbose=2,
    )
    ds_s = time.time() - t0
    pred = ds_model.predict([*te, sku_test], batch_size=4096, verbose=0)
    results["models"]["deepsequence_three_term"] = {
        "train_seconds": ds_s,
        "test": eval_arrays(
            y_test,
            pred["final_forecast"],
            pred["non_zero_probability"],
            zero_rate,
        ),
    }

    # ------------------------------------------------------------------
    # 2) LightGBM
    # ------------------------------------------------------------------
    print("\n=== LightGBM ===")
    Xlgb_tr = np.concatenate([X_train, sku_train.astype(np.float32)], axis=1)
    Xlgb_va = np.concatenate([X_val, sku_val.astype(np.float32)], axis=1)
    Xlgb_te = np.concatenate([X_test, sku_test.astype(np.float32)], axis=1)
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
    )
    t0 = time.time()
    lgb_model.fit(
        Xlgb_tr,
        y_train,
        eval_set=[(Xlgb_va, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    lgb_s = time.time() - t0
    yhat_lgb = np.maximum(lgb_model.predict(Xlgb_te), 0.0)
    # proxy p from magnitude for ranking
    p_lgb = np.clip(1.0 - np.exp(-yhat_lgb), 0, 1)
    results["models"]["lightgbm"] = {
        "train_seconds": lgb_s,
        "best_iteration": int(getattr(lgb_model, "best_iteration_", lgb_model.n_estimators_)),
        "test": eval_arrays(y_test, yhat_lgb, p_lgb, zero_rate),
    }

    # ------------------------------------------------------------------
    # 3-4) DeepAR-lite + Transformer (sequences)
    # ------------------------------------------------------------------
    print("\nBuilding sequence windows...")
    Xseq, yseq, sku_seq_raw, split_seq = build_sequence_tables(
        train_df, val_df, test_df, args.lookback
    )
    sku_seq = np.array([sku_map[s] for s in sku_seq_raw], dtype=np.int32).reshape(-1, 1)
    m_tr = split_seq == "train"
    m_va = split_seq == "val"
    m_te = split_seq == "test"
    print(
        f"windows train/val/test={m_tr.sum()}/{m_va.sum()}/{m_te.sum()} "
        f"lookback={args.lookback}"
    )

    for name, builder in [
        ("deepar_lite", build_deepar),
        ("temporal_transformer", build_transformer),
    ]:
        model = builder(args.lookback, n_skus)
        train_s = train_seq_model(
            model,
            Xseq[m_tr],
            yseq[m_tr],
            sku_seq[m_tr],
            Xseq[m_va],
            yseq[m_va],
            sku_seq[m_va],
            zero_rate,
            args,
            name,
        )
        yhat, p = predict_seq(model, Xseq[m_te], sku_seq[m_te])
        results["models"][name] = {
            "train_seconds": train_s,
            "test": eval_arrays(yseq[m_te], yhat, p, zero_rate),
            "n_test_windows": int(m_te.sum()),
        }

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    comparison = []
    for name, payload in results["models"].items():
        t = payload["test"]
        r = t.get("rounded") or {}
        comparison.append(
            {
                "model": name,
                "test_mae": t["mae_all"],
                "test_mae_rounded": r.get("mae_all"),
                "mae_rounding_gain": r.get("mae_all_delta_vs_continuous"),
                "test_mae_nonzero": t["mae_nonzero"],
                "test_mae_nonzero_rounded": r.get("mae_nonzero"),
                "mean_final": t["mean_final"],
                "mean_final_rounded": r.get("mean_final"),
                "aucroc": t.get("aucroc"),
                "aucpr": t.get("aucpr"),
                "best_f1_gated_mae": (t.get("best_f1") or {}).get("mae_gated"),
                "best_f1_gated_mae_rounded": (t.get("best_f1_rounded") or {}).get(
                    "mae_gated"
                ),
                "train_seconds": payload["train_seconds"],
            }
        )
    comparison = sorted(
        comparison, key=lambda row: row.get("test_mae_rounded") or row["test_mae"]
    )
    results["comparison"] = comparison
    results["historical_full_data_reference"] = {
        "lightgbm_test_mae": 1.2864,
        "twostage_test_mae": 0.9869,
        "simple_additive_test_mae": 0.9876,
        "note": "Full ~6099 SKU historical runs — not same slice",
    }

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (same 800 SKUs)")
    print("=" * 70)
    print(json.dumps(comparison, indent=2))
    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
