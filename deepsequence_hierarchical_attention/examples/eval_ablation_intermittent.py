#!/usr/bin/env python3
"""
Ablation: same 800 SKUs / recipe, WITH vs WITHOUT causal intermittent features.

Also fits a LightGBM baseline on the same feature matrix as a sanity check
that the slice is learnable.
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
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.intermittent_features import (
    INTERMITTENT_FEATURE_NAMES,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
    )
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden_dim", type=int, default=48)
    p.add_argument("--sku_embedding_dim", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=0.0025)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=str(ROOT / "eval_results_ablation.json"))
    return p.parse_args()


def filter_aligned(df, holidays, sku_set):
    mask = df["id_var"].isin(sku_set).to_numpy()
    return df.loc[mask].reset_index(drop=True), holidays.loc[mask].reset_index(drop=True)


def build_feature_matrices(cfg, train_df, val_df, test_df, h_tr, h_va, h_te, include_intermittent: bool):
    """Build full feature matrices; optionally drop intermittent columns from regressor."""
    Xtr, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte, states = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)

    # Normalize time_index to [0,1] using train stats
    t_col = "time_index"
    tmin, tmax = float(Xtr[t_col].min()), float(Xtr[t_col].max())
    span = max(tmax - tmin, 1.0)
    for X in (Xtr, Xva, Xte):
        X[t_col] = (X[t_col] - tmin) / span

    if include_intermittent:
        feature_names = list(cfg.feature_names)
        trend_idx = cfg.trend_indices
        seasonal_idx = cfg.seasonal_indices
        holiday_idx = cfg.holiday_indices
        regressor_names = cfg.lag_names + cfg.intermittent_names
    else:
        # Drop intermittent columns; keep lags only in regressor
        drop = list(INTERMITTENT_FEATURE_NAMES)
        feature_names = [c for c in cfg.feature_names if c not in drop]
        Xtr = Xtr[feature_names]
        Xva = Xva[feature_names]
        Xte = Xte[feature_names]
        # Recompute indices against reduced frame
        name_to_i = {n: i for i, n in enumerate(feature_names)}
        trend_idx = [name_to_i["time_index"]]
        seasonal_idx = [name_to_i[n] for n in cfg.cyclical_names]
        holiday_idx = [name_to_i[n] for n in cfg.holiday_names + cfg.binary_holiday_names]
        regressor_names = list(cfg.lag_names)

    regressor_idx = [feature_names.index(n) for n in regressor_names]

    def pack(X):
        A = X.to_numpy(np.float32)
        return (
            A[:, trend_idx],
            A[:, seasonal_idx],
            A[:, holiday_idx],
            A[:, regressor_idx],
            A,  # flat for LightGBM
        )

    return {
        "feature_names": feature_names,
        "regressor_names": regressor_names,
        "n_features": len(feature_names),
        "train": pack(Xtr),
        "val": pack(Xva),
        "test": pack(Xte),
        "dims": {
            "trend": len(trend_idx),
            "seasonal": len(seasonal_idx),
            "holiday": len(holiday_idx),
            "regressor": len(regressor_idx),
        },
    }


def classification_metrics(y, yhat, p, zero_rate):
    y = y.reshape(-1)
    yhat = yhat.reshape(-1)
    p = p.reshape(-1)
    y_bin = (y > 0).astype(np.float32)
    prior_thr = max(0.05, 1.0 - zero_rate)

    def at(thr):
        pred = (p >= thr).astype(np.float32)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_bin, pred, average="binary", zero_division=0
        )
        nz = y > 0
        return {
            "threshold": float(thr),
            "accuracy": float(((pred == y_bin).mean())),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mae_all": float(mean_absolute_error(y, yhat)),
            "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else None,
            "mae_gated": float(mean_absolute_error(y, yhat * pred)),
            "pred_nonzero_rate": float(pred.mean()),
        }

    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        m = at(thr)
        if best is None or m["f1"] > best["f1"]:
            best = m

    out = at(prior_thr)
    out["aucpr"] = float(average_precision_score(y_bin, p))
    out["aucroc"] = float(roc_auc_score(y_bin, p))
    out["mean_p"] = float(p.mean())
    out["mean_final"] = float(yhat.mean())
    out["mean_actual"] = float(y.mean())
    out["best_f1"] = best
    # accuracy at 0.5 (Keras-style)
    out["accuracy_at_0_5"] = float((((p >= 0.5).astype(np.float32) == y_bin).mean()))
    return out


def train_nn_full(feats, y_train, y_val, y_test, sku_train, sku_val, sku_test, n_skus, zero_rate, avg_nz, args, label):
    dims = feats["dims"]
    tr, va, te = feats["train"][:4], feats["val"][:4], feats["test"][:4]

    base = build_hierarchical_model_lightweight(
        n_temporal_features=dims["trend"],
        n_fourier_features=dims["seasonal"],
        n_holiday_features=dims["holiday"],
        n_lag_features=dims["regressor"],
        n_skus=n_skus,
        hidden_dim=args.hidden_dim,
        sku_embedding_dim=args.sku_embedding_dim,
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
    wrapped = AdaptiveWeightedModel(
        base_model=base,
        bce_loss_fn=WeightedBCELoss(weight_nonzero=pos_weight, weight_zero=1.0),
        mae_loss_fn=tf.keras.losses.MeanAbsoluteError(),
        zero_rate=zero_rate,
        avg_nonzero_demand=avg_nz,
        pos_weight=pos_weight,
        use_fixed_weights=True,
        bce_weight=0.5,
        mae_weight=0.5,
    )
    wrapped.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))

    ytr = {
        "base_forecast": y_train,
        "final_forecast": y_train,
        "non_zero_binary": (y_train > 0).astype(np.float32),
        "non_zero_probability": (y_train > 0).astype(np.float32),
    }
    yva = {
        "base_forecast": y_val,
        "final_forecast": y_val,
        "non_zero_binary": (y_val > 0).astype(np.float32),
        "non_zero_probability": (y_val > 0).astype(np.float32),
    }
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]

    print(f"\n=== Training NN ({label}) regressor={feats['regressor_names']} ===")
    t0 = time.time()
    wrapped.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=2,
    )
    train_s = time.time() - t0

    def eval_split(parts, y, sku):
        pred = wrapped.predict([*parts, sku], batch_size=4096, verbose=0)
        yhat = np.asarray(pred["final_forecast"]).reshape(-1)
        p = np.asarray(pred["non_zero_probability"]).reshape(-1)
        return classification_metrics(y, yhat, p, zero_rate)

    return {
        "label": label,
        "regressor_names": feats["regressor_names"],
        "n_features": feats["n_features"],
        "train_seconds": train_s,
        "val": eval_split(va, y_val, sku_val),
        "test": eval_split(te, y_test, sku_test),
    }


def train_lgb(feats, y_train, y_val, y_test, sku_train, sku_val, sku_test, zero_rate):
    if lgb is None:
        return {"error": "lightgbm not installed"}

    def flat(parts, sku):
        # concat component arrays + sku
        return np.concatenate([*parts[:4], sku.astype(np.float32)], axis=1)

    Xtr = flat(feats["train"], sku_train)
    Xva = flat(feats["val"], sku_val)
    Xte = flat(feats["test"], sku_test)

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    print("\n=== Training LightGBM sanity baseline ===")
    t0 = time.time()
    model.fit(
        Xtr,
        y_train,
        eval_set=[(Xva, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    train_s = time.time() - t0
    yhat = model.predict(Xte)
    yhat = np.maximum(yhat, 0.0)
    # proxy probability: clipped normalized prediction
    p = np.clip(yhat / (yhat.mean() + 1e-6) * (1 - zero_rate), 0, 1)
    metrics = classification_metrics(y_test, yhat, p, zero_rate)
    # override silly p-based metrics for LGB — keep MAE primary
    return {
        "label": "lightgbm_same_features",
        "train_seconds": train_s,
        "test_mae_all": float(mean_absolute_error(y_test, yhat)),
        "test_mae_nonzero": float(
            mean_absolute_error(y_test[y_test > 0], yhat[y_test > 0])
        )
        if (y_test > 0).any()
        else None,
        "mean_final": float(yhat.mean()),
        "mean_actual": float(y_test.mean()),
        "metrics_proxy": metrics,
    }


def main():
    args = parse_args()
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
        rng.choice(train_df["id_var"].unique(), size=min(args.max_skus, train_df["id_var"].nunique()), replace=False)
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
    avg_nz = float(y_train[y_train > 0].mean())
    print(f"SKUs={n_skus} rows={len(y_train)}/{len(y_val)}/{len(y_test)} zero_rate={zero_rate:.3f}")

    cfg = load_feature_config()

    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "epochs": args.epochs,
            "seed": args.seed,
            "zero_rate": zero_rate,
            "recipe": "AdaptiveWeightedModel 50/50",
        },
        "historical_baselines_full_data": {
            "lightgbm_test_mae": 1.2864,
            "twostage_test_mae": 0.9869,
            "simple_additive_test_mae": 0.9876,
        },
        "runs": [],
    }

    for include_intermittent, label in [
        (False, "nn_lags_only"),
        (True, "nn_lags_plus_intermittent"),
    ]:
        print(f"\nBuilding features include_intermittent={include_intermittent}")
        feats = build_feature_matrices(
            cfg, train_df, val_df, test_df, h_tr, h_va, h_te, include_intermittent
        )
        run = train_nn_full(
            feats,
            y_train,
            y_val,
            y_test,
            sku_train,
            sku_val,
            sku_test,
            n_skus,
            zero_rate,
            avg_nz,
            args,
            label,
        )
        results["runs"].append(run)

        # LightGBM on the WITH-intermittent feature set only once at end
        if include_intermittent:
            results["lightgbm"] = train_lgb(
                feats, y_train, y_val, y_test, sku_train, sku_val, sku_test, zero_rate
            )

    # Compact comparison table
    comparison = []
    for run in results["runs"]:
        comparison.append(
            {
                "model": run["label"],
                "test_mae_all": run["test"]["mae_all"],
                "test_mae_nonzero": run["test"]["mae_nonzero"],
                "test_mae_gated_best_f1": run["test"]["best_f1"]["mae_gated"],
                "test_acc_at_0_5": run["test"]["accuracy_at_0_5"],
                "test_aucroc": run["test"]["aucroc"],
                "test_aucpr": run["test"]["aucpr"],
                "test_best_f1": run["test"]["best_f1"]["f1"],
                "mean_final": run["test"]["mean_final"],
            }
        )
    if "lightgbm" in results and "test_mae_all" in results["lightgbm"]:
        comparison.append(
            {
                "model": "lightgbm_same_features",
                "test_mae_all": results["lightgbm"]["test_mae_all"],
                "test_mae_nonzero": results["lightgbm"]["test_mae_nonzero"],
                "mean_final": results["lightgbm"]["mean_final"],
            }
        )
    results["comparison"] = comparison

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)
    print(json.dumps(comparison, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
