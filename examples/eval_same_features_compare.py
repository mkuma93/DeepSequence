#!/usr/bin/env python3
"""
Same-feature bake-off (800 SKUs, seed=42).

All models see the SAME causal feature contract (feature_config):
  trend + cyclical + lags + intermittent + holiday distances
  + Quantity history for sequence models.
  (v1.6+: no binary is_* holidays — redundant with days_from_*)

  - DeepSequence (tabular, gated)
  - LightGBM L1 (tabular)
  - DeepAR-lite (sequence windows of Quantity + full X)
  - Temporal transformer (same windows)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.losses import three_term_loss_config
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from eval_helpers import (
    build_deepar,
    build_transformer,
    filter_aligned,
    kpi_block,
    predict_seq,
    split_components,
    strata_report,
    train_volume_terciles,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default=None,
        help="Panel data directory (or set DEEPSEQUENCE_DATA_DIR). Required for a real bake-off run.",
    )
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_same_features_v16_distance_holidays.json"),
    )
    return p.parse_args()


def build_full_feature_sequences(
    train_df, val_df, test_df, X_train, X_val, X_test, lookback: int
):
    """
    Per-SKU causal windows:
      hist[τ] = [Quantity_τ, X_τ...] for τ in [t-L, t)
      target   = Quantity_t
    """
    metas = []
    feats = []
    offset = 0
    for split, df, X in [
        ("train", train_df, X_train),
        ("val", val_df, X_val),
        ("test", test_df, X_test),
    ]:
        y = df["Quantity"].to_numpy(np.float32).reshape(-1, 1)
        block = np.concatenate([y, X.astype(np.float32)], axis=1)
        feats.append(block)
        n = len(df)
        metas.append(
            pd.DataFrame(
                {
                    "id_var": df["id_var"].to_numpy(),
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
    return (
        X,
        np.asarray(ys, np.float32),
        np.asarray(skus),
        np.asarray(splits),
        n_channels,
    )


def train_seq_three_term(model, Xtr, ytr, skutr, Xva, yva, skuva, zero_rate, args, label):
    cfg = three_term_loss_config(zero_rate, alpha_bce=0.2, w_gated=1.0, w_mag=1.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0025),
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
    print(f"\n=== {label} (same features, three_term) ===")
    t0 = time.time()
    model.fit(
        [Xtr, skutr],
        ytr_d,
        validation_data=([Xva, skuva], yva_d),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
            ),
        ],
        verbose=2,
    )
    return time.time() - t0


def main():
    import os

    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    data_dir_raw = args.data_dir or os.environ.get("DEEPSEQUENCE_DATA_DIR")
    if not data_dir_raw:
        raise SystemExit(
            "Pass --data_dir PATH or set DEEPSEQUENCE_DATA_DIR to your panel data directory."
        )
    data_dir = Path(data_dir_raw)

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

    volume_map, volume_stats = train_volume_terciles(train_df)
    print("Volume terciles:")
    for b, st in volume_stats.items():
        print(
            f"  {b}: n={st['n_skus']} mean_vol={st['train_volume_mean_sku']:.1f} "
            f"zr={st['train_zero_rate']:.3f}"
        )

    cats = pd.Categorical(train_df["id_var"])
    sku_map = {k: i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return df["id_var"].map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val, sku_test = enc(train_df), enc(val_df), enc(test_df)
    sku_test_raw = test_df["id_var"].to_numpy()
    zero_rate = float((y_train == 0).mean())

    cfg = load_feature_config()
    print("Building causal features (shared contract)...")
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, states = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
    feature_names = list(Xtr_df.columns)
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    t_idx = cfg.trend_indices[0]
    tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "seed": args.seed,
            "lookback": args.lookback,
            "zero_rate": zero_rate,
            "feature_contract": f"feature_config v{cfg.config['metadata']['version']} + Quantity in sequence hist",
            "n_tabular_features": len(feature_names),
            "feature_names": feature_names,
            "sequence_channels": ["Quantity"] + feature_names,
            "volume_stats": volume_stats,
            "note": "DeepAR/TST use full causal X in lookback (same columns as tabular). Holiday distance only (no is_* binaries).",
        },
        "models": {},
    }

    tr, va, te = (
        split_components(X_train, cfg),
        split_components(X_val, cfg),
        split_components(X_test, cfg),
    )

    # ------------------------------------------------------------------
    # DeepSequence
    # ------------------------------------------------------------------
    print("\n=== DeepSequence (same features) ===")
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
    t0 = time.time()
    ds_model.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            )
        ],
        verbose=2,
    )
    ds_s = time.time() - t0
    pred = ds_model.predict([*te, sku_test], batch_size=4096, verbose=0)
    yhat_ds = np.asarray(pred["final_forecast"]).reshape(-1)
    p_ds = np.asarray(pred["non_zero_probability"]).reshape(-1)
    results["models"]["deepsequence"] = {
        "train_seconds": ds_s,
        "overall": kpi_block(y_test, yhat_ds, p_ds),
        "strata": strata_report(y_test, yhat_ds, p_ds, sku_test_raw, volume_map),
    }

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    print("\n=== LightGBM L1 (same features) ===")
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
    p_lgb = np.clip(1.0 - np.exp(-yhat_lgb), 0, 1)
    results["models"]["lightgbm"] = {
        "train_seconds": lgb_s,
        "overall": kpi_block(y_test, yhat_lgb, p_lgb),
        "strata": strata_report(y_test, yhat_lgb, p_lgb, sku_test_raw, volume_map),
    }

    # ------------------------------------------------------------------
    # DeepAR + TST on full-feature sequences
    # ------------------------------------------------------------------
    print("\nBuilding FULL-feature sequence windows...")
    Xseq, yseq, sku_seq_raw, split_seq, n_channels = build_full_feature_sequences(
        train_df, val_df, test_df, X_train, X_val, X_test, args.lookback
    )
    print(f"sequence channels={n_channels} (=1 Quantity + {n_channels-1} causal features)")
    sku_seq = np.array([sku_map[s] for s in sku_seq_raw], dtype=np.int32).reshape(-1, 1)
    m_tr = split_seq == "train"
    m_va = split_seq == "val"
    m_te = split_seq == "test"
    print(f"windows train/val/test={m_tr.sum()}/{m_va.sum()}/{m_te.sum()}")

    for name, builder in [
        ("deepar_lite", build_deepar),
        ("temporal_transformer", build_transformer),
    ]:
        model = builder(args.lookback, n_skus, n_channels=n_channels)
        train_s = train_seq_three_term(
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
            "n_channels": n_channels,
            "overall": kpi_block(yseq[m_te], yhat, p),
            "strata": strata_report(
                yseq[m_te], yhat, p, sku_seq_raw[m_te], volume_map
            ),
        }

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    comparison = {"overall": [], "low": [], "mid": [], "high": []}
    for model, payload in results["models"].items():
        for band in comparison:
            block = (
                payload["overall"]
                if band == "overall"
                else payload["strata"][band]
            )
            comparison[band].append(
                {
                    "model": model,
                    "mae_rounded": block.get("mae_all_rounded"),
                    "mae_nonzero": block.get("mae_nonzero"),
                    "aucroc": block.get("aucroc"),
                    "aucpr": block.get("aucpr"),
                    "bias": block.get("bias"),
                }
            )
    for band in comparison:
        comparison[band] = sorted(
            comparison[band],
            key=lambda r: (r["mae_rounded"] is None, r["mae_rounded"] or 1e9),
        )
    results["comparison"] = comparison

    # prior calendar-only refs for context
    results["prior_calendar_only_sequence_reference"] = {
        "deepar_lite_mae_rounded": 1.571,
        "temporal_transformer_mae_rounded": 1.626,
        "note": "Earlier DeepAR/TST used [y,dow,month,doy] only — not same-feature",
    }

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("SAME-FEATURE COMPARISON")
    print("=" * 70)
    for band in ("overall", "low", "mid", "high"):
        print(f"\n[{band}]")
        for row in comparison[band]:
            print(
                f"  {row['model']:32s} mae={row['mae_rounded']:.3f} "
                f"nz={row['mae_nonzero']:.3f} auc={row.get('aucroc')} "
                f"ap={row.get('aucpr')} bias={row['bias']:.3f}"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
