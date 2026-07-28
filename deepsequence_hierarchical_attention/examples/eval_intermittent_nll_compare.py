#!/usr/bin/env python3
"""
Same 800-SKU slice, but train ALL models with intermittent / hurdle losses
(not MAE three_term):

  - DeepSequence / DeepAR / TST: hurdle_poisson
        L = α·weighted_BCE(y>0, p) + PoissonNLL(y | λ)_{y>0}
  - LightGBM: Tweedie objective (classic intermittent-friendly GBDT loss)

Reports overall intermittent KPIs + train-volume terciles for comparison
against the prior three_term/MAE bake-off.
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
from deepsequence_hierarchical_attention.losses import hurdle_poisson_loss_config
from eval_baselines_compare import (
    build_deepar,
    build_sequence_tables,
    build_transformer,
    predict_seq,
)
from eval_volume_strata import (
    filter_aligned,
    kpi_block,
    split_components,
    strata_report,
    train_volume_terciles,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
    )
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha_bce", type=float, default=1.0)
    p.add_argument("--w_mag", type=float, default=1.0)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_intermittent_nll.json"),
    )
    return p.parse_args()


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

    tr, va, te = (
        split_components(X_train, cfg),
        split_components(X_val, cfg),
        split_components(X_test, cfg),
    )

    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "seed": args.seed,
            "loss": "hurdle_poisson (NN) / tweedie (LightGBM)",
            "alpha_bce": args.alpha_bce,
            "w_mag": args.w_mag,
            "zero_rate": zero_rate,
            "volume_stats": volume_stats,
            "three_term_reference": {
                "deepar_lite_mae_rounded": 1.571,
                "temporal_transformer_mae_rounded": 1.626,
                "lightgbm_l1_mae_rounded": 1.845,
                "deepsequence_three_term_mae_rounded": 2.083,
            },
        },
        "models": {},
    }

    loss_cfg = hurdle_poisson_loss_config(
        zero_rate, alpha_bce=args.alpha_bce, w_mag=args.w_mag
    )

    # ------------------------------------------------------------------
    # DeepSequence
    # ------------------------------------------------------------------
    ds = build_hierarchical_model_lightweight(
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
    _ = ds(
        [*(np.zeros((1, x.shape[1]), np.float32) for x in tr), np.zeros((1, 1), np.int32)],
        training=False,
    )
    losses = {k: v for k, v in loss_cfg["losses"].items()}
    weights = {k: loss_cfg["weights"][k] for k in losses}
    ds.compile(optimizer=tf.keras.optimizers.Adam(0.0025), loss=losses, loss_weights=weights)
    ytr = {
        "final_forecast": y_train.reshape(-1, 1),
        "base_forecast": y_train.reshape(-1, 1),
        "non_zero_probability": (y_train > 0).astype(np.float32).reshape(-1, 1),
    }
    yva = {
        "final_forecast": y_val.reshape(-1, 1),
        "base_forecast": y_val.reshape(-1, 1),
        "non_zero_probability": (y_val > 0).astype(np.float32).reshape(-1, 1),
    }
    print("\n=== DeepSequence (hurdle_poisson) ===")
    t0 = time.time()
    ds.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
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
    ds_s = time.time() - t0
    pred = ds.predict([*te, sku_test], batch_size=4096, verbose=0)
    yhat_ds = np.asarray(pred["final_forecast"]).reshape(-1)
    p_ds = np.asarray(pred["non_zero_probability"]).reshape(-1)
    results["models"]["deepsequence_hurdle_poisson"] = {
        "train_seconds": ds_s,
        "overall": kpi_block(y_test, yhat_ds, p_ds),
        "strata": strata_report(y_test, yhat_ds, p_ds, sku_test_raw, volume_map),
    }

    # ------------------------------------------------------------------
    # LightGBM Tweedie
    # ------------------------------------------------------------------
    print("\n=== LightGBM (tweedie) ===")
    Xlgb_tr = np.concatenate([X_train, sku_train.astype(np.float32)], axis=1)
    Xlgb_va = np.concatenate([X_val, sku_val.astype(np.float32)], axis=1)
    Xlgb_te = np.concatenate([X_test, sku_test.astype(np.float32)], axis=1)
    lgb_model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
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
    results["models"]["lightgbm_tweedie"] = {
        "train_seconds": lgb_s,
        "overall": kpi_block(y_test, yhat_lgb, p_lgb),
        "strata": strata_report(y_test, yhat_lgb, p_lgb, sku_test_raw, volume_map),
    }

    # ------------------------------------------------------------------
    # DeepAR + TST
    # ------------------------------------------------------------------
    print("\nBuilding sequence windows...")
    Xseq, yseq, sku_seq_raw, split_seq = build_sequence_tables(
        train_df, val_df, test_df, args.lookback
    )
    sku_seq = np.array([sku_map[s] for s in sku_seq_raw], dtype=np.int32).reshape(-1, 1)
    m_tr = split_seq == "train"
    m_va = split_seq == "val"
    m_te = split_seq == "test"

    for name, builder in [
        ("deepar_lite_hurdle_poisson", build_deepar),
        ("temporal_transformer_hurdle_poisson", build_transformer),
    ]:
        model = builder(args.lookback, n_skus)
        losses = {k: v for k, v in loss_cfg["losses"].items()}
        weights = {k: loss_cfg["weights"][k] for k in losses}
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0025),
            loss=losses,
            loss_weights=weights,
        )
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
        print(f"\n=== {name} ===")
        t0 = time.time()
        model.fit(
            [Xseq[m_tr], sku_seq[m_tr]],
            ytr_d,
            validation_data=([Xseq[m_va], sku_seq[m_va]], yva_d),
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
        train_s = time.time() - t0
        yhat, p = predict_seq(model, Xseq[m_te], sku_seq[m_te])
        results["models"][name] = {
            "train_seconds": train_s,
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
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 70)
    print("INTERMITTENT NLL / TWEEDIE COMPARISON")
    print("=" * 70)
    for band in ("overall", "low", "mid", "high"):
        print(f"\n[{band}]")
        for row in comparison[band]:
            print(
                f"  {row['model']:40s} mae={row['mae_rounded']:.3f} "
                f"nz={row['mae_nonzero']:.3f} auc={row.get('aucroc')} "
                f"ap={row.get('aucpr')}"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
