#!/usr/bin/env python3
"""
Requested bake-off (same 800-SKU / seed=42 slice):

  1. LightGBM + Tweedie
  2. DeepAR-lite + Tweedie deviance on final_forecast
  3. Temporal Transformer + Tweedie deviance on final_forecast
  4. DeepSequence three_term + residual TF (preserve DS gate), three_term loss

Also reports train-volume tercile KPIs.
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
from deepsequence_hierarchical_attention.losses import (
    three_term_loss_config,
    tweedie_loss_config,
)
from deepsequence_hierarchical_attention.residual_transformer import (
    build_residual_transformer,
    build_residual_windows,
    predict_residual_transformer,
    train_residual_transformer,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
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
    p.add_argument("--epochs_tf", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tweedie_power", type=float, default=1.5)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_tweedie_vs_ds_residual.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
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
            "tweedie_power": args.tweedie_power,
            "zero_rate": zero_rate,
            "volume_stats": volume_stats,
            "models": [
                "lightgbm_tweedie",
                "deepar_lite_tweedie",
                "temporal_transformer_tweedie",
                "ds_three_term_residual_preserve_gate",
            ],
        },
        "models": {},
    }

    # ------------------------------------------------------------------
    # 1) LightGBM Tweedie
    # ------------------------------------------------------------------
    print("\n=== LightGBM (tweedie) ===")
    Xlgb_tr = np.concatenate([X_train, sku_train.astype(np.float32)], axis=1)
    Xlgb_va = np.concatenate([X_val, sku_val.astype(np.float32)], axis=1)
    Xlgb_te = np.concatenate([X_test, sku_test.astype(np.float32)], axis=1)
    lgb_model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=args.tweedie_power,
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
        "loss": "tweedie",
        "train_seconds": lgb_s,
        "overall": kpi_block(y_test, yhat_lgb, p_lgb),
        "strata": strata_report(y_test, yhat_lgb, p_lgb, sku_test_raw, volume_map),
    }

    # ------------------------------------------------------------------
    # 2-3) DeepAR + TST with Tweedie on final_forecast
    # ------------------------------------------------------------------
    print("\nBuilding sequence windows...")
    Xseq, yseq, sku_seq_raw, split_seq = build_sequence_tables(
        train_df, val_df, test_df, args.lookback
    )
    sku_seq = np.array([sku_map[s] for s in sku_seq_raw], dtype=np.int32).reshape(-1, 1)
    m_tr = split_seq == "train"
    m_va = split_seq == "val"
    m_te = split_seq == "test"

    tw_cfg = tweedie_loss_config(power=args.tweedie_power, bce_weight=0.0)
    for name, builder in [
        ("deepar_lite_tweedie", build_deepar),
        ("temporal_transformer_tweedie", build_transformer),
    ]:
        print(f"\n=== {name} ===")
        model = builder(args.lookback, n_skus)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0025),
            loss=tw_cfg["losses"],
            loss_weights=tw_cfg["weights"],
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
            "loss": "tweedie_deviance",
            "train_seconds": train_s,
            "overall": kpi_block(yseq[m_te], yhat, p),
            "strata": strata_report(
                yseq[m_te], yhat, p, sku_seq_raw[m_te], volume_map
            ),
        }

    # ------------------------------------------------------------------
    # 4) DS three_term + residual preserve-gate (three_term)
    # ------------------------------------------------------------------
    print("\n=== DeepSequence three_term + residual (preserve gate) ===")
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

    pred_tr = ds_model.predict([*tr, sku_train], batch_size=4096, verbose=0)
    pred_va = ds_model.predict([*va, sku_val], batch_size=4096, verbose=0)
    pred_te = ds_model.predict([*te, sku_test], batch_size=4096, verbose=0)
    base_tr = np.asarray(pred_tr["base_forecast"]).reshape(-1).astype(np.float32)
    base_va = np.asarray(pred_va["base_forecast"]).reshape(-1).astype(np.float32)
    base_te = np.asarray(pred_te["base_forecast"]).reshape(-1).astype(np.float32)
    p_tr = np.asarray(pred_tr["non_zero_probability"]).reshape(-1).astype(np.float32)
    p_va = np.asarray(pred_va["non_zero_probability"]).reshape(-1).astype(np.float32)
    p_te = np.asarray(pred_te["non_zero_probability"]).reshape(-1).astype(np.float32)

    def pack(df, ystruct, p_arr, split):
        return pd.DataFrame(
            {
                "id_var": df["id_var"].to_numpy(),
                "ds": pd.to_datetime(df["ds"]),
                "y": df["Quantity"].to_numpy(np.float32),
                "y_struct": ystruct.astype(np.float32),
                "p_ds": p_arr.astype(np.float32),
                "split": split,
            }
        )

    panel = pd.concat(
        [
            pack(train_df, base_tr, p_tr, "train"),
            pack(val_df, base_va, p_va, "val"),
            pack(test_df, base_te, p_te, "test"),
        ],
        ignore_index=True,
    )
    Xr, yr, ysr, pr, skur, splitsr = build_residual_windows(panel, lookback=args.lookback)
    sku_r = np.array([sku_map[s] for s in skur], dtype=np.int32).reshape(-1, 1)
    r_tr, r_va, r_te = splitsr == "train", splitsr == "val", splitsr == "test"

    tf_model = build_residual_transformer(
        args.lookback,
        Xr.shape[-1],
        n_skus,
        d_model=32,
        n_heads=4,
        n_blocks=1,
        preserve_ds_gate=True,
    )
    t0 = time.time()
    train_residual_transformer(
        tf_model,
        Xr[r_tr],
        yr[r_tr],
        ysr[r_tr],
        sku_r[r_tr],
        Xr[r_va],
        yr[r_va],
        ysr[r_va],
        sku_r[r_va],
        zero_rate,
        epochs=args.epochs_tf,
        batch_size=min(512, args.batch_size),
        alpha_bce=0.0,  # gate fixed from DS three_term
    )
    tf_s = time.time() - t0
    yhat_r, p_r, _, _ = predict_residual_transformer(
        tf_model, Xr[r_te], ysr[r_te], sku_r[r_te]
    )
    results["models"]["ds_three_term_residual_preserve_gate"] = {
        "loss": "ds=three_term; residual=gated MAE + mag MAE (p_ds preserved)",
        "train_seconds": ds_s + tf_s,
        "ds_train_seconds": ds_s,
        "residual_train_seconds": tf_s,
        "overall": kpi_block(yr[r_te], yhat_r, p_r),
        "strata": strata_report(yr[r_te], yhat_r, p_r, skur[r_te], volume_map),
        # also report plain DS three_term final for reference
        "ds_three_term_only": strata_report(
            y_test,
            np.asarray(pred_te["final_forecast"]).reshape(-1),
            p_te,
            sku_test_raw,
            volume_map,
        ),
    }

    # ------------------------------------------------------------------
    # Comparison tables
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
                    "loss": payload.get("loss"),
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
    print("TWEEDIE (LGBM/DeepAR/TST) vs DS three_term + residual")
    print("=" * 70)
    for band in ("overall", "low", "mid", "high"):
        print(f"\n[{band}]")
        for row in comparison[band]:
            print(
                f"  {row['model']:42s} mae={row['mae_rounded']:.3f} "
                f"nz={row['mae_nonzero']:.3f} auc={row.get('aucroc')} "
                f"ap={row.get('aucpr')} bias={row['bias']:.3f}"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
