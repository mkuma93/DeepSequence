#!/usr/bin/env python3
"""
Compare residual-transformer stacks on DeepSequence structural base.

Contract: holiday + regressor absorbed into y_struct; DS gate p_ds preserved.
Residual channels: [y_struct, y, residual, p_ds]
  - y & residual masked on the predict step; p_ds never masked
  - TimeDistributed multiply by p_ds at each lookback step
  - final = p_ds_t * relu(y_struct_t + delta)  (no new TF gate)

  A) Freeze DS → train residual transformer
  B) End-to-end co-adaptation (refresh struct → fit transformer)

Final: yhat = gate * relu(y_struct_t + delta_t)
Loss: three-term style (weighted gated MAE + nonzero mag MAE + light BCE)
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
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.forecast_postprocess import round_forecast
from deepsequence_hierarchical_attention.losses import three_term_loss_config
from deepsequence_hierarchical_attention.residual_transformer import (
    build_residual_transformer,
    build_residual_windows,
    predict_residual_transformer,
    train_residual_transformer,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
    )
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--lookback", type=int, default=14)
    p.add_argument("--epochs_struct", type=int, default=8)
    p.add_argument("--epochs_tf", type=int, default=10)
    p.add_argument("--epochs_e2e", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument(
        "--n_blocks",
        type=int,
        default=1,
        help="Number of causal transformer encoder blocks",
    )
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_residual_transformer.json"),
    )
    return p.parse_args()


def filter_aligned(df, holidays, sku_set):
    mask = df["id_var"].isin(sku_set).to_numpy()
    return df.loc[mask].reset_index(drop=True), holidays.loc[mask].reset_index(drop=True)


def structural_parts(X, cfg):
    """Trend / seasonal / holiday / regressor for the structural DS base."""
    return (
        X[:, cfg.trend_indices].astype(np.float32),
        X[:, cfg.seasonal_indices].astype(np.float32),
        X[:, cfg.holiday_indices].astype(np.float32),
        X[:, cfg.regressor_indices].astype(np.float32),
    )


def eval_pack(y, yhat, p=None, zero_rate=0.9):
    y = np.asarray(y, np.float64).reshape(-1)
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    yhat_r = round_forecast(yhat)
    nz = y > 0
    yb = (y > 0).astype(np.float64)
    out = {
        "mae_all": float(mean_absolute_error(y, yhat)),
        "mae_all_rounded": float(mean_absolute_error(y, yhat_r)),
        "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else None,
        "mae_nonzero_rounded": float(mean_absolute_error(y[nz], yhat_r[nz]))
        if nz.any()
        else None,
        "mean_final": float(yhat.mean()),
        "mean_actual": float(y.mean()),
    }
    if p is not None:
        p = np.asarray(p, np.float64).reshape(-1)
        out["mean_p"] = float(p.mean())
        out["aucroc"] = float(roc_auc_score(yb, p))
        out["aucpr"] = float(average_precision_score(yb, p))
    return out


# ---------------------------------------------------------------------------
# Structural DeepSequence (holiday + regressor in base; residual TF corrects δ)
# ---------------------------------------------------------------------------

def build_structural_ds(
    n_trend, n_seasonal, n_holiday, n_lag, n_skus, hidden=48, sku_dim=4
):
    """DS base that already accounts for holiday + regressor before residual TF."""
    return build_hierarchical_model_lightweight(
        n_temporal_features=n_trend,
        n_fourier_features=n_seasonal,
        n_holiday_features=n_holiday,
        n_lag_features=n_lag,
        n_skus=n_skus,
        hidden_dim=hidden,
        sku_embedding_dim=sku_dim,
        dropout_rate=0.2,
        use_cross_layers=True,
        use_intermittent=True,
        enable_trend=True,
        enable_seasonal=True,
        enable_holiday=True,
        enable_regressor=True,
        n_changepoints=15,
    )


def predict_struct(model, trend, seasonal, holiday, lags, sku):
    out = model.predict(
        [trend, seasonal, holiday, lags, sku], batch_size=4096, verbose=0
    )
    # Prefer base_forecast (pre-gate structural); fall back to final
    if isinstance(out, dict):
        base = np.asarray(out.get("base_forecast", out["final_forecast"])).reshape(-1)
        final = np.asarray(out["final_forecast"]).reshape(-1)
        p = np.asarray(out.get("non_zero_probability", np.ones_like(final))).reshape(-1)
    else:
        base = final = np.asarray(out).reshape(-1)
        p = np.ones_like(final)
    return base.astype(np.float32), final.astype(np.float32), p.astype(np.float32)


# ---------------------------------------------------------------------------
# Sequence builder (SKU-isolated windows via package module)
# ---------------------------------------------------------------------------

def build_windows_from_row_struct(
    train_df, val_df, test_df, struct_tr, struct_va, struct_te, p_tr, p_va, p_te, lookback
):
    """
    Residual channels + preserved DS gate p_ds (TimeDistributed multiply in TF head).
    """
    def pack(df, ystruct, p_ds):
        y = df["Quantity"].to_numpy(np.float32)
        return pd.DataFrame(
            {
                "id_var": df["id_var"].to_numpy(),
                "ds": pd.to_datetime(df["ds"]),
                "y": y,
                "y_struct": ystruct.astype(np.float32),
                "p_ds": np.asarray(p_ds, np.float32),
            }
        )

    parts = [
        pack(train_df, struct_tr, p_tr).assign(split="train"),
        pack(val_df, struct_va, p_va).assign(split="val"),
        pack(test_df, struct_te, p_te).assign(split="test"),
    ]
    all_df = pd.concat(parts, ignore_index=True)
    X, yt, yst, pt, skus, splits = build_residual_windows(all_df, lookback=lookback)
    return X, yt, yst, pt, skus, splits


def train_transformer(model, Xtr, ytr, ystr, skutr, Xva, yva, ysva, skuva, zero_rate, epochs, bs, lr=0.002):
    t0 = time.time()
    wrapped = train_residual_transformer(
        model,
        Xtr,
        ytr,
        ystr,
        skutr,
        Xva,
        yva,
        ysva,
        skuva,
        zero_rate,
        epochs=epochs,
        batch_size=bs,
        learning_rate=lr,
    )
    return time.time() - t0, wrapped


def predict_transformer(model, X, ystruct, sku):
    yhat, p, base, _ = predict_residual_transformer(model, X, ystruct, sku)
    return yhat, p, base


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

    cats = pd.Categorical(train_df["id_var"])
    sku_map = {k: i for i, k in enumerate(cats.categories)}
    n_skus = len(sku_map)

    def enc(df):
        return df["id_var"].map(sku_map).astype(np.int32).to_numpy().reshape(-1, 1)

    cfg = load_feature_config()
    print(
        "Building causal features "
        "(structural DS uses trend/seasonal/holiday + regressor; "
        "residual TF gets residual channels only)..."
    )
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

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val, sku_test = enc(train_df), enc(val_df), enc(test_df)
    zero_rate = float((y_train == 0).mean())
    print(f"n_skus={n_skus} zero_rate={zero_rate:.3f}")

    tr = structural_parts(X_train, cfg)
    va = structural_parts(X_val, cfg)
    te = structural_parts(X_test, cfg)
    n_lag = tr[3].shape[1]

    # ------------------------------------------------------------------
    # Stage 0: train structural DS (holiday + regressor already in base)
    # ------------------------------------------------------------------
    print("\n=== Train structural DeepSequence (holiday + regressor in base) ===")
    ds = build_structural_ds(
        len(cfg.trend_indices),
        len(cfg.seasonal_indices),
        len(cfg.holiday_indices),
        n_lag,
        n_skus,
        hidden=args.hidden,
    )
    _ = ds(
        [
            np.zeros((1, tr[0].shape[1]), np.float32),
            np.zeros((1, tr[1].shape[1]), np.float32),
            np.zeros((1, tr[2].shape[1]), np.float32),
            np.zeros((1, n_lag), np.float32),
            np.zeros((1, 1), np.int32),
        ],
        training=False,
    )
    loss_cfg = three_term_loss_config(zero_rate, alpha_bce=0.2)
    # Structural DS may not expose base in compile the same way — use AdaptiveWeighted-like via compile on available outputs
    compile_losses = {
        k: v for k, v in loss_cfg["losses"].items() if k in ("final_forecast", "non_zero_probability", "base_forecast")
    }
    compile_weights = {k: loss_cfg["weights"][k] for k in compile_losses}
    ds.compile(
        optimizer=tf.keras.optimizers.Adam(0.0025),
        loss=compile_losses,
        loss_weights=compile_weights,
    )
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
    t0 = time.time()
    ds.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=args.epochs_struct,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            )
        ],
        verbose=2,
    )
    struct_train_s = time.time() - t0

    struct_tr, _, p_tr = predict_struct(ds, *tr, sku_train)
    struct_va, _, p_va = predict_struct(ds, *va, sku_val)
    struct_te, _, p_struct_te = predict_struct(ds, *te, sku_test)
    p_te = p_struct_te

    results = {
        "config": {
            "lookback": args.lookback,
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "zero_rate": zero_rate,
            "sequence_channels": [
                "y_struct",
                "y_masked_at_t",
                "residual_masked_at_t",
                "p_ds",
            ],
            "structural_includes": ["trend", "seasonal", "holiday", "regressor"],
            "preserve_ds_gate": True,
            "note": "TD multiply by p_ds each step; final uses DS gate (no new TF sigmoid)",
            "struct_train_seconds": struct_train_s,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_blocks": args.n_blocks,
        },
        "models": {
            "structural_ds_only": {
                "test": eval_pack(y_test, struct_te, p_struct_te, zero_rate),
                "note": "base_forecast from DS with holiday+regressor (pre-gate structural)",
            }
        },
    }

    # Build windows using frozen structural predictions first (for freeze path)
    print("\nBuilding sequence windows...")
    Xseq, yseq, ystruct_seq, pseq, sku_raw, split_seq = build_windows_from_row_struct(
        train_df,
        val_df,
        test_df,
        struct_tr,
        struct_va,
        struct_te,
        p_tr,
        p_va,
        p_te,
        args.lookback,
    )
    sku_seq = np.array([sku_map[s] for s in sku_raw], dtype=np.int32).reshape(-1, 1)
    m_tr = split_seq == "train"
    m_va = split_seq == "val"
    m_te = split_seq == "test"
    print(f"windows train/val/test={m_tr.sum()}/{m_va.sum()}/{m_te.sum()}")

    # ------------------------------------------------------------------
    # A) Freeze DS, train transformer
    # ------------------------------------------------------------------
    print(
        f"\n=== A) Freeze DS → residual TF (preserve p_ds via TD multiply) "
        f"(blocks={args.n_blocks}, d_model={args.d_model}, heads={args.n_heads}) ==="
    )
    tf_a = build_residual_transformer(
        args.lookback,
        Xseq.shape[-1],
        n_skus,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
    )
    # freeze is implicit: y_struct is a fixed input feature, DS not in graph
    train_s_a, wrapped_a = train_transformer(
        tf_a,
        Xseq[m_tr],
        yseq[m_tr],
        ystruct_seq[m_tr],
        sku_seq[m_tr],
        Xseq[m_va],
        yseq[m_va],
        ystruct_seq[m_va],
        sku_seq[m_va],
        zero_rate,
        args.epochs_tf,
        args.batch_size,
    )
    yhat_a, p_a, _ = predict_transformer(
        tf_a, Xseq[m_te], ystruct_seq[m_te], sku_seq[m_te]
    )
    results["models"]["freeze_then_transformer"] = {
        "train_seconds": train_s_a,
        "test": eval_pack(yseq[m_te], yhat_a, p_a, zero_rate),
        "n_test_windows": int(m_te.sum()),
    }

    # ------------------------------------------------------------------
    # B) End-to-end lite: re-compute y_struct inside training via online DS?
    #    True E2E through full DS per timestep is expensive. Practical E2E:
    #    jointly train transformer while ALSO fine-tuning a small adapter on
    #    y_struct channel — OR unfreeze by rebuilding windows each epoch.
    #
    #    Here we do a practical E2E:
    #      1) continue training DS a few epochs with lower LR (unfreeze)
    #      2) refresh structural predictions / windows
    #      3) train transformer from scratch on refreshed sequences
    #      4) optional: one more refresh + short transformer fine-tune
    #    This co-adapts DS to residual head without per-step DS graph.
    # ------------------------------------------------------------------
    print("\n=== B) End-to-end co-adaptation (refresh struct → fit transformer) ===")
    # Fine-tune structural DS lightly
    ds.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=compile_losses,
        loss_weights=compile_weights,
    )
    t0 = time.time()
    ds.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=max(3, args.epochs_e2e // 3),
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
            )
        ],
        verbose=2,
    )
    # refresh structures
    struct_tr2, _, p_tr2 = predict_struct(ds, *tr, sku_train)
    struct_va2, _, p_va2 = predict_struct(ds, *va, sku_val)
    struct_te2, _, p_struct_te2 = predict_struct(ds, *te, sku_test)
    Xseq2, yseq2, ystruct_seq2, pseq2, sku_raw2, split_seq2 = build_windows_from_row_struct(
        train_df,
        val_df,
        test_df,
        struct_tr2,
        struct_va2,
        struct_te2,
        p_tr2,
        p_va2,
        p_struct_te2,
        args.lookback,
    )
    sku_seq2 = np.array([sku_map[s] for s in sku_raw2], dtype=np.int32).reshape(-1, 1)
    m_tr2 = split_seq2 == "train"
    m_va2 = split_seq2 == "val"
    m_te2 = split_seq2 == "test"

    tf_b = build_residual_transformer(
        args.lookback,
        Xseq2.shape[-1],
        n_skus,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
    )
    train_s_b, wrapped_b = train_transformer(
        tf_b,
        Xseq2[m_tr2],
        yseq2[m_tr2],
        ystruct_seq2[m_tr2],
        sku_seq2[m_tr2],
        Xseq2[m_va2],
        yseq2[m_va2],
        ystruct_seq2[m_va2],
        sku_seq2[m_va2],
        zero_rate,
        args.epochs_tf,
        args.batch_size,
    )
    # second refresh + short fine-tune
    struct_tr3, _, p_tr3 = predict_struct(ds, *tr, sku_train)
    struct_va3, _, p_va3 = predict_struct(ds, *va, sku_val)
    struct_te3, _, p_te3 = predict_struct(ds, *te, sku_test)
    Xseq3, yseq3, ystruct_seq3, pseq3, sku_raw3, split_seq3 = build_windows_from_row_struct(
        train_df,
        val_df,
        test_df,
        struct_tr3,
        struct_va3,
        struct_te3,
        p_tr3,
        p_va3,
        p_te3,
        args.lookback,
    )
    sku_seq3 = np.array([sku_map[s] for s in sku_raw3], dtype=np.int32).reshape(-1, 1)
    m_tr3 = split_seq3 == "train"
    m_va3 = split_seq3 == "val"
    m_te3 = split_seq3 == "test"
    train_s_b2, _ = train_transformer(
        tf_b,
        Xseq3[m_tr3],
        yseq3[m_tr3],
        ystruct_seq3[m_tr3],
        sku_seq3[m_tr3],
        Xseq3[m_va3],
        yseq3[m_va3],
        ystruct_seq3[m_va3],
        sku_seq3[m_va3],
        zero_rate,
        max(3, args.epochs_e2e // 4),
        args.batch_size,
        lr=0.001,
    )
    e2e_s = time.time() - t0
    yhat_b, p_b, _ = predict_transformer(
        tf_b, Xseq3[m_te3], ystruct_seq3[m_te3], sku_seq3[m_te3]
    )
    results["models"]["e2e_coadapt_transformer"] = {
        "train_seconds": e2e_s,
        "transformer_fit_seconds": train_s_b + train_s_b2,
        "test": eval_pack(yseq3[m_te3], yhat_b, p_b, zero_rate),
        "n_test_windows": int(m_te3.sum()),
        "note": "DS fine-tune + refresh windows + transformer (practical E2E co-adaptation)",
    }

    # also report refreshed structural alone
    results["models"]["structural_ds_after_e2e_finetune"] = {
        "test": eval_pack(y_test, struct_te3, p_struct_te2, zero_rate),
    }

    comparison = []
    for name, payload in results["models"].items():
        t = payload["test"]
        comparison.append(
            {
                "model": name,
                "test_mae": t["mae_all"],
                "test_mae_rounded": t["mae_all_rounded"],
                "test_mae_nonzero": t["mae_nonzero"],
                "aucroc": t.get("aucroc"),
                "mean_final": t["mean_final"],
            }
        )
    comparison = sorted(comparison, key=lambda r: r["test_mae_rounded"])
    results["comparison"] = comparison
    results["baselines_same_slice_reference"] = {
        "deepar_lite_mae_rounded": 1.571,
        "transformer_mae_rounded": 1.626,
        "lightgbm_mae_rounded": 1.845,
        "deepsequence_three_term_mae_rounded": 2.083,
        "deepsequence_gated_tau0.35_mae": 2.011,
        "predict_zero": float(mean_absolute_error(y_test, np.zeros_like(y_test))),
    }

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("RESIDUAL TRANSFORMER COMPARISON")
    print("=" * 70)
    print(json.dumps(comparison, indent=2))
    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
