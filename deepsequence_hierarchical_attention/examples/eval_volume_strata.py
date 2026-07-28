#!/usr/bin/env python3
"""
Volume-stratified intermittent KPI comparison (same 800-SKU / seed slice).

Train-only SKU volume = sum(Quantity) on train → terciles low / mid / high.
Models: DeepSequence three_term, LightGBM, DeepAR-lite, temporal transformer,
        residual TF with preserved DS gate (freeze path).

KPIs per stratum: MAE (rounded), nonzero MAE, AUROC, AUPRC, bias.
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
from deepsequence_hierarchical_attention.forecast_postprocess import round_forecast
from deepsequence_hierarchical_attention.losses import three_term_loss_config
from deepsequence_hierarchical_attention.residual_transformer import (
    build_residual_transformer,
    build_residual_windows,
    predict_residual_transformer,
    train_residual_transformer,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss

# Reuse sequence builders from baseline compare
from eval_baselines_compare import (
    build_deepar,
    build_sequence_tables,
    build_transformer,
    train_seq_model,
    predict_seq,
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
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_volume_strata.json"),
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


def train_volume_terciles(train_df: pd.DataFrame) -> dict:
    """SKU → {low, mid, high} from train sum(Quantity) terciles."""
    vol = train_df.groupby("id_var")["Quantity"].sum().astype(np.float64)
    # qcut can fail on ties; rank then cut
    ranks = vol.rank(method="first")
    labels = pd.qcut(ranks, 3, labels=["low", "mid", "high"])
    mapping = labels.to_dict()
    stats = {}
    for band in ("low", "mid", "high"):
        skus = [s for s, b in mapping.items() if b == band]
        stats[band] = {
            "n_skus": len(skus),
            "train_volume_sum": float(vol.loc[skus].sum()),
            "train_volume_mean_sku": float(vol.loc[skus].mean()),
            "train_volume_min": float(vol.loc[skus].min()),
            "train_volume_max": float(vol.loc[skus].max()),
            "train_zero_rate": float(
                (train_df.loc[train_df["id_var"].isin(skus), "Quantity"] == 0).mean()
            ),
        }
    return mapping, stats


def kpi_block(y, yhat, p=None):
    y = np.asarray(y, np.float64).reshape(-1)
    yhat = np.maximum(np.asarray(yhat, np.float64).reshape(-1), 0.0)
    yhat_r = round_forecast(yhat)
    nz = y > 0
    out = {
        "n_rows": int(len(y)),
        "n_nonzero": int(nz.sum()),
        "zero_rate": float((~nz).mean()) if len(y) else None,
        "mae_all": float(mean_absolute_error(y, yhat)) if len(y) else None,
        "mae_all_rounded": float(mean_absolute_error(y, yhat_r)) if len(y) else None,
        "mae_nonzero": float(mean_absolute_error(y[nz], yhat[nz])) if nz.any() else None,
        "mean_final": float(yhat.mean()) if len(y) else None,
        "mean_actual": float(y.mean()) if len(y) else None,
        "bias": float(yhat.mean() - y.mean()) if len(y) else None,
        "predict_zero_mae": float(mean_absolute_error(y, np.zeros_like(y))) if len(y) else None,
    }
    if p is not None and len(y) and len(np.unique((y > 0).astype(int))) == 2:
        p = np.asarray(p, np.float64).reshape(-1)
        yb = (y > 0).astype(np.float64)
        out["mean_p"] = float(p.mean())
        out["aucroc"] = float(roc_auc_score(yb, p))
        out["aucpr"] = float(average_precision_score(yb, p))
    return out


def strata_report(y, yhat, p, skus, volume_map):
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    skus = np.asarray(skus).reshape(-1)
    p = None if p is None else np.asarray(p).reshape(-1)
    bands = np.array([volume_map.get(s, "unk") for s in skus])
    out = {"overall": kpi_block(y, yhat, p)}
    # volume-weighted MAE: weight each row by that SKU's train volume share of total
    # (approx: use band mean volume as proxy per row belonging to band)
    for band in ("low", "mid", "high"):
        m = bands == band
        out[band] = kpi_block(
            y[m], yhat[m], None if p is None else p[m]
        )
        out[band]["n_skus_in_pred"] = int(len(set(skus[m].tolist())))
    # equal-SKU mean of per-SKU MAE (rounded) within overall
    per_sku = []
    for s in np.unique(skus):
        m = skus == s
        if m.sum() == 0:
            continue
        per_sku.append(
            {
                "sku": str(s),
                "band": volume_map.get(s, "unk"),
                "mae_rounded": float(
                    mean_absolute_error(y[m], round_forecast(yhat[m]))
                ),
                "mae_nonzero": float(
                    mean_absolute_error(y[m][y[m] > 0], yhat[m][y[m] > 0])
                )
                if (y[m] > 0).any()
                else None,
                "n_rows": int(m.sum()),
            }
        )
    by_band_sku = {}
    for band in ("low", "mid", "high"):
        rows = [r for r in per_sku if r["band"] == band]
        maes = [r["mae_rounded"] for r in rows]
        nzs = [r["mae_nonzero"] for r in rows if r["mae_nonzero"] is not None]
        by_band_sku[band] = {
            "equal_sku_mae_rounded_mean": float(np.mean(maes)) if maes else None,
            "equal_sku_mae_nonzero_mean": float(np.mean(nzs)) if nzs else None,
            "n_skus": len(rows),
        }
    out["equal_sku_means"] = by_band_sku
    return out


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
    print("Volume terciles (train sum Quantity):")
    for b, st in volume_stats.items():
        print(
            f"  {b}: n_skus={st['n_skus']} vol_mean={st['train_volume_mean_sku']:.1f} "
            f"range=[{st['train_volume_min']:.0f},{st['train_volume_max']:.0f}] "
            f"zero_rate={st['train_zero_rate']:.3f}"
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

    tr, va, te = split_components(X_train, cfg), split_components(X_val, cfg), split_components(X_test, cfg)
    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "seed": args.seed,
            "lookback": args.lookback,
            "zero_rate": zero_rate,
            "volume_definition": "train sum(Quantity) terciles",
            "volume_stats": volume_stats,
        },
        "models": {},
    }

    # ------------------------------------------------------------------
    # DeepSequence three_term
    # ------------------------------------------------------------------
    print("\n=== DeepSequence three_term ===")
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
    results["models"]["deepsequence_three_term"] = {
        "train_seconds": ds_s,
        "strata": strata_report(y_test, yhat_ds, p_ds, sku_test_raw, volume_map),
    }

    # ------------------------------------------------------------------
    # LightGBM
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
    p_lgb = np.clip(1.0 - np.exp(-yhat_lgb), 0, 1)
    results["models"]["lightgbm"] = {
        "train_seconds": lgb_s,
        "strata": strata_report(y_test, yhat_lgb, p_lgb, sku_test_raw, volume_map),
    }

    # ------------------------------------------------------------------
    # DeepAR + temporal transformer
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
        ("deepar_lite", build_deepar),
        ("temporal_transformer", build_transformer),
    ]:
        print(f"\n=== {name} ===")
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
            "strata": strata_report(
                yseq[m_te], yhat, p, sku_seq_raw[m_te], volume_map
            ),
        }

    # ------------------------------------------------------------------
    # Residual TF preserving DS gate (freeze path)
    # ------------------------------------------------------------------
    print("\n=== Residual TF (preserve DS gate) ===")
    # structural outputs on all splits for windows
    pred_tr = ds_model.predict([*tr, sku_train], batch_size=4096, verbose=0)
    pred_va = ds_model.predict([*va, sku_val], batch_size=4096, verbose=0)
    base_tr = np.asarray(pred_tr["base_forecast"]).reshape(-1).astype(np.float32)
    base_va = np.asarray(pred_va["base_forecast"]).reshape(-1).astype(np.float32)
    base_te = np.asarray(pred["base_forecast"]).reshape(-1).astype(np.float32)
    p_tr = np.asarray(pred_tr["non_zero_probability"]).reshape(-1).astype(np.float32)
    p_va = np.asarray(pred_va["non_zero_probability"]).reshape(-1).astype(np.float32)
    p_te = p_ds.astype(np.float32)

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
        alpha_bce=0.0,
    )
    tf_s = time.time() - t0
    yhat_r, p_r, _, _ = predict_residual_transformer(
        tf_model, Xr[r_te], ysr[r_te], sku_r[r_te]
    )
    results["models"]["ds_residual_preserve_gate"] = {
        "train_seconds": tf_s,
        "note": "freeze DS three_term → residual TF with TD p_ds multiply",
        "strata": strata_report(yr[r_te], yhat_r, p_r, skur[r_te], volume_map),
    }

    # ------------------------------------------------------------------
    # Compact comparison tables
    # ------------------------------------------------------------------
    comparison = {"overall": [], "low": [], "mid": [], "high": []}
    for model, payload in results["models"].items():
        strata = payload["strata"]
        for band in comparison:
            block = strata[band] if band != "overall" else strata["overall"]
            comparison[band].append(
                {
                    "model": model,
                    "mae_rounded": block.get("mae_all_rounded"),
                    "mae_nonzero": block.get("mae_nonzero"),
                    "aucroc": block.get("aucroc"),
                    "aucpr": block.get("aucpr"),
                    "bias": block.get("bias"),
                    "n_rows": block.get("n_rows"),
                    "zero_rate": block.get("zero_rate"),
                }
            )
        # attach equal-sku means onto high-level summary
    for band in ("low", "mid", "high"):
        comparison[band] = sorted(
            comparison[band],
            key=lambda r: (r["mae_rounded"] is None, r["mae_rounded"] or 1e9),
        )
    comparison["overall"] = sorted(
        comparison["overall"],
        key=lambda r: (r["mae_rounded"] is None, r["mae_rounded"] or 1e9),
    )
    # equal-sku summary
    equal_sku = {}
    for model, payload in results["models"].items():
        equal_sku[model] = payload["strata"]["equal_sku_means"]
    results["comparison"] = comparison
    results["equal_sku_mae_by_band"] = equal_sku

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("VOLUME STRATA COMPARISON (MAE rounded)")
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
