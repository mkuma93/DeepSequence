#!/usr/bin/env python3
"""
Corrected A/B: drop engineered lags, but restore short-term memory via residual TF.

  1. DS three_term (gated) + residual TF (preserve p_ds)
  2. DS ungated Tweedie + residual TF (p_ds≡1, Tweedie on final; no sigmoid)

Same no-lag feature contract (keep intermittent). Windows cover lag_1/2/7.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.losses import tweedie_deviance_loss
from deepsequence_hierarchical_attention.residual_transformer import (
    build_residual_transformer,
    build_residual_windows,
    predict_residual_transformer,
    train_residual_transformer,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from eval_volume_strata import (
    filter_aligned,
    kpi_block,
    strata_report,
    train_volume_terciles,
)
from eval_nolag_residual_compare import (
    component_indices_nolag,
    drop_lag_columns,
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
        default=str(ROOT / "eval_results_nolag_residual_tweedie_ab.json"),
    )
    return p.parse_args()


def p_from_mean(yhat):
    return np.clip(1.0 - np.exp(-np.maximum(yhat, 0.0)), 0.0, 1.0)


def pack_panel(df, ystruct, p_arr, split):
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


def train_residual_tweedie(
    model, Xtr, ytr, ystr, skutr, Xva, yva, ysva, skuva, power, epochs, batch_size
):
    """Ungated residual: train softplus/relu mean with Tweedie (p_ds≡1 in windows)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.002),
        loss={
            "final_forecast": tweedie_deviance_loss(power),
            "base_forecast": tweedie_deviance_loss(power),
            "non_zero_probability": "mse",
            "delta": "mse",
        },
        loss_weights={
            "final_forecast": 1.0,
            "base_forecast": 0.0,
            "non_zero_probability": 0.0,
            "delta": 0.0,
        },
    )
    t0 = time.time()
    model.fit(
        [Xtr, skutr, ystr.reshape(-1, 1)],
        {
            "final_forecast": ytr.reshape(-1, 1),
            "base_forecast": ytr.reshape(-1, 1),
            "non_zero_probability": np.ones((len(ytr), 1), np.float32),
            "delta": np.zeros((len(ytr), 1), np.float32),
        },
        validation_data=(
            [Xva, skuva, ysva.reshape(-1, 1)],
            {
                "final_forecast": yva.reshape(-1, 1),
                "base_forecast": yva.reshape(-1, 1),
                "non_zero_probability": np.ones((len(yva), 1), np.float32),
                "delta": np.zeros((len(yva), 1), np.float32),
            },
        ),
        epochs=epochs,
        batch_size=batch_size,
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

    cfg = load_feature_config()
    print("No-lag features (intermittent kept); residual TF covers short-term memory...")
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, states = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
    Xtr_df, dropped = drop_lag_columns(Xtr_df, cfg)
    Xva_df, _ = drop_lag_columns(Xva_df, cfg)
    Xte_df, _ = drop_lag_columns(Xte_df, cfg)
    feature_names = list(Xtr_df.columns)
    print(f"dropped={dropped}")

    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    if "time_index" in feature_names:
        t_idx = feature_names.index("time_index")
        tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
        span = max(tmax - tmin, 1.0)
        for X in (X_train, X_val, X_test):
            X[:, t_idx] = (X[:, t_idx] - tmin) / span

    trend_i, seas_i, hol_i, inter_i = component_indices_nolag(feature_names, cfg)

    def parts(X):
        return (
            X[:, trend_i].astype(np.float32),
            X[:, seas_i].astype(np.float32),
            X[:, hol_i].astype(np.float32),
            X[:, inter_i].astype(np.float32),
        )

    tr, va, te = parts(X_train), parts(X_val), parts(X_test)
    n_inter = len(inter_i)
    pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))

    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "seed": args.seed,
            "lookback": args.lookback,
            "zero_rate": zero_rate,
            "tweedie_power": args.tweedie_power,
            "dropped_lags": dropped,
            "note": "Lags dropped; residual lookback restores short-term memory for both arms.",
            "volume_stats": volume_stats,
        },
        "models": {},
    }

    def fit_ds(use_intermittent: bool, label: str):
        print(f"\n=== {label} ===")
        base = build_hierarchical_model_lightweight(
            n_temporal_features=len(trend_i),
            n_fourier_features=len(seas_i),
            n_holiday_features=len(hol_i),
            n_lag_features=n_inter,
            n_skus=n_skus,
            hidden_dim=48,
            sku_embedding_dim=4,
            dropout_rate=0.23,
            use_cross_layers=True,
            use_intermittent=use_intermittent,
            enable_regressor=True,
            n_changepoints=15,
        )
        _ = base(
            [
                *(np.zeros((1, x.shape[1]), np.float32) for x in tr),
                np.zeros((1, 1), np.int32),
            ],
            training=False,
        )
        t0 = time.time()
        if use_intermittent:
            model = AdaptiveWeightedModel(
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
            model.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
            ytr = {
                "final_forecast": y_train.reshape(-1, 1),
                "base_forecast": y_train.reshape(-1, 1),
            }
            yva = {
                "final_forecast": y_val.reshape(-1, 1),
                "base_forecast": y_val.reshape(-1, 1),
            }
            model.fit(
                [*tr, sku_train],
                ytr,
                validation_data=([*va, sku_val], yva),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=3,
                        restore_best_weights=True,
                        verbose=1,
                    )
                ],
                verbose=2,
            )
            pred_tr = model.predict([*tr, sku_train], batch_size=4096, verbose=0)
            pred_va = model.predict([*va, sku_val], batch_size=4096, verbose=0)
            pred_te = model.predict([*te, sku_test], batch_size=4096, verbose=0)
            # structural channel for residual = ungated magnitude path
            ystruct_tr = np.asarray(pred_tr["base_forecast"]).reshape(-1).astype(np.float32)
            ystruct_va = np.asarray(pred_va["base_forecast"]).reshape(-1).astype(np.float32)
            ystruct_te = np.asarray(pred_te["base_forecast"]).reshape(-1).astype(np.float32)
            p_tr = np.asarray(pred_tr["non_zero_probability"]).reshape(-1).astype(np.float32)
            p_va = np.asarray(pred_va["non_zero_probability"]).reshape(-1).astype(np.float32)
            p_te = np.asarray(pred_te["non_zero_probability"]).reshape(-1).astype(np.float32)
            yhat_struct = np.asarray(pred_te["final_forecast"]).reshape(-1)
        else:
            base.compile(
                optimizer=tf.keras.optimizers.Adam(0.0025),
                loss={"final_forecast": tweedie_deviance_loss(args.tweedie_power)},
            )
            base.fit(
                [*tr, sku_train],
                {"final_forecast": y_train.reshape(-1, 1)},
                validation_data=(
                    [*va, sku_val],
                    {"final_forecast": y_val.reshape(-1, 1)},
                ),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=3,
                        restore_best_weights=True,
                        verbose=1,
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=2,
                        min_lr=1e-5,
                        verbose=1,
                    ),
                ],
                verbose=2,
            )
            model = base
            pred_tr = model.predict([*tr, sku_train], batch_size=4096, verbose=0)
            pred_va = model.predict([*va, sku_val], batch_size=4096, verbose=0)
            pred_te = model.predict([*te, sku_test], batch_size=4096, verbose=0)
            ystruct_tr = np.asarray(pred_tr["final_forecast"]).reshape(-1).astype(np.float32)
            ystruct_va = np.asarray(pred_va["final_forecast"]).reshape(-1).astype(np.float32)
            ystruct_te = np.asarray(pred_te["final_forecast"]).reshape(-1).astype(np.float32)
            p_tr = np.ones_like(ystruct_tr)
            p_va = np.ones_like(ystruct_va)
            p_te = np.ones_like(ystruct_te)
            yhat_struct = ystruct_te

        train_s = time.time() - t0
        return {
            "train_seconds_struct": train_s,
            "ystruct": (ystruct_tr, ystruct_va, ystruct_te),
            "p": (p_tr, p_va, p_te),
            "yhat_struct": yhat_struct,
            "p_te": p_te,
        }

    # ------------------------------------------------------------------
    # Arm A: gated three_term + residual preserve-gate
    # ------------------------------------------------------------------
    a = fit_ds(True, "DS three_term gated (no-lag structural)")
    panel_a = pd.concat(
        [
            pack_panel(train_df, a["ystruct"][0], a["p"][0], "train"),
            pack_panel(val_df, a["ystruct"][1], a["p"][1], "val"),
            pack_panel(test_df, a["ystruct"][2], a["p"][2], "test"),
        ],
        ignore_index=True,
    )
    Xr, yr, ysr, pr, skur, splitsr = build_residual_windows(panel_a, lookback=args.lookback)
    sku_r = np.array([sku_map[s] for s in skur], dtype=np.int32).reshape(-1, 1)
    r_tr, r_va, r_te = splitsr == "train", splitsr == "val", splitsr == "test"

    print("\n=== Residual TF (preserve DS gate) ===")
    tf_a = build_residual_transformer(
        args.lookback, Xr.shape[-1], n_skus, preserve_ds_gate=True
    )
    t0 = time.time()
    train_residual_transformer(
        tf_a,
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
    tf_a_s = time.time() - t0
    yhat_a, p_a, _, _ = predict_residual_transformer(
        tf_a, Xr[r_te], ysr[r_te], sku_r[r_te]
    )
    results["models"]["ds_three_term_residual_preserve_gate"] = {
        "gate": True,
        "loss": "three_term + residual gated MAE",
        "train_seconds": a["train_seconds_struct"] + tf_a_s,
        "overall": kpi_block(yr[r_te], yhat_a, p_a),
        "strata": strata_report(yr[r_te], yhat_a, p_a, skur[r_te], volume_map),
        "structural_only_overall": kpi_block(
            y_test, a["yhat_struct"], a["p_te"]
        ),
    }

    # ------------------------------------------------------------------
    # Arm B: ungated Tweedie + residual (p≡1), Tweedie on residual final
    # ------------------------------------------------------------------
    b = fit_ds(False, "DS ungated Tweedie (no-lag structural)")
    panel_b = pd.concat(
        [
            pack_panel(train_df, b["ystruct"][0], b["p"][0], "train"),
            pack_panel(val_df, b["ystruct"][1], b["p"][1], "val"),
            pack_panel(test_df, b["ystruct"][2], b["p"][2], "test"),
        ],
        ignore_index=True,
    )
    Xr_b, yr_b, ysr_b, pr_b, skur_b, splits_b = build_residual_windows(
        panel_b, lookback=args.lookback
    )
    sku_rb = np.array([sku_map[s] for s in skur_b], dtype=np.int32).reshape(-1, 1)
    b_tr, b_va, b_te = splits_b == "train", splits_b == "val", splits_b == "test"

    print("\n=== Residual TF ungated (p_ds≡1) + Tweedie ===")
    tf_b = build_residual_transformer(
        args.lookback, Xr_b.shape[-1], n_skus, preserve_ds_gate=True, name="residual_ungated"
    )
    # p_ds already ≡ 1 from structural → Multiply is identity; no sigmoid learned
    tf_b_s = train_residual_tweedie(
        tf_b,
        Xr_b[b_tr],
        yr_b[b_tr],
        ysr_b[b_tr],
        sku_rb[b_tr],
        Xr_b[b_va],
        yr_b[b_va],
        ysr_b[b_va],
        sku_rb[b_va],
        args.tweedie_power,
        args.epochs_tf,
        min(512, args.batch_size),
    )
    yhat_b, _, _, _ = predict_residual_transformer(
        tf_b, Xr_b[b_te], ysr_b[b_te], sku_rb[b_te]
    )
    p_b = p_from_mean(yhat_b)
    results["models"]["ds_ungated_tweedie_residual"] = {
        "gate": False,
        "loss": "tweedie_deviance (struct + residual)",
        "train_seconds": b["train_seconds_struct"] + tf_b_s,
        "overall": kpi_block(yr_b[b_te], yhat_b, p_b),
        "strata": strata_report(yr_b[b_te], yhat_b, p_b, skur_b[b_te], volume_map),
        "structural_only_overall": kpi_block(
            y_test, b["yhat_struct"], p_from_mean(b["yhat_struct"])
        ),
    }

    comparison = {"overall": [], "low": [], "mid": [], "high": []}
    for model, payload in results["models"].items():
        for band in comparison:
            block = (
                payload["overall"] if band == "overall" else payload["strata"][band]
            )
            comparison[band].append(
                {
                    "model": model,
                    "gate": payload["gate"],
                    "mae_rounded": block.get("mae_all_rounded"),
                    "mae_nonzero": block.get("mae_nonzero"),
                    "bias": block.get("bias"),
                }
            )
    for band in comparison:
        comparison[band] = sorted(
            comparison[band],
            key=lambda r: (r["mae_rounded"] is None, r["mae_rounded"] or 1e9),
        )
    results["comparison"] = comparison

    mae_a = results["models"]["ds_three_term_residual_preserve_gate"]["overall"][
        "mae_all_rounded"
    ]
    mae_b = results["models"]["ds_ungated_tweedie_residual"]["overall"]["mae_all_rounded"]
    results["verdict"] = {
        "ds_three_term_residual_mae": mae_a,
        "ds_ungated_tweedie_residual_mae": mae_b,
        "ungated_tweedie_residual_beats_gated": mae_b < mae_a,
    }

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("NO-LAG + RESIDUAL: gated three_term vs ungated Tweedie")
    print("=" * 70)
    for band in ("overall", "low", "mid", "high"):
        print(f"\n[{band}]")
        for row in comparison[band]:
            g = "gated" if row["gate"] else "ungated"
            print(
                f"  {row['model']:42s} ({g}) mae={row['mae_rounded']:.3f} "
                f"nz={row['mae_nonzero']:.3f} bias={row['bias']:.3f}"
            )
    print(
        f"\nVerdict: ungated+Tweedie+residual beats gated+residual? "
        f"{mae_b < mae_a} ({mae_b:.3f} vs {mae_a:.3f})"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
