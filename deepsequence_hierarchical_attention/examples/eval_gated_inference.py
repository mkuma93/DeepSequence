#!/usr/bin/env python3
"""
Hypothesis: val-tuned hard gate at inference
  if p < τ → 0
  else     → forecast (continuous / rounded / base)

Choose τ on val to minimize total MAE subject to
  nonzero_MAE(τ) <= nonzero_MAE(ungated) + eps
then evaluate on test.
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
    p.add_argument("--eps_nonzero", type=float, default=0.05,
                   help="Allowed nonzero MAE increase vs ungated")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_gated_inference.json"),
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


def metrics_at(y, magnitude, p, thr, round_mag: bool):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mag = np.maximum(np.asarray(magnitude, dtype=np.float64).reshape(-1), 0.0)
    if round_mag:
        mag = round_forecast(mag)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    gate = (p >= thr).astype(np.float64)
    pred = mag * gate
    nz = y > 0
    return {
        "threshold": float(thr),
        "mae_all": float(mean_absolute_error(y, pred)),
        "mae_nonzero": float(mean_absolute_error(y[nz], pred[nz])) if nz.any() else None,
        "mean_final": float(pred.mean()),
        "pred_nonzero_rate": float(gate.mean()),
        "gate_recall": float(((gate == 1) & nz).sum() / max(nz.sum(), 1)),
        "gate_precision": float((nz & (gate == 1)).sum() / max(gate.sum(), 1)),
    }


def sweep_gate(y, yhat, base, p, nonzero_cap):
    """Return best τ per mode under nonzero MAE constraint."""
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    base = np.asarray(base).reshape(-1)
    p = np.asarray(p).reshape(-1)
    nz = y > 0

    modes = {
        "final": (yhat, False),
        "final_rounded": (yhat, True),
        "base": (base, False),
        "base_rounded": (base, True),
    }
    ungated = {}
    for name, (mag, do_round) in modes.items():
        m = round_forecast(mag) if do_round else np.maximum(mag, 0.0)
        ungated[name] = float(mean_absolute_error(y[nz], m[nz]))

    best = {}
    grid = np.linspace(0.0, 0.95, 20)
    for name, (mag, do_round) in modes.items():
        cap = ungated[name] + nonzero_cap
        candidates = []
        unconstrained_best = None
        for thr in grid:
            m = metrics_at(y, mag, p, thr, do_round)
            m["mode"] = name
            m["nonzero_cap"] = float(cap)
            m["ungated_nonzero_mae"] = float(ungated[name])
            m["feasible"] = m["mae_nonzero"] is not None and m["mae_nonzero"] <= cap + 1e-9
            candidates.append(m)
            if unconstrained_best is None or m["mae_all"] < unconstrained_best["mae_all"]:
                unconstrained_best = dict(m)

        feasible = [c for c in candidates if c["feasible"]]
        if feasible:
            chosen = min(feasible, key=lambda c: c["mae_all"])
            chosen["selection"] = "min_mae_all_under_nonzero_cap"
        else:
            chosen = min(candidates, key=lambda c: (c["mae_nonzero"], c["mae_all"]))
            chosen["selection"] = "fallback_min_nonzero_mae"
        chosen["unconstrained_best_mae_all"] = unconstrained_best
        best[name] = {"chosen": chosen, "n_feasible": len(feasible)}
    return best, ungated


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

    print("\n=== Train DeepSequence three_term ===")
    t0 = time.time()
    model.fit(
        [*tr, sku_train],
        {"final_forecast": y_train.reshape(-1, 1), "base_forecast": y_train.reshape(-1, 1)},
        validation_data=(
            [*va, sku_val],
            {"final_forecast": y_val.reshape(-1, 1), "base_forecast": y_val.reshape(-1, 1)},
        ),
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
    train_s = time.time() - t0

    def predict(parts, sku):
        out = model.predict([*parts, sku], batch_size=4096, verbose=0)
        return (
            np.asarray(out["final_forecast"]).reshape(-1),
            np.asarray(out["base_forecast"]).reshape(-1),
            np.asarray(out["non_zero_probability"]).reshape(-1),
        )

    yhat_va, base_va, p_va = predict(va, sku_val)
    yhat_te, base_te, p_te = predict(te, sku_test)

    modes = ["final", "final_rounded", "base", "base_rounded"]
    print(f"\n=== Val gate sweep (eps_nonzero={args.eps_nonzero}) ===")
    val_best, ungated_nz = sweep_gate(
        y_val, yhat_va, base_va, p_va, args.eps_nonzero
    )

    # Evaluate chosen rules on test
    test_results = {}
    # ungated references (τ=0 => always on)
    test_results["ungated_final"] = metrics_at(y_test, yhat_te, p_te, 0.0, False)
    test_results["ungated_final_rounded"] = metrics_at(
        y_test, yhat_te, p_te, 0.0, True
    )
    test_results["predict_zero"] = {
        "mae_all": float(mean_absolute_error(y_test, np.zeros_like(y_test))),
        "mae_nonzero": float(
            mean_absolute_error(y_test[y_test > 0], np.zeros(np.sum(y_test > 0)))
        ),
    }

    mode_src = {
        "final": (yhat_te, False),
        "final_rounded": (yhat_te, True),
        "base": (base_te, False),
        "base_rounded": (base_te, True),
    }

    summary = []
    for mode in modes:
        thr = val_best[mode]["chosen"]["threshold"]
        mag, do_round = mode_src[mode]
        te_m = metrics_at(y_test, mag, p_te, thr, do_round)
        te_m["mode"] = mode
        te_m["val_chosen"] = {
            k: val_best[mode]["chosen"][k]
            for k in (
                "threshold",
                "mae_all",
                "mae_nonzero",
                "selection",
                "feasible",
                "nonzero_cap",
                "ungated_nonzero_mae",
            )
        }
        te_m["val_n_feasible"] = val_best[mode]["n_feasible"]
        test_results[f"gated_{mode}"] = te_m
        summary.append(
            {
                "rule": f"gated_{mode}",
                "val_tau": thr,
                "test_mae_all": te_m["mae_all"],
                "test_mae_nonzero": te_m["mae_nonzero"],
                "test_mean_final": te_m["mean_final"],
                "gate_recall": te_m["gate_recall"],
                "gate_precision": te_m["gate_precision"],
                "selection": te_m["val_chosen"]["selection"],
            }
        )

    # Unconstrained min-total-MAE on val (final_rounded) → test
    unc = val_best["final_rounded"]["chosen"]["unconstrained_best_mae_all"]
    thr_u = unc["threshold"]
    te_u = metrics_at(y_test, yhat_te, p_te, thr_u, True)
    test_results["gated_final_rounded_unconstrained"] = {
        **te_u,
        "note": "val min total MAE with no nonzero cap",
        "val": unc,
    }
    summary.append(
        {
            "rule": "gated_final_rounded_unconstrained",
            "val_tau": thr_u,
            "test_mae_all": te_u["mae_all"],
            "test_mae_nonzero": te_u["mae_nonzero"],
            "test_mean_final": te_u["mean_final"],
            "gate_recall": te_u["gate_recall"],
            "gate_precision": te_u["gate_precision"],
            "selection": "unconstrained_min_mae_all",
        }
    )

    summary = sorted(summary, key=lambda r: r["test_mae_all"])
    results = {
        "config": {
            "recipe": "three_term + val-tuned hard gate",
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "epochs": args.epochs,
            "eps_nonzero": args.eps_nonzero,
            "train_seconds": train_s,
            "zero_rate": zero_rate,
            "aucroc_test": float(
                roc_auc_score((y_test > 0).astype(float), p_te)
            ),
            "aucpr_test": float(
                average_precision_score((y_test > 0).astype(float), p_te)
            ),
        },
        "ungated_nonzero_mae_val": ungated_nz,
        "test": test_results,
        "summary_sorted_by_test_mae": summary,
    }
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("GATED INFERENCE SUMMARY (test)")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(
        "ungated_final test MAE",
        test_results["ungated_final"]["mae_all"],
        "nz",
        test_results["ungated_final"]["mae_nonzero"],
    )
    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
