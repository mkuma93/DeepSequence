#!/usr/bin/env python3
"""
Fixed-ω vs learnable-ω seasonal Fourier on DS three_term (no-lag contract).

Fixed: precomputed dow/month/year sin/cos (periods 7 / 12 / 365.25).
Learnable: SeasonalComponent LearnableFourierFeatures; input = raw time_index
           (days since epoch); ω trainable, init periods [7,14,30,91,365].
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
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_learnable_frequencies", type=int, default=5)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_learnable_omega.json"),
    )
    return p.parse_args()


def extract_learned_periods(model) -> list[float] | None:
    for layer in model.layers:
        if hasattr(layer, "get_learned_periods"):
            try:
                return [float(x) for x in layer.get_learned_periods()]
            except Exception:
                pass
        # nested: seasonal has fourier_layer
        if hasattr(layer, "fourier_layer") and hasattr(
            layer.fourier_layer, "get_learned_periods"
        ):
            try:
                return [float(x) for x in layer.fourier_layer.get_learned_periods()]
            except Exception:
                pass
    # walk AdaptiveWeightedModel.base_model
    base = getattr(model, "base_model", None) or getattr(model, "base", None)
    if base is not None and base is not model:
        return extract_learned_periods(base)
    for layer in getattr(model, "layers", []):
        for sub in getattr(layer, "layers", []) or []:
            if hasattr(sub, "get_learned_periods"):
                try:
                    return [float(x) for x in sub.get_learned_periods()]
                except Exception:
                    pass
            if hasattr(sub, "fourier_layer") and hasattr(
                sub.fourier_layer, "get_learned_periods"
            ):
                try:
                    return [
                        float(x) for x in sub.fourier_layer.get_learned_periods()
                    ]
                except Exception:
                    pass
    # Keras 3: search weights named log_frequencies
    for w in model.weights:
        if "log_frequencies" in w.name:
            freqs = np.exp(w.numpy())
            periods = (2 * np.pi) / freqs
            return [float(x) for x in periods]
    return None


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    data_dir = Path(args.data_dir)
    init_periods = [7.0, 14.0, 30.0, 91.0, 365.0][: args.n_learnable_frequencies]

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
    sku_test_raw = test_df["id_var"].to_numpy()
    zero_rate = float((y_train == 0).mean())
    pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))

    cfg = load_feature_config()
    print("Building no-lag features...")
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, states = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
    Xtr_df, dropped = drop_lag_columns(Xtr_df, cfg)
    Xva_df, _ = drop_lag_columns(Xva_df, cfg)
    Xte_df, _ = drop_lag_columns(Xte_df, cfg)
    feature_names = list(Xtr_df.columns)

    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)

    # Raw days for learnable ω; normalized copy for trend (same as prior evals)
    t_idx = feature_names.index("time_index")
    time_raw_tr = X_train[:, t_idx : t_idx + 1].copy()
    time_raw_va = X_val[:, t_idx : t_idx + 1].copy()
    time_raw_te = X_test[:, t_idx : t_idx + 1].copy()
    tmin, tmax = float(time_raw_tr.min()), float(time_raw_tr.max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    trend_i, seas_i, hol_i, inter_i = component_indices_nolag(feature_names, cfg)

    def parts_fixed(X):
        return (
            X[:, trend_i].astype(np.float32),
            X[:, seas_i].astype(np.float32),
            X[:, hol_i].astype(np.float32),
            X[:, inter_i].astype(np.float32),
        )

    def parts_learnable(X, time_raw):
        return (
            X[:, trend_i].astype(np.float32),
            time_raw.astype(np.float32),
            X[:, hol_i].astype(np.float32),
            X[:, inter_i].astype(np.float32),
        )

    tr_f, va_f, te_f = parts_fixed(X_train), parts_fixed(X_val), parts_fixed(X_test)
    tr_l = parts_learnable(X_train, time_raw_tr)
    va_l = parts_learnable(X_val, time_raw_va)
    te_l = parts_learnable(X_test, time_raw_te)

    results = {
        "config": {
            "max_skus": args.max_skus,
            "n_skus": n_skus,
            "seed": args.seed,
            "zero_rate": zero_rate,
            "dropped_lags": dropped,
            "init_periods": init_periods,
            "n_learnable_frequencies": args.n_learnable_frequencies,
            "volume_stats": volume_stats,
        },
        "models": {},
    }

    def train_ds(tr, va, te, use_learnable: bool, label: str):
        print(f"\n=== {label} ===")
        n_fourier = 1 if use_learnable else len(seas_i)
        base = build_hierarchical_model_lightweight(
            n_temporal_features=len(trend_i),
            n_fourier_features=n_fourier,
            n_holiday_features=len(hol_i),
            n_lag_features=len(inter_i),
            n_skus=n_skus,
            hidden_dim=48,
            sku_embedding_dim=4,
            dropout_rate=0.23,
            use_cross_layers=True,
            use_intermittent=True,
            enable_regressor=True,
            n_changepoints=15,
            use_learnable_fourier=use_learnable,
            n_learnable_frequencies=args.n_learnable_frequencies,
            fourier_periods=init_periods if use_learnable else None,
        )
        _ = base(
            [
                *(np.zeros((1, x.shape[1]), np.float32) for x in tr),
                np.zeros((1, 1), np.int32),
            ],
            training=False,
        )
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
        t0 = time.time()
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
        train_s = time.time() - t0
        pred = model.predict([*te, sku_test], batch_size=4096, verbose=0)
        yhat = np.asarray(pred["final_forecast"]).reshape(-1)
        p = np.asarray(pred["non_zero_probability"]).reshape(-1)
        learned = extract_learned_periods(base) if use_learnable else None
        if learned:
            print(f"learned periods (days): {learned}")
        return {
            "use_learnable_fourier": use_learnable,
            "train_seconds": train_s,
            "init_periods": init_periods if use_learnable else [7.0, 12.0, 365.25],
            "learned_periods": learned,
            "overall": kpi_block(y_test, yhat, p),
            "strata": strata_report(y_test, yhat, p, sku_test_raw, volume_map),
        }

    results["models"]["ds_three_term_fixed_omega"] = train_ds(
        tr_f, va_f, te_f, False, "DS three_term fixed ω (precomputed sin/cos)"
    )
    results["models"]["ds_three_term_learnable_omega"] = train_ds(
        tr_l, va_l, te_l, True, "DS three_term learnable ω"
    )

    comparison = {"overall": [], "low": [], "mid": [], "high": []}
    for model, payload in results["models"].items():
        for band in comparison:
            block = (
                payload["overall"] if band == "overall" else payload["strata"][band]
            )
            comparison[band].append(
                {
                    "model": model,
                    "mae_rounded": block.get("mae_all_rounded"),
                    "mae_nonzero": block.get("mae_nonzero"),
                    "aucroc": block.get("aucroc"),
                    "bias": block.get("bias"),
                }
            )
    for band in comparison:
        comparison[band] = sorted(
            comparison[band],
            key=lambda r: (r["mae_rounded"] is None, r["mae_rounded"] or 1e9),
        )
    results["comparison"] = comparison
    mae_f = results["models"]["ds_three_term_fixed_omega"]["overall"]["mae_all_rounded"]
    mae_l = results["models"]["ds_three_term_learnable_omega"]["overall"][
        "mae_all_rounded"
    ]
    results["verdict"] = {
        "fixed_mae": mae_f,
        "learnable_mae": mae_l,
        "learnable_beats_fixed": mae_l < mae_f,
        "learned_periods": results["models"]["ds_three_term_learnable_omega"].get(
            "learned_periods"
        ),
    }

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("FIXED ω vs LEARNABLE ω")
    print("=" * 70)
    for band in ("overall", "low", "mid", "high"):
        print(f"\n[{band}]")
        for row in comparison[band]:
            print(
                f"  {row['model']:40s} mae={row['mae_rounded']:.3f} "
                f"nz={row['mae_nonzero']:.3f} bias={row['bias']:.3f}"
            )
    print(
        f"\nVerdict: learnable beats fixed? {mae_l < mae_f} "
        f"({mae_l:.3f} vs {mae_f:.3f})"
    )
    print(
        "learned periods:",
        results["models"]["ds_three_term_learnable_omega"].get("learned_periods"),
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
