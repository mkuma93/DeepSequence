#!/usr/bin/env python3
"""
Locked-panel reclaim bake-off: DeepSequence vs DS+residual vs TST.

Uses lag-based DeepSequence (paper trunk) plus the optional residual transformer
that preserves DS gate ``p``. Does not revive the old no-lag structural stack.

Gate calibration knobs (opt-in; paper defaults stay off in the package):
  --gate_prior           bias-init gate toward train zero rate
  --gate_raw_regressors  feed projected lag/intermittent features into the gate
  --gate_temp_calibrate  post-hoc validation temperature on ``p``
"""

from __future__ import annotations

import argparse
import json
import os
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
from deepsequence_hierarchical_attention.residual_transformer import (
    build_residual_transformer,
    build_residual_windows,
    predict_residual_transformer,
    train_residual_transformer,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from eval_helpers import (
    add_panel_seed_args,
    apply_iwmae_gate,
    apply_probability_temperature,
    build_transformer,
    calibrate_iwmae_gate,
    calibrate_probability_temperature,
    filter_aligned,
    kpi_block,
    predict_seq,
    resolve_eval_seeds,
    select_eval_skus,
    split_components,
    strata_report,
    train_mase_scale,
    train_volume_terciles,
)
from eval_same_features_compare import build_full_feature_sequences, train_seq_three_term


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--residual_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    add_panel_seed_args(p)
    p.add_argument(
        "--models",
        default="deepsequence,deepsequence_calibrated,deepsequence_residual,temporal_transformer",
        help="Comma-separated subset of reclaim models",
    )
    p.add_argument("--gate_prior", action="store_true", default=True)
    p.add_argument("--no_gate_prior", action="store_false", dest="gate_prior")
    p.add_argument("--gate_raw_regressors", action="store_true", default=True)
    p.add_argument(
        "--no_gate_raw_regressors", action="store_false", dest="gate_raw_regressors"
    )
    p.add_argument("--gate_temp_calibrate", action="store_true", default=False)
    p.add_argument(
        "--no_gate_temp_calibrate", action="store_false", dest="gate_temp_calibrate"
    )
    p.add_argument(
        "--gate_iwmae_calibrate",
        action="store_true",
        default=True,
        help="Post-hoc val search over scale + p-threshold (preferred for hot gates)",
    )
    p.add_argument(
        "--no_gate_iwmae_calibrate", action="store_false", dest="gate_iwmae_calibrate"
    )
    p.add_argument(
        "--out_json",
        default=str(ROOT / "ab_runs" / "reclaim" / "daily_h1_residual_reclaim.json"),
    )
    return p.parse_args()


def _train_ds(tr, va, te, y_train, y_val, sku_train, sku_val, sku_test, zero_rate, n_skus, cfg, args, *, calibrate: bool):
    prior = zero_rate if calibrate and args.gate_prior else None
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
        gate_use_raw_regressors=bool(calibrate and args.gate_raw_regressors),
        intermittent_prior_zero_rate=prior,
        intermittent_gate_temperature=1.0,
        intermittent_learnable_temperature=False,
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
    ytr = {"final_forecast": y_train.reshape(-1, 1), "base_forecast": y_train.reshape(-1, 1)}
    yva = {"final_forecast": y_val.reshape(-1, 1), "base_forecast": y_val.reshape(-1, 1)}
    t0 = time.time()
    model.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            )
        ],
        verbose=2,
    )
    train_s = time.time() - t0

    def _predict(parts, sku):
        pred = model.predict([*parts, sku], batch_size=4096, verbose=0)
        return (
            np.asarray(pred["final_forecast"]).reshape(-1),
            np.asarray(pred["base_forecast"]).reshape(-1),
            np.asarray(pred["non_zero_probability"]).reshape(-1),
        )

    yhat_tr, base_tr, p_tr = _predict(tr, sku_train)
    yhat_va, base_va, p_va = _predict(va, sku_val)
    yhat_te, base_te, p_te = _predict(te, sku_test)

    calib = None
    iwmae_calib = None
    if calibrate and args.gate_temp_calibrate:
        calib = calibrate_probability_temperature(y_val, yhat_va, p_va)
        yhat_te, p_te = apply_probability_temperature(
            yhat_te, p_te, temperature=calib["temperature"]
        )
        yhat_va, p_va = apply_probability_temperature(
            yhat_va, p_va, temperature=calib["temperature"]
        )
        yhat_tr, p_tr = apply_probability_temperature(
            yhat_tr, p_tr, temperature=calib["temperature"]
        )
        base_tr = yhat_tr / np.clip(p_tr, 1e-6, 1.0)
        base_va = yhat_va / np.clip(p_va, 1e-6, 1.0)
        base_te = yhat_te / np.clip(p_te, 1e-6, 1.0)

    if calibrate and args.gate_iwmae_calibrate:
        iwmae_calib = calibrate_iwmae_gate(y_val, yhat_va, p_va)
        yhat_te = apply_iwmae_gate(
            yhat_te, p_te, scale=iwmae_calib["scale"], threshold=iwmae_calib["threshold"]
        )
        yhat_va = apply_iwmae_gate(
            yhat_va, p_va, scale=iwmae_calib["scale"], threshold=iwmae_calib["threshold"]
        )
        yhat_tr = apply_iwmae_gate(
            yhat_tr, p_tr, scale=iwmae_calib["scale"], threshold=iwmae_calib["threshold"]
        )
        # Keep residual magnitude channel consistent with calibrated final.
        base_tr = yhat_tr / np.clip(p_tr, 1e-6, 1.0)
        base_va = yhat_va / np.clip(p_va, 1e-6, 1.0)
        base_te = yhat_te / np.clip(p_te, 1e-6, 1.0)

    return {
        "model": model,
        "train_seconds": train_s,
        "yhat": {"train": yhat_tr, "val": yhat_va, "test": yhat_te},
        "base": {"train": base_tr, "val": base_va, "test": base_te},
        "p": {"train": p_tr, "val": p_va, "test": p_te},
        "calibration": calib,
        "iwmae_calibration": iwmae_calib,
        "calibrated": bool(calibrate),
    }


def _panel_from_splits(train_df, val_df, test_df, base, p):
    frames = []
    for split, df, key in (
        ("train", train_df, "train"),
        ("val", val_df, "val"),
        ("test", test_df, "test"),
    ):
        part = pd.DataFrame(
            {
                "id_var": df["id_var"].to_numpy(),
                "ds": pd.to_datetime(df["ds"]),
                "y": df["Quantity"].to_numpy(np.float32),
                "y_struct": np.asarray(base[key], np.float32),
                "p_ds": np.asarray(p[key], np.float32),
                "split": split,
            }
        )
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()
    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    data_seed, train_seed = resolve_eval_seeds(args.seed, args.data_seed, args.train_seed)
    tf.keras.utils.set_random_seed(train_seed)

    data_dir_raw = args.data_dir or os.environ.get("DEEPSEQUENCE_DATA_DIR")
    if not data_dir_raw:
        raise SystemExit("Pass --data_dir or set DEEPSEQUENCE_DATA_DIR")
    data_dir = Path(data_dir_raw)

    train_df = pd.read_csv(data_dir / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(data_dir / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(data_dir / "test_split.csv", parse_dates=["ds"])
    h_tr = pd.read_csv(data_dir / "holiday_features_train.csv")
    h_va = pd.read_csv(data_dir / "holiday_features_val.csv")
    h_te = pd.read_csv(data_dir / "holiday_features_test.csv")

    chosen_list = select_eval_skus(
        train_df["id_var"].unique(),
        max_skus=args.max_skus,
        data_seed=data_seed,
        sku_list_path=args.sku_list,
        save_sku_list_path=args.save_sku_list,
    )
    chosen = set(chosen_list)
    print(
        f"Panel lock: data_seed={data_seed} train_seed={train_seed} n_skus={len(chosen)}"
        + (f" sku_list={args.sku_list}" if args.sku_list else "")
    )
    train_df, h_tr = filter_aligned(train_df, h_tr, chosen)
    val_df, h_va = filter_aligned(val_df, h_va, chosen)
    test_df, h_te = filter_aligned(test_df, h_te, chosen)

    volume_map, volume_stats = train_volume_terciles(train_df)
    mase_scale = train_mase_scale(train_df, season=7)

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
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(val_df, h_va, prior_states=states, return_states=True)
    Xte_df, _ = cfg.create_features(test_df, h_te, prior_states=states, return_states=True)
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
            "data_seed": data_seed,
            "train_seed": train_seed,
            "sku_list": args.sku_list,
            "lookback": args.lookback,
            "epochs": args.epochs,
            "residual_epochs": args.residual_epochs,
            "zero_rate": zero_rate,
            "gate_prior": args.gate_prior,
            "gate_raw_regressors": args.gate_raw_regressors,
            "gate_temp_calibrate": args.gate_temp_calibrate,
            "gate_iwmae_calibrate": args.gate_iwmae_calibrate,
            "volume_stats": volume_stats,
            "models_run": sorted(selected),
            "note": (
                "Lag-based DeepSequence + residual transformer (preserve p). "
                "Calibrated DS enables prior/raw-regressor gate + val temperature."
            ),
        },
        "models": {},
    }

    ds_bundle = None
    if "deepsequence" in selected or "deepsequence_residual" in selected:
        print("\n=== DeepSequence (paper defaults / no gate reclaim) ===")
        plain = _train_ds(
            tr, va, te, y_train, y_val, sku_train, sku_val, sku_test,
            zero_rate, n_skus, cfg, args, calibrate=False,
        )
        if "deepsequence" in selected:
            results["models"]["deepsequence"] = {
                "train_seconds": plain["train_seconds"],
                "overall": kpi_block(
                    y_test, plain["yhat"]["test"], plain["p"]["test"], mase_scale=mase_scale
                ),
                "strata": strata_report(
                    y_test,
                    plain["yhat"]["test"],
                    plain["p"]["test"],
                    sku_test_raw,
                    volume_map,
                    mase_scale=mase_scale,
                ),
            }
        # Residual always sits on calibrated DS when available; else plain.
        ds_bundle = plain

    if "deepsequence_calibrated" in selected or "deepsequence_residual" in selected:
        print("\n=== DeepSequence (gate calibration on) ===")
        cal = _train_ds(
            tr, va, te, y_train, y_val, sku_train, sku_val, sku_test,
            zero_rate, n_skus, cfg, args, calibrate=True,
        )
        if "deepsequence_calibrated" in selected:
            results["models"]["deepsequence_calibrated"] = {
                "train_seconds": cal["train_seconds"],
                "calibration": cal["calibration"],
                "iwmae_calibration": cal["iwmae_calibration"],
                "overall": kpi_block(
                    y_test, cal["yhat"]["test"], cal["p"]["test"], mase_scale=mase_scale
                ),
                "strata": strata_report(
                    y_test,
                    cal["yhat"]["test"],
                    cal["p"]["test"],
                    sku_test_raw,
                    volume_map,
                    mase_scale=mase_scale,
                ),
            }
        ds_bundle = cal

    if "deepsequence_residual" in selected:
        if ds_bundle is None:
            raise SystemExit("deepsequence_residual requires a DeepSequence trunk")
        print("\n=== DeepSequence + residual transformer (preserve p) ===")
        panel = _panel_from_splits(train_df, val_df, test_df, ds_bundle["base"], ds_bundle["p"])
        Xr, yr, ystruct, p_ds, sku_raw, splits = build_residual_windows(
            panel, lookback=args.lookback
        )
        sku_r = np.array([sku_map[s] for s in sku_raw], dtype=np.int32).reshape(-1, 1)
        m_tr = splits == "train"
        m_va = splits == "val"
        m_te = splits == "test"
        print(f"residual windows train/val/test={m_tr.sum()}/{m_va.sum()}/{m_te.sum()}")

        rt = build_residual_transformer(
            lookback=args.lookback,
            n_channels=Xr.shape[-1],
            n_skus=n_skus,
            preserve_ds_gate=True,
            encoder_gate_mix=0.0,
        )
        t0 = time.time()
        wrapped = train_residual_transformer(
            rt,
            Xr[m_tr],
            yr[m_tr],
            ystruct[m_tr],
            sku_r[m_tr],
            Xr[m_va],
            yr[m_va],
            ystruct[m_va],
            sku_r[m_va],
            zero_rate=zero_rate,
            epochs=args.residual_epochs,
            batch_size=args.batch_size,
            learning_rate=0.002,
        )
        rt_s = time.time() - t0
        yhat_r, p_r, base_r, delta_r = predict_residual_transformer(
            wrapped.base, Xr[m_te], ystruct[m_te], sku_r[m_te]
        )
        # Trunk forecast on the same residual test origins (y_struct * p_ds).
        yhat_trunk_aligned = np.maximum(ystruct[m_te] * p_ds[m_te], 0.0)
        results["models"]["deepsequence_residual"] = {
            "train_seconds": ds_bundle["train_seconds"] + rt_s,
            "residual_train_seconds": rt_s,
            "trunk": "deepsequence_calibrated"
            if ds_bundle.get("calibrated")
            else "deepsequence",
            "mean_abs_delta": float(np.mean(np.abs(delta_r))),
            "trunk_aligned_overall": kpi_block(
                yr[m_te], yhat_trunk_aligned, p_ds[m_te], mase_scale=mase_scale
            ),
            "overall": kpi_block(yr[m_te], yhat_r, p_r, mase_scale=mase_scale),
            "strata": strata_report(
                yr[m_te],
                yhat_r,
                p_r,
                sku_raw[m_te],
                volume_map,
                mase_scale=mase_scale,
            ),
        }

    if "temporal_transformer" in selected:
        print("\n=== Temporal transformer (same features) ===")
        Xseq, yseq, sku_seq_raw, split_seq, n_channels = build_full_feature_sequences(
            train_df, val_df, test_df, X_train, X_val, X_test, args.lookback
        )
        sku_seq = np.array([sku_map[s] for s in sku_seq_raw], dtype=np.int32).reshape(-1, 1)
        m_tr = split_seq == "train"
        m_va = split_seq == "val"
        m_te = split_seq == "test"
        model = build_transformer(args.lookback, n_skus, n_channels=n_channels)
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
            "temporal_transformer",
        )
        yhat, p = predict_seq(model, Xseq[m_te], sku_seq[m_te])
        results["models"]["temporal_transformer"] = {
            "train_seconds": train_s,
            "n_channels": n_channels,
            "overall": kpi_block(yseq[m_te], yhat, p, mase_scale=mase_scale),
            "strata": strata_report(
                yseq[m_te], yhat, p, sku_seq_raw[m_te], volume_map, mase_scale=mase_scale
            ),
        }

    comparison = []
    for name, payload in results["models"].items():
        block = payload["overall"]
        comparison.append(
            {
                "model": name,
                "iwmae_rounded": block.get("iwmae_rounded"),
                "mae_rounded": block.get("mae_all_rounded"),
                "mean_p": block.get("mean_p"),
                "bias": block.get("bias"),
                "occ_f1": block.get("occ_f1"),
            }
        )
    comparison = sorted(
        comparison,
        key=lambda r: (
            r["iwmae_rounded"] is None,
            r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
        ),
    )
    results["comparison"] = comparison

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("RECLAIM H=1 (primary: iwmae_rounded)")
    print("=" * 70)
    for row in comparison:
        print(
            f"  {row['model']:28s} iwmae={row['iwmae_rounded']:.3f} "
            f"mae={row['mae_rounded']:.3f} mean_p={row.get('mean_p')} "
            f"bias={row['bias']:.3f} occ_f1={row.get('occ_f1')}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
