#!/usr/bin/env python3
"""
Locked daily H=1: plain DeepSequence vs hybrid-decoupled DS vs TST.

Hybrid = current-row hierarchical experts + causal lookback MHA feeding both
magnitude and occurrence, with a decoupled gate (no softplus base / component
scalars into the occurrence branch).
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
from deepsequence_hierarchical_attention.hybrid_temporal import (
    build_hierarchical_model_hybrid,
)
from train_lightweight_adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from eval_helpers import (
    add_panel_seed_args,
    build_transformer,
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
from eval_same_features_compare import train_seq_three_term

ALL_MODELS = ("deepsequence", "deepsequence_hybrid", "temporal_transformer")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    add_panel_seed_args(p)
    p.add_argument("--models", default=",".join(ALL_MODELS))
    p.add_argument("--temporal_d_model", type=int, default=32)
    p.add_argument("--temporal_n_heads", type=int, default=4)
    p.add_argument("--temporal_n_blocks", type=int, default=1)
    p.add_argument(
        "--decouple_gate",
        action="store_true",
        default=True,
        help="Decouple occurrence from magnitude experts/base (default on for hybrid)",
    )
    p.add_argument("--no_decouple_gate", action="store_false", dest="decouple_gate")
    p.add_argument(
        "--hybrid_result_key",
        default="deepsequence_hybrid",
        help="JSON key for the hybrid model result block",
    )
    p.add_argument(
        "--out_json",
        default=str(ROOT / "ab_runs" / "reclaim" / "daily_h1_hybrid_temporal.json"),
    )
    return p.parse_args()


def build_hybrid_aligned_windows(
    train_df, val_df, test_df, X_train, X_val, X_test, lookback: int
):
    """
    Same causal windows as TST, plus tabular X at the predict day ``t``.

    Returns
    -------
    Xseq, Xtab, y, sku_raw, splits, n_channels
    """
    metas = []
    feats = []
    tabs = []
    offset = 0
    for split, df, X in [
        ("train", train_df, X_train),
        ("val", val_df, X_val),
        ("test", test_df, X_test),
    ]:
        y = df["Quantity"].to_numpy(np.float32).reshape(-1, 1)
        X = X.astype(np.float32)
        block = np.concatenate([y, X], axis=1)
        feats.append(block)
        tabs.append(X)
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
    tab_all = np.concatenate(tabs, axis=0)
    meta = pd.concat(metas, ignore_index=True)
    meta = meta.sort_values(["id_var", "ds"], kind="mergesort").reset_index(drop=True)
    n_channels = feat_all.shape[1]

    xs, xt, ys, skus, splits = [], [], [], [], []
    for sku, g in meta.groupby("id_var", sort=False):
        pos = g["_pos"].to_numpy()
        arr = feat_all[pos]
        tab = tab_all[pos]
        y = g["y"].to_numpy(np.float32)
        sp = g["split"].to_numpy()
        n = len(g)
        if n <= lookback:
            continue
        for t in range(lookback, n):
            xs.append(arr[t - lookback : t])
            xt.append(tab[t])
            ys.append(y[t])
            skus.append(sku)
            splits.append(sp[t])

    Xseq = (
        np.stack(xs).astype(np.float32)
        if xs
        else np.zeros((0, lookback, n_channels), np.float32)
    )
    Xtab = (
        np.stack(xt).astype(np.float32)
        if xt
        else np.zeros((0, tab_all.shape[1]), np.float32)
    )
    return (
        Xseq,
        Xtab,
        np.asarray(ys, np.float32),
        np.asarray(skus),
        np.asarray(splits),
        n_channels,
    )


def _train_adaptive(model, inputs_tr, y_train, inputs_va, y_val, zero_rate, args, label):
    pos_weight = min(20.0, zero_rate / max(1 - zero_rate, 1e-3))
    wrapped = AdaptiveWeightedModel(
        base_model=model,
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
    wrapped.compile(optimizer=tf.keras.optimizers.Adam(0.0025))
    ytr = {
        "final_forecast": y_train.reshape(-1, 1),
        "base_forecast": y_train.reshape(-1, 1),
    }
    yva = {
        "final_forecast": y_val.reshape(-1, 1),
        "base_forecast": y_val.reshape(-1, 1),
    }
    print(f"\n=== {label} ===")
    t0 = time.time()
    wrapped.fit(
        inputs_tr,
        ytr,
        validation_data=(inputs_va, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            )
        ],
        verbose=2,
    )
    return wrapped, time.time() - t0


def main():
    args = parse_args()
    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    unknown = selected - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"Unknown models: {sorted(unknown)}")

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
            "zero_rate": zero_rate,
            "temporal_d_model": args.temporal_d_model,
            "temporal_n_heads": args.temporal_n_heads,
            "temporal_n_blocks": args.temporal_n_blocks,
            "decouple_gate": args.decouple_gate,
            "volume_stats": volume_stats,
            "models_run": sorted(selected),
            "note": (
                "Hybrid: experts + causal MHA lookback → b and p; "
                "decoupled gate (raw regressors+SKU+temporal, no base/components)."
            ),
        },
        "models": {},
    }

    if "deepsequence" in selected:
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
        model, train_s = _train_adaptive(
            base,
            [*tr, sku_train],
            y_train,
            [*va, sku_val],
            y_val,
            zero_rate,
            args,
            "DeepSequence (plain tabular)",
        )
        pred = model.predict([*te, sku_test], batch_size=4096, verbose=0)
        yhat = np.asarray(pred["final_forecast"]).reshape(-1)
        p = np.asarray(pred["non_zero_probability"]).reshape(-1)
        results["models"]["deepsequence"] = {
            "train_seconds": train_s,
            "overall": kpi_block(y_test, yhat, p, mase_scale=mase_scale),
            "strata": strata_report(
                y_test, yhat, p, sku_test_raw, volume_map, mase_scale=mase_scale
            ),
        }

    need_windows = bool(selected & {"deepsequence_hybrid", "temporal_transformer"})
    if need_windows:
        print("\nBuilding hybrid/TST aligned windows...")
        Xseq, Xtab, yseq, sku_seq_raw, split_seq, n_channels = build_hybrid_aligned_windows(
            train_df, val_df, test_df, X_train, X_val, X_test, args.lookback
        )
        sku_seq = np.array([sku_map[s] for s in sku_seq_raw], dtype=np.int32).reshape(-1, 1)
        m_tr = split_seq == "train"
        m_va = split_seq == "val"
        m_te = split_seq == "test"
        print(
            f"windows train/val/test={m_tr.sum()}/{m_va.sum()}/{m_te.sum()} "
            f"channels={n_channels}"
        )
        tr_w = split_components(Xtab[m_tr], cfg)
        va_w = split_components(Xtab[m_va], cfg)
        te_w = split_components(Xtab[m_te], cfg)

        if "deepsequence_hybrid" in selected:
            hybrid_key = args.hybrid_result_key
            print(
                f"\nHybrid config: d_model={args.temporal_d_model} "
                f"heads={args.temporal_n_heads} blocks={args.temporal_n_blocks} "
                f"decouple_gate={args.decouple_gate} key={hybrid_key}"
            )
            hybrid = build_hierarchical_model_hybrid(
                n_temporal_features=len(cfg.trend_indices),
                n_fourier_features=len(cfg.seasonal_indices),
                n_holiday_features=len(cfg.holiday_indices),
                n_lag_features=len(cfg.regressor_indices),
                n_skus=n_skus,
                n_sequence_channels=n_channels,
                lookback=args.lookback,
                temporal_d_model=args.temporal_d_model,
                temporal_n_heads=args.temporal_n_heads,
                temporal_n_blocks=args.temporal_n_blocks,
                decouple_gate=args.decouple_gate,
                hidden_dim=48,
                sku_embedding_dim=4,
                dropout_rate=0.23,
                use_cross_layers=True,
                use_intermittent=True,
                n_changepoints=15,
            )
            dummy_seq = np.zeros((1, args.lookback, n_channels), np.float32)
            _ = hybrid(
                [
                    *(np.zeros((1, x.shape[1]), np.float32) for x in tr_w),
                    np.zeros((1, 1), np.int32),
                    dummy_seq,
                ],
                training=False,
            )
            model, train_s = _train_adaptive(
                hybrid,
                [*tr_w, sku_seq[m_tr], Xseq[m_tr]],
                yseq[m_tr],
                [*va_w, sku_seq[m_va], Xseq[m_va]],
                yseq[m_va],
                zero_rate,
                args,
                f"DeepSequence hybrid ({hybrid_key})",
            )
            pred = model.predict(
                [*te_w, sku_seq[m_te], Xseq[m_te]], batch_size=4096, verbose=0
            )
            yhat = np.asarray(pred["final_forecast"]).reshape(-1)
            p = np.asarray(pred["non_zero_probability"]).reshape(-1)
            results["models"][hybrid_key] = {
                "train_seconds": train_s,
                "n_channels": n_channels,
                "temporal_d_model": args.temporal_d_model,
                "temporal_n_heads": args.temporal_n_heads,
                "temporal_n_blocks": args.temporal_n_blocks,
                "decouple_gate": args.decouple_gate,
                "overall": kpi_block(yseq[m_te], yhat, p, mase_scale=mase_scale),
                "strata": strata_report(
                    yseq[m_te],
                    yhat,
                    p,
                    sku_seq_raw[m_te],
                    volume_map,
                    mase_scale=mase_scale,
                ),
            }

        if "temporal_transformer" in selected:
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
                    yseq[m_te],
                    yhat,
                    p,
                    sku_seq_raw[m_te],
                    volume_map,
                    mase_scale=mase_scale,
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
                "inventory_nv_cost_rounded_cu2": block.get(
                    "inventory_nv_cost_rounded_cu2"
                ),
                "inventory_holding_proxy_zero": block.get(
                    "inventory_holding_proxy_zero"
                ),
                "inventory_stockout_proxy_nz": block.get(
                    "inventory_stockout_proxy_nz"
                ),
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
    print("HYBRID TEMPORAL H=1 (primary: iwmae_rounded)")
    print("=" * 70)
    for row in comparison:
        nv = row.get("inventory_nv_cost_rounded_cu2")
        nv_s = f"{nv:.3f}" if nv is not None else "n/a"
        print(
            f"  {row['model']:28s} iwmae={row['iwmae_rounded']:.3f} "
            f"nv_cu2={nv_s} mean_p={row.get('mean_p')} bias={row['bias']:.3f} "
            f"occ_f1={row.get('occ_f1')}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
