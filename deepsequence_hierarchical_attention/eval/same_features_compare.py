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
  - TFT-lite (same windows; GRN + var-select + LSTM + attention)
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deepsequence_hierarchical_attention.data.feature_config_loader import load_feature_config
from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.losses import three_term_loss_config
from deepsequence_hierarchical_attention.training.adaptive_loss import AdaptiveWeightedModel, WeightedBCELoss
from deepsequence_hierarchical_attention.eval.helpers import (
    add_panel_seed_args,
    build_deepar,
    build_tft,
    build_transformer,
    filter_aligned,
    fit_bce_sample_weight_dict,
    kpi_block,
    predict_seq,
    resolve_eval_seeds,
    resolve_sku_zero_rates,
    select_eval_skus,
    split_components,
    strata_report,
    train_mase_scale,
    train_volume_terciles,
)

ALL_MODELS = (
    "deepsequence",
    "lightgbm",
    "deepar_lite",
    "temporal_transformer",
    "tft_lite",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default=None,
        help="Panel data directory (or set DEEPSEQUENCE_DATA_DIR). Required for a real bake-off run.",
    )
    p.add_argument("--max_skus", type=int, default=800)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lookback", type=int, default=14)
    add_panel_seed_args(p)
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=f"Comma-separated subset of: {','.join(ALL_MODELS)}",
    )
    p.add_argument(
        "--ds_train_gate_calibrate",
        action="store_true",
        default=False,
        help=(
            "Opt-in train-time gate calibration for DeepSequence: prior zero_rate, "
            "raw regressors→gate, learnable logit scale, softplus p-scale, rate-match."
        ),
    )
    p.add_argument("--ds_gate_prob_scale_init", type=float, default=0.85)
    p.add_argument("--ds_gate_rate_match_weight", type=float, default=0.01)
    p.add_argument(
        "--merge_json",
        default=None,
        help="If set, merge model results into this existing JSON (keeps prior models).",
    )
    p.add_argument(
        "--out_json",
        default=str(ROOT / "eval_results_same_features_v16_distance_holidays.json"),
    )
    # DeepSequence stack overrides (None = preferred builder defaults:
    # softsign + mono + mixer on + L1 attn on + cross off + additive).
    p.add_argument("--output_activation", default=None)
    p.add_argument("--trend_monotonic", type=int, default=None)
    p.add_argument("--holiday_monotonic", type=int, default=None)
    p.add_argument("--regressor_monotonic", type=int, default=None)
    p.add_argument("--context_aware_component_mixer", type=int, default=None)
    p.add_argument("--context_film_seasonal_holiday", type=int, default=None)
    p.add_argument("--level1_selection_attention", type=int, default=None)
    p.add_argument(
        "--use_cross_layers",
        type=int,
        default=None,
        help="DCN cross on component outputs (1/0). None = builder default (False).",
    )
    return p.parse_args()


def _ds_builder_kwargs(args) -> dict:
    """Resolve DeepSequence stack kwargs from CLI (explicit overrides only)."""
    import inspect

    sig = inspect.signature(build_hierarchical_model_lightweight)
    defaults = {
        k: sig.parameters[k].default
        for k in (
            "output_activation",
            "trend_monotonic",
            "holiday_monotonic",
            "regressor_monotonic",
            "context_aware_component_mixer",
            "context_film_seasonal_holiday",
            "level1_selection_attention",
            "use_cross_layers",
        )
    }
    out = dict(defaults)
    for key in (
        "output_activation",
        "trend_monotonic",
        "holiday_monotonic",
        "regressor_monotonic",
        "context_aware_component_mixer",
        "context_film_seasonal_holiday",
        "level1_selection_attention",
        "use_cross_layers",
    ):
        val = getattr(args, key, None)
        if val is None:
            continue
        if key == "output_activation":
            out[key] = str(val)
        else:
            out[key] = bool(int(val))
    return out


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


def train_seq_three_term(
    model, Xtr, ytr, skutr, Xva, yva, skuva, zero_rate, args, label, sku_zero_rates=None
):
    # Panel pos_weight in the loss; per-SKU via relative sample weights when rates given.
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
    sw_tr = sw_va = None
    if sku_zero_rates is not None:
        sw_tr = fit_bce_sample_weight_dict(
            ytr, skutr, sku_zero_rates, panel_zero_rate=zero_rate
        )
        sw_va = fit_bce_sample_weight_dict(
            yva, skuva, sku_zero_rates, panel_zero_rate=zero_rate
        )
    print(f"\n=== {label} (same features, three_term) ===")
    t0 = time.time()
    fit_kw = dict(
        x=[Xtr, skutr],
        y=ytr_d,
        validation_data=([Xva, skuva], yva_d, sw_va) if sw_va is not None else ([Xva, skuva], yva_d),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
            ),
        ],
        verbose=2,
    )
    if sw_tr is not None:
        fit_kw["sample_weight"] = sw_tr
    model.fit(**fit_kw)
    return time.time() - t0


def main():
    import os

    args = parse_args()
    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    unknown = selected - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"Unknown --models: {sorted(unknown)}. Choose from {ALL_MODELS}")
    need_seq = bool(selected & {"deepar_lite", "temporal_transformer", "tft_lite"})

    data_seed, train_seed = resolve_eval_seeds(
        args.seed, args.data_seed, args.train_seed
    )
    tf.keras.utils.set_random_seed(train_seed)
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

    chosen_list = select_eval_skus(
        train_df["id_var"].unique(),
        max_skus=args.max_skus,
        data_seed=data_seed,
        sku_list_path=args.sku_list,
        save_sku_list_path=args.save_sku_list,
    )
    chosen = set(chosen_list)
    print(
        f"Panel lock: data_seed={data_seed} train_seed={train_seed} "
        f"n_skus={len(chosen)}"
        + (f" sku_list={args.sku_list}" if args.sku_list else "")
    )
    train_df, h_tr = filter_aligned(train_df, h_tr, chosen)
    val_df, h_va = filter_aligned(val_df, h_va, chosen)
    test_df, h_te = filter_aligned(test_df, h_te, chosen)

    volume_map, volume_stats = train_volume_terciles(train_df)
    mase_scale = train_mase_scale(train_df, season=7)
    print("Volume terciles:")
    for b, st in volume_stats.items():
        print(
            f"  {b}: n={st['n_skus']} mean_vol={st['train_volume_mean_sku']:.1f} "
            f"zr={st['train_zero_rate']:.3f}"
        )
    print(f"MASE scale (train seasonal-naive |y_t-y_{{t-7}}| mean): {mase_scale}")

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

    merge_path = Path(args.merge_json) if args.merge_json else None
    if merge_path and merge_path.exists():
        results = json.loads(merge_path.read_text())
        results.setdefault("models", {})
        results.setdefault("config", {})
        results["config"]["models_run_this_pass"] = sorted(selected)
        results["config"]["note"] = (
            "Sequence models (DeepAR/TST/TFT) use full causal X in lookback "
            "(same columns as tabular). Holiday distance only (no is_* binaries)."
        )
    else:
        results = {
            "config": {
                "max_skus": args.max_skus,
                "n_skus": n_skus,
                "seed": args.seed,
                "data_seed": data_seed,
                "train_seed": train_seed,
                "sku_list": args.sku_list,
                "lookback": args.lookback,
                "zero_rate": zero_rate,
                "feature_contract": f"feature_config v{cfg.config['metadata']['version']} + Quantity in sequence hist",
                "n_tabular_features": len(feature_names),
                "feature_names": feature_names,
                "sequence_channels": ["Quantity"] + feature_names,
                "volume_stats": volume_stats,
                "models_run_this_pass": sorted(selected),
                "note": (
                    "Sequence models (DeepAR/TST/TFT) use full causal X in lookback "
                    "(same columns as tabular). Holiday distance only (no is_* binaries)."
                ),
            },
            "models": {},
        }

    tr, va, te = (
        split_components(X_train, cfg),
        split_components(X_val, cfg),
        split_components(X_test, cfg),
    )

    if "deepsequence" in selected:
        print("\n=== DeepSequence (same features) ===")
        train_cal = bool(args.ds_train_gate_calibrate)
        _, sku_rates = resolve_sku_zero_rates(y_train, sku_train, n_skus=n_skus)
        nz_target = max(1e-6, 1.0 - float(zero_rate))
        if train_cal:
            print(
                "  train-time gate calibration ON "
                f"(prior={zero_rate:.3f}, p_scale_init={args.ds_gate_prob_scale_init}, "
                f"rate_match_w={args.ds_gate_rate_match_weight})"
            )
        ds_stack = _ds_builder_kwargs(args)
        print(
            f"  per-SKU BCE imbalance ON "
            f"(panel_zr={zero_rate:.3f}, n_skus={n_skus})"
        )
        print(
            "  ds_stack="
            + ", ".join(f"{k}={v}" for k, v in ds_stack.items())
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
            use_intermittent=True,
            n_changepoints=15,
            gate_use_raw_regressors=train_cal,
            intermittent_prior_zero_rate=zero_rate if train_cal else None,
            intermittent_prior_zero_rates=sku_rates if train_cal else None,
            intermittent_learnable_logit_scale=train_cal,
            intermittent_logit_scale_init=1.0,
            gate_prob_scale=train_cal,
            gate_prob_scale_init=float(args.ds_gate_prob_scale_init),
            gate_prob_scale_trainable=True,
            gate_rate_match_weight=(
                float(args.ds_gate_rate_match_weight) if train_cal else 0.0
            ),
            gate_rate_match_target=nz_target if train_cal else None,
            **ds_stack,
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
            sku_zero_rates=sku_rates,
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
                    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
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
            "overall": kpi_block(y_test, yhat_ds, p_ds, mase_scale=mase_scale),
            "strata": strata_report(
                y_test, yhat_ds, p_ds, sku_test_raw, volume_map, mase_scale=mase_scale
            ),
        }

    if "lightgbm" in selected:
        print("\n=== LightGBM L1 (same features) ===")
        import lightgbm as lgb

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
            "overall": kpi_block(y_test, yhat_lgb, p_lgb, mase_scale=mase_scale),
            "strata": strata_report(
                y_test, yhat_lgb, p_lgb, sku_test_raw, volume_map, mase_scale=mase_scale
            ),
        }

    if need_seq:
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
        # Rates from tabular train targets (same SKU index space).
        _, seq_sku_rates = resolve_sku_zero_rates(y_train, sku_train, n_skus=n_skus)

        seq_builders = [
            ("deepar_lite", build_deepar),
            ("temporal_transformer", build_transformer),
            ("tft_lite", build_tft),
        ]
        for name, builder in seq_builders:
            if name not in selected:
                continue
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
                sku_zero_rates=seq_sku_rates,
            )
            yhat, p = predict_seq(model, Xseq[m_te], sku_seq[m_te])
            results["models"][name] = {
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

    results["mase_scale_season7"] = mase_scale
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
                    "iwmae_rounded": block.get("iwmae_rounded"),
                    "mase_rounded": block.get("mase_rounded"),
                    "occ_f1": block.get("occ_f1"),
                    "underforecast_rate_nonzero": block.get(
                        "underforecast_rate_nonzero"
                    ),
                    "aucroc": block.get("aucroc"),
                    "aucpr": block.get("aucpr"),
                    "bias": block.get("bias"),
                    "bias_nonzero": block.get("bias_nonzero"),
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
    for band in comparison:
        comparison[band] = sorted(
            comparison[band],
            key=lambda r: (
                r["iwmae_rounded"] is None,
                r["iwmae_rounded"] if r["iwmae_rounded"] is not None else 1e9,
            ),
        )
    results["comparison"] = comparison

    results["prior_calendar_only_sequence_reference"] = {
        "deepar_lite_mae_rounded": 1.571,
        "temporal_transformer_mae_rounded": 1.626,
        "note": "Earlier DeepAR/TST used [y,dow,month,doy] only — not same-feature",
    }

    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print("SAME-FEATURE COMPARISON (primary sort: iwmae_rounded)")
    print("=" * 70)
    for band in ("overall", "low", "mid", "high"):
        print(f"\n[{band}]")
        for row in comparison[band]:
            nv = row.get("inventory_nv_cost_rounded_cu2")
            nv_s = f"{nv:.3f}" if nv is not None else "n/a"
            print(
                f"  {row['model']:32s} iwmae={row['iwmae_rounded']:.3f} "
                f"nv_cu2={nv_s} "
                f"mae={row['mae_rounded']:.3f} nz={row['mae_nonzero']:.3f} "
                f"mase={row.get('mase_rounded')} occ_f1={row.get('occ_f1')} "
                f"under={row.get('underforecast_rate_nonzero')} "
                f"auc={row.get('aucroc')} bias={row['bias']:.3f}"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
