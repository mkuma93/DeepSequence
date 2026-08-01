#!/usr/bin/env python3
"""Dump interpretable DeepSequence components for a few SKUs (JSON sample).

Trains a small one-step DS model on a daily panel subset and writes per-row
``trend``, ``seasonal``, ``holiday``, ``regressor``, ``component_alpha``,
``p``, ``b``, ``yhat`` under ``--out_json``.

Recursive MH: components are one-step only; this script dumps the one-step
readout. For MH, call ``predict_with_components`` at each rollout step.

Example::

  TF_USE_LEGACY_KERAS=1 .venv-test/bin/python -m deepsequence_hierarchical_attention.eval.dump_component_readout \\
    --data_dir "$DEEPSEQUENCE_DATA_DIR" --max_skus 8 --epochs 3 \\
    --out_json ab_runs/reclaim/component_readout_sample.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]

from deepsequence_hierarchical_attention import (  # noqa: E402
    build_hierarchical_model_lightweight,
    predict_with_components,
)
from deepsequence_hierarchical_attention.losses import three_term_loss_config  # noqa: E402
from deepsequence_hierarchical_attention.eval.helpers import select_eval_skus, split_components  # noqa: E402
from deepsequence_hierarchical_attention.data.feature_config_loader import load_feature_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--feature_config", default=str(ROOT / "feature_config.yaml"))
    p.add_argument("--max_skus", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_sample_rows", type=int, default=60)
    p.add_argument(
        "--out_json",
        default=str(ROOT / "ab_runs/reclaim/component_readout_sample.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(
        args.data_dir
        or os.environ.get(
            "DEEPSEQUENCE_DATA_DIR",
            "/Users/mritunjaykumar/Library/CloudStorage/"
            "GoogleDrive-mritunjay.kmr1@gmail.com/My Drive/jubilant/data",
        )
    )
    cfg = load_feature_config(args.feature_config)
    train_df = pd.read_csv(data_dir / "train_split.csv")
    val_df = pd.read_csv(data_dir / "val_split.csv")
    test_df = pd.read_csv(data_dir / "test_split.csv")
    h_tr = pd.read_csv(data_dir / "holiday_features_train.csv")
    h_va = pd.read_csv(data_dir / "holiday_features_val.csv")
    h_te = pd.read_csv(data_dir / "holiday_features_test.csv")

    universe = sorted(set(train_df["id_var"].astype(str)))
    skus = select_eval_skus(universe, max_skus=args.max_skus, data_seed=args.seed)
    sku_set = set(str(s) for s in skus)

    def _mask(df, hol):
        m = df["id_var"].astype(str).isin(sku_set).to_numpy()
        return df.loc[m].reset_index(drop=True), hol.loc[m].reset_index(drop=True)

    train_df, h_tr = _mask(train_df, h_tr)
    val_df, h_va = _mask(val_df, h_va)
    test_df, h_te = _mask(test_df, h_te)

    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(
        val_df, h_va, prior_states=states, return_states=True
    )
    Xte_df, _ = cfg.create_features(
        test_df, h_te, prior_states=states, return_states=True
    )

    # Align y / sku with feature row order (create_features sorts by id, ds)
    def _aligned_y_sku(df):
        d = df.copy()
        d["ds"] = pd.to_datetime(d["ds"])
        d = d.sort_values(["id_var", "ds"], kind="mergesort").reset_index(drop=True)
        return d["Quantity"].to_numpy(np.float32), d["id_var"].astype(str).to_numpy()

    ytr, sku_tr = _aligned_y_sku(train_df)
    yva, sku_va = _aligned_y_sku(val_df)
    yte, sku_te = _aligned_y_sku(test_df)
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)

    sku_map = {s: i for i, s in enumerate(sorted(sku_set))}
    sid_tr = np.array([[sku_map[s]] for s in sku_tr], np.int32)
    sid_va = np.array([[sku_map[s]] for s in sku_va], np.int32)
    sid_te = np.array([[sku_map[s]] for s in sku_te], np.int32)

    tr, se, ho, la = split_components(X_train, cfg)
    trv, sev, hov, lav = split_components(X_val, cfg)
    trt, set_, hot, lat = split_components(X_test, cfg)

    # Scale trend like bake-off
    tmin, tmax = float(tr.min()), float(tr.max())
    span = max(tmax - tmin, 1e-6)
    for arr in (tr, trv, trt):
        arr[:] = (arr - tmin) / span

    zr = float(np.mean(ytr <= 0))
    model = build_hierarchical_model_lightweight(
        n_temporal_features=tr.shape[1],
        n_fourier_features=se.shape[1],
        n_holiday_features=ho.shape[1],
        n_lag_features=la.shape[1],
        n_skus=len(sku_map),
        hidden_dim=32,
        use_cross_layers=False,
        use_intermittent=True,
        n_changepoints=10,
        horizon=1,
    )
    loss_cfg = three_term_loss_config(zero_rate=zr)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_cfg["losses"],
        loss_weights=loss_cfg.get("weights"),
    )
    y_dict = {
        "final_forecast": ytr.reshape(-1, 1),
        "base_forecast": ytr.reshape(-1, 1),
        "non_zero_probability": (ytr > 0).astype(np.float32).reshape(-1, 1),
    }
    y_val = {
        "final_forecast": yva.reshape(-1, 1),
        "base_forecast": yva.reshape(-1, 1),
        "non_zero_probability": (yva > 0).astype(np.float32).reshape(-1, 1),
    }
    model.fit(
        [tr, se, ho, la, sid_tr],
        y_dict,
        validation_data=([trv, sev, hov, lav, sid_va], y_val),
        epochs=int(args.epochs),
        batch_size=256,
        verbose=0,
    )

    rng = np.random.default_rng(args.seed)
    idx = []
    for s in list(skus)[: min(3, len(skus))]:
        rows = np.where(sku_te == str(s))[0]
        if len(rows) == 0:
            continue
        take = min(20, len(rows))
        idx.extend(rng.choice(rows, size=take, replace=False).tolist())
    idx = np.asarray(idx[: args.n_sample_rows], dtype=int)
    xb = [trt[idx], set_[idx], hot[idx], lat[idx], sid_te[idx]]
    comps = predict_with_components(model, xb, verbose=0)

    rows = []
    for i, ridx in enumerate(idx):
        rows.append(
            {
                "sku": str(sku_te[ridx]),
                "y": float(yte[ridx]),
                "yhat": float(comps["final_forecast"][i].reshape(-1)[0]),
                "p": float(comps["non_zero_probability"][i].reshape(-1)[0]),
                "b": float(comps["base_forecast"][i].reshape(-1)[0]),
                "trend": float(comps["trend"][i].reshape(-1)[0]),
                "seasonal": float(comps["seasonal"][i].reshape(-1)[0]),
                "holiday": float(comps["holiday"][i].reshape(-1)[0]),
                "regressor": float(comps["regressor"][i].reshape(-1)[0]),
                "component_alpha": comps["component_alpha"][i].reshape(-1).tolist(),
                "mixed_contribution": {
                    k: float(comps[f"mixed_contribution_{k}"][i].reshape(-1)[0])
                    for k in ("trend", "seasonal", "holiday", "regressor")
                },
            }
        )

    payload = {
        "note": (
            "Expert scalars are post Level-1 / softsign / FiLM (values mixed by "
            "Level-2). yhat=p*b. One-step only; recursive MH should probe each step."
        ),
        "skus": [str(s) for s in skus],
        "n_rows": len(rows),
        "rows": rows,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
