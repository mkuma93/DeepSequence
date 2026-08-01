#!/usr/bin/env python3
"""Bias debug: does restoring MaskedEntropyAttention's softplus scale recover
the published lower mean_p / bias on the locked daily panel?

Also logs per-epoch val mean_p, bias, recall — to see if loss converges while
occurrence stays over-aggressive.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepsequence_hierarchical_attention.data.feature_config_loader import load_feature_config  # noqa: E402
from deepsequence_hierarchical_attention.eval.helpers import (  # noqa: E402
    filter_aligned,
    kpi_block,
    select_eval_skus,
    split_components,
)
from deepsequence_hierarchical_attention.components_lightweight import (  # noqa: E402
    MaskedEntropyAttention,
    build_hierarchical_model_lightweight,
)
from deepsequence_hierarchical_attention.training.adaptive_loss import (  # noqa: E402
    AdaptiveWeightedModel,
    WeightedBCELoss,
)

DATA = Path(
    "/Users/mritunjaykumar/Library/CloudStorage/GoogleDrive-mritunjay.kmr1@gmail.com/"
    "My Drive/jubilant/data"
)
SKU_LIST = ROOT / "ab_runs/recompare/sku_list_daily_data42.json"
OUT = ROOT / "ab_runs/bias_debug"
EPOCHS = 8
BATCH = 1024
SEED = 42

_ORIG_BUILD = MaskedEntropyAttention.build
_ORIG_CALL = MaskedEntropyAttention.call


def restore_old_entropy_scale():
    """Re-introduce trainable softplus scale (published behavior)."""

    def build(self, input_shape):
        _ORIG_BUILD(self, input_shape)
        self.raw_entropy_scale = self.add_weight(
            name=f"{self.name}_raw_entropy_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.01),
            trainable=True,
            regularizer=tf.keras.regularizers.l2(1e-6),
        )

    def call(self, inputs, training=None):
        x = self.layer_norm(inputs)
        scores = self.attention_dense(x)
        logits = self.attention_scale * tf.tanh(scores)
        temp = tf.maximum(
            tf.constant(0.3, dtype=tf.float32),
            tf.constant(self.temperature, dtype=tf.float32),
        )
        logits = logits / temp
        weights = tf.nn.softmax(logits, axis=-1)
        attended = inputs * weights
        output = self.projection(attended)
        output = self.dropout(output, training=training)
        entropy = -tf.reduce_sum(
            weights * tf.math.log(weights + 1e-8), axis=-1
        )
        present_scalar = tf.cast(self.present_value, tf.float32)
        entropy_scale = tf.nn.softplus(self.raw_entropy_scale)
        self.add_loss(
            present_scalar
            * self.entropy_weight
            * entropy_scale
            * tf.reduce_mean(entropy)
        )
        return output

    MaskedEntropyAttention.build = build
    MaskedEntropyAttention.call = call


def use_current_entropy():
    MaskedEntropyAttention.build = _ORIG_BUILD
    MaskedEntropyAttention.call = _ORIG_CALL


class EpochProbe(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, rows):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.rows = rows

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x_val, batch_size=4096, verbose=0)
        yhat = np.asarray(pred["final_forecast"]).reshape(-1)
        p = np.asarray(pred["non_zero_probability"]).reshape(-1)
        y = self.y_val.reshape(-1)
        bias = float(yhat.mean() - y.mean())
        mean_p = float(p.mean())
        mae = float(np.abs(yhat - y).mean())
        # occurrence at 0.5
        pred_pos = p >= 0.5
        true_pos = y > 0
        recall = float((pred_pos & true_pos).sum() / max(true_pos.sum(), 1))
        precision = float((pred_pos & true_pos).sum() / max(pred_pos.sum(), 1))
        row = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", np.nan)),
            "val_loss": float(logs.get("val_loss", np.nan)),
            "val_mean_p": mean_p,
            "val_bias": bias,
            "val_mae": mae,
            "val_occ_recall@0.5": recall,
            "val_occ_precision@0.5": precision,
        }
        self.rows.append(row)
        print(
            f"  probe ep{epoch+1}: val_loss={row['val_loss']:.3f} "
            f"mean_p={mean_p:.3f} bias={bias:+.3f} mae={mae:.3f} "
            f"rec@0.5={recall:.3f} prec@0.5={precision:.3f}"
        )


def run_arm(arm: str):
    print(f"\n{'='*72}\nARM={arm}\n{'='*72}")
    if arm == "old_entropy_scale":
        restore_old_entropy_scale()
    else:
        use_current_entropy()

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    train_df = pd.read_csv(DATA / "train_split.csv", parse_dates=["ds"])
    val_df = pd.read_csv(DATA / "val_split.csv", parse_dates=["ds"])
    test_df = pd.read_csv(DATA / "test_split.csv", parse_dates=["ds"])
    h_tr = pd.read_csv(DATA / "holiday_features_train.csv")
    h_va = pd.read_csv(DATA / "holiday_features_val.csv")
    h_te = pd.read_csv(DATA / "holiday_features_test.csv")
    chosen = set(
        select_eval_skus(
            train_df["id_var"].unique(),
            max_skus=800,
            data_seed=42,
            sku_list_path=str(SKU_LIST),
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

    y_train = train_df["Quantity"].to_numpy(np.float32)
    y_val = val_df["Quantity"].to_numpy(np.float32)
    y_test = test_df["Quantity"].to_numpy(np.float32)
    sku_train, sku_val, sku_test = enc(train_df), enc(val_df), enc(test_df)
    zero_rate = float((y_train == 0).mean())

    cfg = load_feature_config()
    Xtr_df, states = cfg.create_features(train_df, h_tr, return_states=True)
    Xva_df, states = cfg.create_features(
        val_df, h_va, prior_states=states, return_states=True
    )
    Xte_df, _ = cfg.create_features(
        test_df, h_te, prior_states=states, return_states=True
    )
    X_train = Xtr_df.to_numpy(np.float32)
    X_val = Xva_df.to_numpy(np.float32)
    X_test = Xte_df.to_numpy(np.float32)
    t_idx = cfg.trend_indices[0]
    tmin, tmax = float(X_train[:, t_idx].min()), float(X_train[:, t_idx].max())
    span = max(tmax - tmin, 1.0)
    for X in (X_train, X_val, X_test):
        X[:, t_idx] = (X[:, t_idx] - tmin) / span

    tr = split_components(X_train, cfg)
    va = split_components(X_val, cfg)
    te = split_components(X_test, cfg)

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
    n_entropy_scale = sum(
        1 for w in base.trainable_weights if "entropy_scale" in w.name
    )
    # Measure regularization mass at init
    _ = base(
        [*(x[:256] for x in tr), sku_train[:256]], training=True
    )
    reg = float(tf.add_n(base.losses).numpy()) if base.losses else 0.0
    print(f"entropy_scale weights={n_entropy_scale}  init_add_loss_sum={reg:.4f}")
    print(f"n_trainable={len(base.trainable_weights)}")

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

    rows = []
    ytr = {
        "final_forecast": y_train.reshape(-1, 1),
        "base_forecast": y_train.reshape(-1, 1),
    }
    yva = {
        "final_forecast": y_val.reshape(-1, 1),
        "base_forecast": y_val.reshape(-1, 1),
    }
    t0 = time.time()
    hist = model.fit(
        [*tr, sku_train],
        ytr,
        validation_data=([*va, sku_val], yva),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[EpochProbe([*va, sku_val], y_val, rows)],
        verbose=2,
    )
    train_s = time.time() - t0

    pred = model.predict([*te, sku_test], batch_size=4096, verbose=0)
    yhat = np.asarray(pred["final_forecast"]).reshape(-1)
    p = np.asarray(pred["non_zero_probability"]).reshape(-1)
    overall = kpi_block(y_test, yhat, p, mase_scale=1.0)
    # mase_scale unused for iwmae; fine

    payload = {
        "arm": arm,
        "epochs": EPOCHS,
        "entropy_scale_weights": n_entropy_scale,
        "init_add_loss_sum": reg,
        "n_trainable": len(base.trainable_weights),
        "train_seconds": train_s,
        "history": hist.history,
        "val_probes": rows,
        "test": {
            "iwmae": overall["iwmae"],
            "iwmae_rounded": overall["iwmae_rounded"],
            "mae_all_rounded": overall["mae_all_rounded"],
            "mean_p": overall["mean_p"],
            "bias": overall["bias"],
            "occ_f1": overall["occ_f1"],
            "underforecast_rate_nonzero": overall["underforecast_rate_nonzero"],
        },
    }
    out = OUT / f"{arm}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"TEST {arm}: iwmae_r={overall['iwmae_rounded']:.4f} "
        f"mean_p={overall['mean_p']:.4f} bias={overall['bias']:+.4f} "
        f"occ_f1={overall['occ_f1']:.4f}"
    )
    print(f"wrote {out}")
    use_current_entropy()
    return payload


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cur = run_arm("current_full_entropy")
    old = run_arm("old_entropy_scale")
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for label, p in [("current", cur), ("old_scale", old)]:
        t = p["test"]
        print(
            f"{label:10s} iwmae_r={t['iwmae_rounded']:.4f} mean_p={t['mean_p']:.3f} "
            f"bias={t['bias']:+.3f} occ_f1={t['occ_f1']:.3f} "
            f"init_reg={p['init_add_loss_sum']:.4f} n_ent_scale={p['entropy_scale_weights']}"
        )
    print("\nPer-epoch val bias / mean_p:")
    print(f"{'ep':>3} {'cur_p':>7} {'old_p':>7} {'cur_bias':>9} {'old_bias':>9} {'cur_vloss':>9} {'old_vloss':>9}")
    for a, b in zip(cur["val_probes"], old["val_probes"]):
        print(
            f"{a['epoch']:3d} {a['val_mean_p']:7.3f} {b['val_mean_p']:7.3f} "
            f"{a['val_bias']:+9.3f} {b['val_bias']:+9.3f} "
            f"{a['val_loss']:9.3f} {b['val_loss']:9.3f}"
        )


if __name__ == "__main__":
    main()
