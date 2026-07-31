"""
Causal residual transformer head for intermittent demand.

Design contract
---------------
1. Holiday / seasonal / regressor are absorbed into lag-based DeepSequence
   ``y_struct`` / ``base_forecast`` (do not strip lags from DS).
2. DeepSequence gate ``p_ds`` is **preserved**, not re-learned:
   - ``p_ds`` is carried at every lookback step
   - final forecast uses the same DS probability at predict step ``t``
   - optional soft encoder mix (default off) can softly scale hidden states
     by ``p_ds``; a hard TD multiply zeros capacity on quiet days and hurts IWMAE

    residual = y - y_struct
    h_τ      = Encoder(seq)_τ                   # optional soft * p_ds
    delta    = Head(h_t)                        # zero-init → identity start
    base     = relu(y_struct_t + delta)
    yhat     = base * p_ds_t                   # DS gate kept

Default sequence channels:

    [y_struct, y_masked_at_t, residual_masked_at_t, p_ds]

``y`` / residual are masked at the predict step; ``p_ds`` is never masked.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

DEFAULT_SEQUENCE_CHANNELS = (
    "y_struct",
    "y_masked_at_t",
    "residual_masked_at_t",
    "p_ds",
)

Y_CHANNEL_INDEX = 1
RESIDUAL_CHANNEL_INDEX = 2
P_DS_CHANNEL_INDEX = 3

DEFAULT_CHANNEL_COLS = ("y_struct", "y", "resid", "p_ds")


@tf.keras.utils.register_keras_serializable(
    package="deepsequence_hierarchical_attention"
)
class SoftEncoderGateMix(tf.keras.layers.Layer):
    """x ← x * (1 - mix + mix * p); mix=0 leaves the encoder ungated."""

    def __init__(self, mix: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def call(self, inputs):
        x, p = inputs
        if self.mix <= 0.0:
            return x
        scale = (1.0 - self.mix) + self.mix * p
        return x * scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"mix": self.mix})
        return cfg


@tf.keras.utils.register_keras_serializable(
    package="deepsequence_hierarchical_attention"
)
class TakeLastTimestep(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, -1, :]

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(
    package="deepsequence_hierarchical_attention"
)
class PassThrough(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs

    def get_config(self):
        return super().get_config()


def build_residual_transformer(
    lookback: int,
    n_channels: int,
    n_skus: int,
    d_model: int = 32,
    n_heads: int = 4,
    n_blocks: int = 1,
    sku_emb_dim: int = 8,
    p_ds_channel_index: int = P_DS_CHANNEL_INDEX,
    preserve_ds_gate: bool = True,
    encoder_gate_mix: float = 0.0,
    name: str = "residual_transformer",
) -> tf.keras.Model:
    """
    Causal residual transformer with optional DS-gate preservation.

    When ``preserve_ds_gate=True`` (default):
      - extracts ``p_ds`` sequence from ``seq[..., p_ds_channel_index]``
      - final = relu(y_struct + delta) * p_ds_t  (no new sigmoid gate)
      - ``encoder_gate_mix`` (default 0) optionally soft-scales hidden states;
        hard TD multiply is intentionally *not* the default

    When ``preserve_ds_gate=False`` (legacy): learns a fresh sigmoid gate.
    """
    hist = tf.keras.Input(shape=(lookback, n_channels), name="seq")
    sku = tf.keras.Input(shape=(1,), dtype=tf.int32, name="sku")
    y_struct_t = tf.keras.Input(shape=(1,), name="y_struct_t")

    emb = tf.keras.layers.Embedding(n_skus, sku_emb_dim)(sku)
    emb = tf.keras.layers.Flatten()(emb)
    emb_t = tf.keras.layers.RepeatVector(lookback)(emb)
    x = tf.keras.layers.Concatenate(axis=-1)([hist, emb_t])
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    if preserve_ds_gate:
        p_seq = hist[:, :, p_ds_channel_index : p_ds_channel_index + 1]
        x = SoftEncoderGateMix(mix=encoder_gate_mix, name="soft_encoder_gate")(
            [x, p_seq]
        )
        p_t = TakeLastTimestep(name="p_ds_t")(p_seq)
    else:
        p_t = None

    key_dim = max(1, d_model // n_heads)
    for b in range(max(1, int(n_blocks))):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=key_dim,
            name=f"mha_{b}",
        )(x, x, use_causal_mask=True)
        x = tf.keras.layers.LayerNormalization(name=f"ln_attn_{b}")(x + attn)
        ff = tf.keras.layers.Dense(d_model * 2, activation="relu", name=f"ff_up_{b}")(x)
        ff = tf.keras.layers.Dense(d_model, name=f"ff_down_{b}")(ff)
        x = tf.keras.layers.LayerNormalization(name=f"ln_ff_{b}")(x + ff)

    h = TakeLastTimestep(name="readout_t")(x)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    # Zero-init delta so training starts at the lag-based DeepSequence forecast.
    delta = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="delta",
    )(h)
    base = tf.keras.layers.Add(name="base_forecast")([y_struct_t, delta])
    base = tf.keras.layers.Activation("relu", name="base_relu")(base)

    if preserve_ds_gate:
        p = PassThrough(name="non_zero_probability")(p_t)
    else:
        p = tf.keras.layers.Dense(1, activation="sigmoid", name="non_zero_probability")(h)

    final = tf.keras.layers.Multiply(name="final_forecast")([base, p])

    return tf.keras.Model(
        [hist, sku, y_struct_t],
        {
            "final_forecast": final,
            "base_forecast": base,
            "non_zero_probability": p,
            "delta": delta,
        },
        name=name,
    )


class ResidualTrainModel(tf.keras.Model):
    """
    Train wrapper for residual transformer.

    Default (preserve DS gate): weighted gated MAE + nonzero magnitude MAE.
    BCE is off (``alpha_bce=0``) because ``p`` comes from DeepSequence.
    """

    def __init__(
        self,
        base: tf.keras.Model,
        zero_rate: float,
        alpha_bce: float = 0.0,
        w_gated: float = 1.0,
        w_mag: float = 1.0,
    ):
        super().__init__()
        self.base = base
        self.alpha_bce = float(alpha_bce)
        self.w_gated = float(w_gated)
        self.w_mag = float(w_mag)
        zr = float(zero_rate)
        nz = max(1.0 - zr, 1e-6)
        self.w_zero = 1.0 / (2.0 * max(zr, 1e-6))
        self.w_nonzero = 1.0 / (2.0 * nz)
        self.pos_weight = min(20.0, zr / nz)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def call(self, inputs, training=None):
        return self.base(inputs, training=training)

    def _loss(self, y_true, out):
        y = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        yhat = tf.reshape(out["final_forecast"], [-1, 1])
        base = tf.reshape(out["base_forecast"], [-1, 1])
        p = tf.reshape(out["non_zero_probability"], [-1, 1])
        yb = tf.cast(y > 0, tf.float32)
        w = self.w_zero * (1.0 - yb) + self.w_nonzero * yb
        gated = tf.reduce_sum(w * tf.abs(y - yhat)) / tf.reduce_sum(w)
        mag = tf.reduce_sum(yb * tf.abs(y - base)) / (tf.reduce_sum(yb) + 1e-6)
        loss = self.w_gated * gated + self.w_mag * mag
        if self.alpha_bce > 0.0:
            pc = tf.clip_by_value(p, 1e-7, 1 - 1e-7)
            bce = tf.reduce_mean(
                -self.pos_weight * yb * tf.math.log(pc)
                - (1.0 - yb) * tf.math.log(1.0 - pc)
            )
            loss = loss + self.alpha_bce * bce
        return loss

    def train_step(self, data):
        x, y = data[0], data[1]
        y_true = y["final_forecast"]
        with tf.GradientTape() as tape:
            out = self.base(x, training=True)
            loss = self._loss(y_true, out)
            if self.base.losses:
                loss = loss + tf.add_n(self.base.losses)
        vars_ = self.base.trainable_variables
        grads = tape.gradient(loss, vars_)
        fixed = [
            None if g is None else tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            for g in grads
        ]
        fixed, _ = tf.clip_by_global_norm(fixed, 5.0)
        self.optimizer.apply_gradients(zip(fixed, vars_))
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(y_true, out["final_forecast"])
        return {"loss": self.loss_tracker.result(), "mae": self.mae_tracker.result()}

    def test_step(self, data):
        x, y = data[0], data[1]
        y_true = y["final_forecast"]
        out = self.base(x, training=False)
        loss = self._loss(y_true, out)
        if self.base.losses:
            loss = loss + tf.add_n(self.base.losses)
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(y_true, out["final_forecast"])
        return {"loss": self.loss_tracker.result(), "mae": self.mae_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_tracker]


def mask_predict_step(
    hist: np.ndarray,
    y_idx: int = Y_CHANNEL_INDEX,
    resid_idx: int = RESIDUAL_CHANNEL_INDEX,
) -> np.ndarray:
    """Zero ``y`` and residual on the last timestep; leave ``p_ds`` intact."""
    out = np.asarray(hist, dtype=np.float32).copy()
    out[..., -1, y_idx] = 0.0
    out[..., -1, resid_idx] = 0.0
    return out


def build_residual_windows(
    panel: pd.DataFrame,
    lookback: int,
    channel_cols: Sequence[str] = DEFAULT_CHANNEL_COLS,
    id_col: str = "id_var",
    time_col: str = "ds",
    y_col: str = "y",
    y_struct_col: str = "y_struct",
    p_ds_col: str = "p_ds",
    split_col: Optional[str] = "split",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
]:
    """
    Build causal lookback windows **per SKU** (no cross-SKU mixing).

    Requires ``p_ds`` (DeepSequence non-zero probability) when using the default
    channel set, so the residual head can TimeDistributed-multiply by it.

    Returns
    -------
    X, y, y_struct, p_ds, sku_ids, splits
    """
    df = panel.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    if "resid" not in df.columns and "resid" in channel_cols:
        df["resid"] = df[y_col].astype(np.float32) - df[y_struct_col].astype(np.float32)
    if p_ds_col not in df.columns and p_ds_col in channel_cols:
        raise ValueError(
            f"panel is missing '{p_ds_col}'. Pass DeepSequence gate probabilities "
            "so the residual transformer can preserve DS intermittent structure."
        )
    df = df.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)

    Xs, yt, yst, pt, skus, splits = [], [], [], [], [], []
    has_split = split_col is not None and split_col in df.columns
    has_p = p_ds_col in df.columns

    for sku, g in df.groupby(id_col, sort=False):
        arr = g[list(channel_cols)].to_numpy(np.float32)
        y = g[y_col].to_numpy(np.float32)
        ys = g[y_struct_col].to_numpy(np.float32)
        pp = g[p_ds_col].to_numpy(np.float32) if has_p else np.ones(len(g), np.float32)
        sp = g[split_col].to_numpy() if has_split else None
        n = len(g)
        if n < lookback:
            continue
        for t in range(lookback - 1, n):
            start = t - lookback + 1
            hist = mask_predict_step(arr[start : t + 1])
            Xs.append(hist)
            yt.append(y[t])
            yst.append(ys[t])
            pt.append(pp[t])
            skus.append(sku)
            if has_split:
                splits.append(sp[t])

    X = (
        np.stack(Xs).astype(np.float32)
        if Xs
        else np.zeros((0, lookback, len(channel_cols)), np.float32)
    )
    return (
        X,
        np.asarray(yt, np.float32),
        np.asarray(yst, np.float32),
        np.asarray(pt, np.float32),
        np.asarray(skus),
        np.asarray(splits) if has_split else None,
    )


def train_residual_transformer(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_struct_train: np.ndarray,
    sku_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_struct_val: np.ndarray,
    sku_val: np.ndarray,
    zero_rate: float,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 0.002,
    alpha_bce: float = 0.0,
    w_gated: float = 1.0,
    w_mag: float = 1.0,
    verbose: int = 2,
) -> ResidualTrainModel:
    """Fit residual transformer; default loss keeps DS gate (no BCE on p)."""
    wrapped = ResidualTrainModel(
        model,
        zero_rate,
        alpha_bce=alpha_bce,
        w_gated=w_gated,
        w_mag=w_mag,
    )
    wrapped.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
    ]
    wrapped.fit(
        [X_train, sku_train, y_struct_train.reshape(-1, 1)],
        {"final_forecast": y_train.reshape(-1, 1)},
        validation_data=(
            [X_val, sku_val, y_struct_val.reshape(-1, 1)],
            {"final_forecast": y_val.reshape(-1, 1)},
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return wrapped


def predict_residual_transformer(
    model: tf.keras.Model,
    X: np.ndarray,
    y_struct: np.ndarray,
    sku: np.ndarray,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(final, p_ds, base, delta)`` as 1-D float arrays."""
    out = model.predict(
        [X, sku, np.asarray(y_struct, np.float32).reshape(-1, 1)],
        batch_size=batch_size,
        verbose=0,
    )
    return (
        np.asarray(out["final_forecast"]).reshape(-1),
        np.asarray(out["non_zero_probability"]).reshape(-1),
        np.asarray(out["base_forecast"]).reshape(-1),
        np.asarray(out["delta"]).reshape(-1),
    )


__all__ = [
    "DEFAULT_SEQUENCE_CHANNELS",
    "DEFAULT_CHANNEL_COLS",
    "Y_CHANNEL_INDEX",
    "RESIDUAL_CHANNEL_INDEX",
    "P_DS_CHANNEL_INDEX",
    "build_residual_transformer",
    "ResidualTrainModel",
    "mask_predict_step",
    "build_residual_windows",
    "train_residual_transformer",
    "predict_residual_transformer",
]
