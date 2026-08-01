"""
Lightweight Model Training with Adaptive Loss Weighting (Uncertainty-based)
Uses learned uncertainty parameters (log-variance) to automatically balance:
- Zero/non-zero classification (BCE)
- Forecast accuracy (MAE on final_forecast)

Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)
Loss = (1/(2*σ²)) * task_loss + log(σ)

IMPORTANT: This matches the latest framework approach:
- Optimizes final_forecast (base * decision) against true y
- No non-zero masking on MAE (optimizes all values)
- This directly optimizes what we care about (actual forecast accuracy)

This is an experimental alternative to fixed loss weights.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import argparse
import traceback

from deepsequence_hierarchical_attention.components_lightweight import (
    build_hierarchical_model_lightweight,
    estimate_zero_rate_by_sku,
    multioutput_bce_sample_weight_dict,
)
from deepsequence_hierarchical_attention.losses import (
    composite_loss,
    three_term_loss_config,
    bce_mae_loss_config,
    spike_aware_loss_config,
    resolve_positive_bce_weight,
)
from deepsequence_hierarchical_attention.data.feature_config_loader import load_feature_config


class AdaptiveLossWeighting(keras.layers.Layer):
    """
    Learns task-specific uncertainty (log-variance) for adaptive weighting.
    
    Each task gets a learnable log-variance parameter σ:
    - Higher uncertainty → lower task weight
    - Lower uncertainty → higher task weight
    - Regularization term (log σ) prevents σ from going to infinity
    
    Formula: L_total = Σ (L_i / (2*exp(log_var_i))) + (log_var_i / 2)
    Simplified: L_total = Σ (L_i * exp(-log_var_i)) + log_var_i
    """
    
    def __init__(self, num_tasks=2, name='adaptive_loss_weighting'):
        super().__init__(name=name)
        self.num_tasks = num_tasks
        # Ensure classification (task 0) retains a minimum share
        self.min_weight_bce = 0.30
        
    def build(self, input_shape):
        # Initialize log-variance parameters (start near 0 = equal weighting)
        self.log_vars = self.add_weight(
            name='log_vars',
            shape=(self.num_tasks,),
            initializer=keras.initializers.Constant(0.0),
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, losses):
        """
        Args:
            losses: List/tuple of task losses [bce_loss, mae_loss]
        Returns:
            Weighted total loss
        """
        # Uncertainty weighting: precision = exp(-log_var)
        # Total loss = precision * loss + regularization
        # Compute precisions for each task
        precisions = tf.exp(-self.log_vars)
        # Normalize to get current weights
        sum_p = tf.reduce_sum(precisions)
        weights = precisions / (sum_p + 1e-8)
        # Enforce minimum classification weight floor (index 0 assumed BCE)
        floor = tf.constant(self.min_weight_bce, dtype=weights.dtype)
        bce_w = weights[0]
        def _apply_floor():
            # Increase BCE precision to meet floor, scale others proportionally
            desired_bce = floor
            other_sum = tf.reduce_sum(weights[1:])
            # If other_sum is zero, just set BCE to 1
            scaled_others = tf.where(other_sum > 0,
                                     weights[1:] * ((1.0 - desired_bce) / other_sum),
                                     tf.zeros_like(weights[1:]))
            new_weights = tf.concat([[desired_bce], scaled_others], axis=0)
            # Map weights back to adjusted precisions (keep total sum equal to sum_p)
            adj_precisions = new_weights * sum_p
            return adj_precisions
        adj_precisions = tf.cond(bce_w < floor, _apply_floor, lambda: precisions)

        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = adj_precisions[i]
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        return tf.add_n(weighted_losses)
    
    def get_weights_summary(self):
        """Get current task weights as percentages."""
        # Handle case where layer wasn't built (e.g., when using fixed weights)
        if not hasattr(self, 'log_vars') or self.log_vars is None:
            return {
                'log_vars': None,
                'precisions': None,
                'weights_pct': None,
                'note': 'Using fixed weights in wrapper'
            }
        log_vars = self.log_vars.numpy()
        precisions = np.exp(-log_vars)
        weights = precisions / precisions.sum()
        return {
            'log_vars': log_vars,
            'precisions': precisions,
            'weights_pct': weights * 100
        }


class AdaptiveWeightedModel(keras.Model):
    """
    Wrapper model with selectable intermittent loss recipes.

    loss_recipe:
      - "legacy": BCE + MAE(final) on nonzero days only (old adaptive path)
      - "bce_mae": BCE + MAE(final) on all days (shape-safe)
      - "three_term": inverse-weighted gated MAE + nonzero mag MAE + light BCE
        (recommended default bake-off)
      - "spike_aware": heavy positive BCE (+ optional focal) + positive-only
        magnitude on ``b`` with optional small zero-day mag weight (opt-in)

    Spike-aware knobs (only used when ``loss_recipe='spike_aware'``):
      - ``positive_bce_weight`` / ``positive_bce_boost``: heavier y>0 BCE
      - ``focal_gamma``: focal focusing (0 = off)
      - ``zero_mag_weight``: quiet-day magnitude calibration on ``b``
      - ``use_mse_mag``: MSE instead of MAE for magnitude

    When ``sku_zero_rates`` is provided (length ``n_skus``), the BCE / occurrence
    term uses per-SKU ``pos_weight`` looked up from the last model input (SKU
    ids). Multi-horizon targets broadcast the same SKU weight across the
    horizon after flatten. Panel ``pos_weight`` remains the fallback when rates
    are omitted. For spike-aware, SKU weights are scaled so the panel mean
    matches ``positive_bce_weight`` (or boosted default).
    """
    
    def __init__(self, base_model, bce_loss_fn, mae_loss_fn, 
                 zero_rate, avg_nonzero_demand, pos_weight,
                 use_fixed_weights=False, bce_weight=0.5, mae_weight=0.5,
                 loss_recipe="legacy",
                 alpha_bce=0.2, w_gated=1.0, w_mag=1.0,
                 horizon_decay=1.0,
                 sku_zero_rates=None,
                 pos_weight_cap=20.0,
                 positive_bce_weight=None,
                 positive_bce_boost=2.0,
                 focal_gamma=0.0,
                 zero_mag_weight=0.05,
                 use_mse_mag=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.bce_loss_fn = bce_loss_fn
        self.mae_loss_fn = mae_loss_fn
        self.use_fixed_weights = use_fixed_weights
        self.bce_weight = bce_weight
        self.mae_weight = mae_weight
        self.loss_recipe = str(loss_recipe)
        self.alpha_bce = float(alpha_bce)
        self.w_gated = float(w_gated)
        self.w_mag = float(w_mag)
        self.horizon_decay = float(horizon_decay)
        self.adaptive_weighting = AdaptiveLossWeighting(num_tasks=2)
        self.zero_rate = zero_rate
        self.avg_nonzero_demand = avg_nonzero_demand
        self.pos_weight_cap = float(pos_weight_cap)
        self.positive_bce_boost = float(positive_bce_boost)
        self.focal_gamma = float(focal_gamma)
        self.zero_mag_weight = float(zero_mag_weight)
        self.use_mse_mag = bool(use_mse_mag)
        if self.loss_recipe == "spike_aware":
            self.pos_weight = resolve_positive_bce_weight(
                zero_rate,
                positive_bce_weight,
                boost=self.positive_bce_boost,
                cap=self.pos_weight_cap,
            )
        else:
            self.pos_weight = float(pos_weight)
        self.positive_bce_weight = float(self.pos_weight)
        zr = float(zero_rate)
        nz = max(1.0 - zr, 1e-6)
        self.w_zero = 1.0 / (2.0 * max(zr, 1e-6))
        self.w_nonzero = 1.0 / (2.0 * nz)
        self.sku_zero_rates = None
        self._sku_pos_weight_table = None
        if sku_zero_rates is not None:
            rates = np.asarray(sku_zero_rates, dtype=np.float32).reshape(-1)
            if rates.size == 0:
                raise ValueError("sku_zero_rates must be non-empty when provided")
            self.sku_zero_rates = rates
            pos_ws = np.array(
                [
                    min(
                        self.pos_weight_cap,
                        float(r) / max(1.0 - float(r), 1e-6),
                    )
                    for r in rates
                ],
                dtype=np.float32,
            )
            if self.loss_recipe == "spike_aware":
                # Scale SKU imbalance weights so the panel reference matches
                # the (boosted) spike-aware positive_bce_weight.
                panel_ref = min(
                    self.pos_weight_cap,
                    zr / max(1.0 - zr, 1e-6),
                )
                scale = self.pos_weight / max(float(panel_ref), 1e-6)
                pos_ws = np.minimum(
                    self.pos_weight_cap, pos_ws * scale
                ).astype(np.float32)
            self._sku_pos_weight_table = tf.constant(pos_ws, dtype=tf.float32)
        
        # Calculate optimal threshold for highly imbalanced data
        # For 90% zeros: threshold should match the non-zero rate (0.1) for balanced predictions
        # This is the expected prior probability: P(non-zero) = 1 - zero_rate
        self.optimal_threshold = max(0.05, 1.0 - zero_rate)
        
        # Metrics
        self.bce_loss_tracker = keras.metrics.Mean(name='zero_probability_loss')
        self.mae_loss_tracker = keras.metrics.Mean(name='final_forecast_loss')
        self.mag_loss_tracker = keras.metrics.Mean(name='magnitude_loss')
        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.mae_tracker = keras.metrics.MeanAbsoluteError(name='base_forecast_mae')
        self.final_mae_tracker = keras.metrics.MeanAbsoluteError(name='final_forecast_mae')
        # Non-zero detection metrics with data-driven threshold
        # For 90% zeros: threshold=0.1 aligns with prior probability P(non-zero)
        # This avoids the common mistake of using 0.5 for imbalanced data
        self.nonzero_precision = keras.metrics.Precision(name='nonzero_precision', thresholds=[self.optimal_threshold])
        self.nonzero_recall = keras.metrics.Recall(name='nonzero_recall', thresholds=[self.optimal_threshold])
        # Threshold-free metrics for imbalanced classification
        self.nonzero_aucpr = keras.metrics.AUC(curve='PR', name='nonzero_aucpr')
        self.nonzero_aucroc = keras.metrics.AUC(curve='ROC', name='nonzero_aucroc')
    
    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def _align(self, *tensors):
        """Flatten to [N, 1] so 1-step [B,1] and multi-horizon [B,H] share loss math."""
        out = []
        for t in tensors:
            t = tf.cast(t, tf.float32)
            out.append(tf.reshape(t, [-1, 1]))
        return out

    def _sku_ids_from_inputs(self, x):
        """Last list/tuple input is SKU; dict may use 'sku' / 'sku_input'."""
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return x[-1]
        if isinstance(x, dict):
            for key in ("sku", "sku_input", "sku_ids"):
                if key in x:
                    return x[key]
        return None

    def _bce_pos_weight_for_batch(self, y_true, sku_ids):
        """Scalar panel pos_weight or per-row SKU weights aligned to flattened y."""
        if self._sku_pos_weight_table is None or sku_ids is None:
            return tf.constant(self.pos_weight, dtype=tf.float32)
        sku = tf.reshape(tf.cast(sku_ids, tf.int32), [-1])
        pw = tf.gather(self._sku_pos_weight_table, sku)  # [B]
        # Multi-horizon: y is [B, H] before align → flat length B*H
        y_rank = y_true.shape.rank
        if y_rank is not None and y_rank >= 2:
            h = tf.shape(y_true)[-1]
            pw = tf.reshape(
                tf.tile(pw[:, tf.newaxis], [1, h]),
                [-1, 1],
            )
        else:
            # Dynamic rank or [B]/[B,1] after we'll align to [-1,1]
            flat_n = tf.reduce_prod(tf.shape(y_true))
            b = tf.shape(sku)[0]
            h = flat_n // b
            pw = tf.reshape(tf.tile(pw[:, tf.newaxis], [1, h]), [-1, 1])
        return pw

    def _compute_task_losses(
        self, y_true, final_forecast, non_zero_probability, base_forecast, sku_ids=None
    ):
        y_true = tf.cast(y_true, tf.float32)
        final_forecast = tf.cast(final_forecast, tf.float32)
        non_zero_probability = tf.cast(non_zero_probability, tf.float32)
        base_forecast = tf.cast(base_forecast, tf.float32)
        bce_pos_w = self._bce_pos_weight_for_batch(y_true, sku_ids)

        # Optional down-weight of farther horizons when rank > 1
        if self.horizon_decay < 1.0:
            H = tf.shape(y_true)[-1]
            idx = tf.cast(tf.range(H), tf.float32)
            hw = tf.pow(tf.constant(self.horizon_decay, tf.float32), idx)
            hw = hw / (tf.reduce_mean(hw) + 1e-6)
            # broadcast [H] -> y_true shape
            while len(hw.shape) < len(y_true.shape):
                hw = hw[tf.newaxis, ...]
            # For static rank-2 [B,H]:
            hw = tf.reshape(hw, [1, -1])

            def _wmean(abs_err, mask=None):
                w = hw
                if mask is not None:
                    w = w * mask
                return tf.reduce_sum(w * abs_err) / (tf.reduce_sum(w) + 1e-6)
        else:
            hw = None

            def _wmean(abs_err, mask=None):
                if mask is None:
                    return tf.reduce_mean(abs_err)
                return tf.reduce_sum(mask * abs_err) / (tf.reduce_sum(mask) + 1e-6)

        y_true, final_forecast, non_zero_probability, base_forecast = self._align(
            y_true, final_forecast, non_zero_probability, base_forecast
        )
        if bce_pos_w.shape.rank is not None and bce_pos_w.shape.rank == 0:
            pass  # scalar panel weight
        else:
            bce_pos_w = tf.reshape(bce_pos_w, [-1, 1])
        # If horizon weights were applied pre-align we'd need different path.
        # Equal-weight flatten is default; horizon_decay handled below when shapes match.
        y_nonzero = tf.cast(y_true > 0, tf.float32)

        if self.loss_recipe == "spike_aware":
            p_clip = tf.clip_by_value(non_zero_probability, 1e-7, 1.0 - 1e-7)
            if self.focal_gamma > 0.0:
                g = tf.constant(self.focal_gamma, dtype=tf.float32)
                focal_pos = tf.pow(1.0 - p_clip, g)
                focal_neg = tf.pow(p_clip, g)
            else:
                focal_pos = 1.0
                focal_neg = 1.0
            bce_loss = tf.reduce_mean(
                -bce_pos_w * y_nonzero * focal_pos * tf.math.log(p_clip)
                - (1.0 - y_nonzero) * focal_neg * tf.math.log(1.0 - p_clip)
            )
            err = (
                tf.square(y_true - base_forecast)
                if self.use_mse_mag
                else tf.abs(y_true - base_forecast)
            )
            mag_pos = tf.reduce_sum(y_nonzero * err) / (tf.reduce_sum(y_nonzero) + 1e-6)
            mag_zero = tf.reduce_sum((1.0 - y_nonzero) * err) / (
                tf.reduce_sum(1.0 - y_nonzero) + 1e-6
            )
            mag_mae = mag_pos + self.zero_mag_weight * mag_zero
            if self.w_gated > 0.0:
                w = self.w_zero * (1.0 - y_nonzero) + self.w_nonzero * y_nonzero
                gated_mae = tf.reduce_sum(w * tf.abs(y_true - final_forecast)) / (
                    tf.reduce_sum(w) + 1e-6
                )
            else:
                gated_mae = tf.constant(0.0, dtype=tf.float32)
            total = (
                self.alpha_bce * bce_loss
                + self.w_mag * mag_mae
                + self.w_gated * gated_mae
            )
            # Report gated term as "mae" tracker when present; else mag.
            return total, bce_loss, gated_mae if self.w_gated > 0.0 else mag_mae, mag_mae, y_nonzero

        if self.loss_recipe == "three_term":
            w = self.w_zero * (1.0 - y_nonzero) + self.w_nonzero * y_nonzero
            gated_mae = tf.reduce_sum(w * tf.abs(y_true - final_forecast)) / tf.reduce_sum(w)
            mag_mae = tf.reduce_sum(y_nonzero * tf.abs(y_true - base_forecast)) / (
                tf.reduce_sum(y_nonzero) + 1e-6
            )
            p_clip = tf.clip_by_value(non_zero_probability, 1e-7, 1.0 - 1e-7)
            bce_loss = tf.reduce_mean(
                -bce_pos_w * y_nonzero * tf.math.log(p_clip)
                - (1.0 - y_nonzero) * tf.math.log(1.0 - p_clip)
            )
            total = (
                self.w_gated * gated_mae
                + self.w_mag * mag_mae
                + self.alpha_bce * bce_loss
            )
            return total, bce_loss, gated_mae, mag_mae, y_nonzero

        # BCE: per-SKU when table is set; else provided loss fn (panel scalar)
        if self._sku_pos_weight_table is not None and sku_ids is not None:
            p_clip = tf.clip_by_value(non_zero_probability, 1e-7, 1.0 - 1e-7)
            bce_loss = tf.reduce_mean(
                -bce_pos_w * y_nonzero * tf.math.log(p_clip)
                - (1.0 - y_nonzero) * tf.math.log(1.0 - p_clip)
            )
        else:
            bce_loss = self.bce_loss_fn(y_nonzero, non_zero_probability)

        if self.loss_recipe == "bce_mae":
            mae_loss = tf.reduce_mean(tf.abs(y_true - final_forecast))
            mag_mae = tf.reduce_sum(y_nonzero * tf.abs(y_true - base_forecast)) / (
                tf.reduce_sum(y_nonzero) + 1e-6
            )
        else:  # legacy
            mae_loss = self.mae_loss_fn(
                y_true, final_forecast, sample_weight=tf.squeeze(y_nonzero, axis=-1)
            )
            mag_mae = mae_loss

        if self.use_fixed_weights or self.loss_recipe == "bce_mae":
            total = self.bce_weight * bce_loss + self.mae_weight * mae_loss
        else:
            bce_norm = tf.stop_gradient(tf.maximum(tf.reduce_mean(tf.abs(bce_loss)), 1e-6))
            mae_norm = tf.stop_gradient(tf.maximum(tf.reduce_mean(tf.abs(mae_loss)), 1e-6))
            total = self.adaptive_weighting([bce_loss / bce_norm, mae_loss / mae_norm])

        return total, bce_loss, mae_loss, mag_mae, y_nonzero
    
    def train_step(self, data):
        if isinstance(data, (tuple, list)) and len(data) == 3:
            x, y, _sample_weight = data
        else:
            x, y = data
        y_true = y.get('base_forecast', y.get('final_forecast'))
        sku_ids = self._sku_ids_from_inputs(x)
        
        with tf.GradientTape() as tape:
            outputs = self.base_model(x, training=True)
            final_forecast = outputs['final_forecast']
            non_zero_probability = outputs['non_zero_probability']
            base_forecast = outputs.get('base_forecast', final_forecast)

            total_loss, bce_loss, mae_loss, mag_mae, y_nonzero = self._compute_task_losses(
                y_true, final_forecast, non_zero_probability, base_forecast, sku_ids=sku_ids
            )
            
            if self.base_model.losses:
                total_loss = total_loss + tf.add_n(self.base_model.losses)
        
        trainable_vars = self.base_model.trainable_variables
        if self.loss_recipe == "legacy" and not self.use_fixed_weights:
            trainable_vars = trainable_vars + self.adaptive_weighting.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        fixed_gradients = []
        for grad in gradients:
            if grad is None:
                fixed_gradients.append(None)
            else:
                fixed_grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
                fixed_gradients.append(fixed_grad)
        fixed_gradients, _ = tf.clip_by_global_norm(fixed_gradients, 5.0)
        
        self.optimizer.apply_gradients(zip(fixed_gradients, trainable_vars))
        
        y_true_m, final_m, base_m, p_m = self._align(
            y_true, final_forecast, base_forecast, non_zero_probability
        )
        self.bce_loss_tracker.update_state(bce_loss)
        self.mae_loss_tracker.update_state(mae_loss)
        self.mag_loss_tracker.update_state(mag_mae)
        self.total_loss_tracker.update_state(total_loss)
        self.nonzero_precision.update_state(y_nonzero, p_m)
        self.nonzero_recall.update_state(y_nonzero, p_m)
        self.nonzero_aucpr.update_state(y_nonzero, p_m)
        self.nonzero_aucroc.update_state(y_nonzero, p_m)
        self.mae_tracker.update_state(y_true_m, base_m, sample_weight=y_nonzero)
        self.final_mae_tracker.update_state(y_true_m, final_m)
        
        return {
            'loss': self.total_loss_tracker.result(),
            'classification_loss': self.bce_loss_tracker.result(),
            'final_forecast_loss': self.mae_loss_tracker.result(),
            'magnitude_loss': self.mag_loss_tracker.result(),
            'nonzero_precision': self.nonzero_precision.result(),
            'nonzero_recall': self.nonzero_recall.result(),
            'nonzero_aucpr': self.nonzero_aucpr.result(),
            'nonzero_aucroc': self.nonzero_aucroc.result(),
            'base_forecast_mae': self.mae_tracker.result(),
            'final_forecast_mae': self.final_mae_tracker.result()
        }
    
    def test_step(self, data):
        if isinstance(data, (tuple, list)) and len(data) == 3:
            x, y, _sample_weight = data
        else:
            x, y = data
        y_true = y.get('base_forecast', y.get('final_forecast'))
        sku_ids = self._sku_ids_from_inputs(x)
        
        outputs = self.base_model(x, training=False)
        final_forecast = outputs['final_forecast']
        non_zero_probability = outputs['non_zero_probability']
        base_forecast = outputs.get('base_forecast', final_forecast)

        total_loss, bce_loss, mae_loss, mag_mae, y_nonzero = self._compute_task_losses(
            y_true, final_forecast, non_zero_probability, base_forecast, sku_ids=sku_ids
        )
        
        if self.base_model.losses:
            total_loss = total_loss + tf.add_n(self.base_model.losses)
        
        y_true_m, final_m, base_m, p_m = self._align(
            y_true, final_forecast, base_forecast, non_zero_probability
        )
        self.bce_loss_tracker.update_state(bce_loss)
        self.mae_loss_tracker.update_state(mae_loss)
        self.mag_loss_tracker.update_state(mag_mae)
        self.total_loss_tracker.update_state(total_loss)
        self.nonzero_precision.update_state(y_nonzero, p_m)
        self.nonzero_recall.update_state(y_nonzero, p_m)
        self.nonzero_aucpr.update_state(y_nonzero, p_m)
        self.nonzero_aucroc.update_state(y_nonzero, p_m)
        self.mae_tracker.update_state(y_true_m, base_m, sample_weight=y_nonzero)
        self.final_mae_tracker.update_state(y_true_m, final_m)
        
        return {
            'loss': self.total_loss_tracker.result(),
            'zero_probability_loss': self.bce_loss_tracker.result(),
            'final_forecast_loss': self.mae_loss_tracker.result(),
            'classification_loss': self.bce_loss_tracker.result(),
            'magnitude_loss': self.mag_loss_tracker.result(),
            'nonzero_precision': self.nonzero_precision.result(),
            'nonzero_recall': self.nonzero_recall.result(),
            'nonzero_aucpr': self.nonzero_aucpr.result(),
            'nonzero_aucroc': self.nonzero_aucroc.result(),
            'base_forecast_mae': self.mae_tracker.result(),
            'final_forecast_mae': self.final_mae_tracker.result()
        }
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.bce_loss_tracker,
            self.mae_loss_tracker,
            self.mag_loss_tracker,
            self.nonzero_precision,
            self.nonzero_recall,
            self.nonzero_aucpr,
            self.nonzero_aucroc,
            self.mae_tracker,
            self.final_mae_tracker
        ]


class FocalLoss(keras.losses.Loss):
    """
    Binary focal loss for handling class imbalance.
    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t = p for y=1, and 1-p for y=0.
    """
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # y_true: probability of zero (0/1), y_pred: predicted zero probability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

        # p_t: probability of the true class
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)

        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)
        loss = -focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        return {'alpha': self.alpha, 'gamma': self.gamma}

class WeightedFocalLoss(FocalLoss):
    """
    Focal loss with proper positive class weighting for imbalanced data.
    - Uses focal parameters (alpha, gamma) for hard-example mining
    - Uses pos_weight to balance class contribution to loss (NOT penalty multiplier)
    - pos_weight should equal zero_rate / (1 - zero_rate) for balanced training
    """
    def __init__(self, alpha=0.10, gamma=3.0, pos_weight=9.0, name='weighted_focal_loss'):
        super().__init__(alpha=alpha, gamma=gamma, name=name)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

        # Compute base focal loss
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)
        focal_loss = -focal_weight * tf.math.log(p_t)

        # Apply class weighting: weight positive class contribution by pos_weight
        # This balances contribution from rare (positive) vs common (negative) class
        # NOT a false-negative penalty multiplier (that causes model collapse)
        class_weight = y_true * self.pos_weight + (1.0 - y_true) * 1.0
        weighted_loss = focal_loss * class_weight

        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'pos_weight': self.pos_weight})
        return cfg

class WeightedBCELoss(tf.keras.losses.Loss):
    """
    Weighted Binary Cross-Entropy loss for imbalanced classification.
    BCE = -[y * log(p) + (1-y) * log(1-p)]
    loss = BCE * (non_zero * weight_nonzero + zero * weight_zero)
    where zero = (1 - non_zero)
    
    Args:
        weight_nonzero: Weight for non-zero (positive) class errors
        weight_zero: Weight for zero (negative) class errors
    """
    def __init__(self, weight_nonzero=9.0, weight_zero=1.0, name='weighted_bce'):
        super().__init__(name=name)
        self.weight_nonzero = weight_nonzero
        self.weight_zero = weight_zero
    
    def call(self, y_true, y_pred):
        # Force matching rank to avoid (N,) vs (N,1) -> (N,N) broadcast
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1, 1])
        # Clip predictions for numerical stability in log
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary Cross-Entropy: -[y * log(p) + (1-y) * log(1-p)]
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        
        # Compute sample weights: non_zero * weight_nonzero + zero * weight_zero
        non_zero = y_true  # 1 for non-zero, 0 for zero
        zero = 1.0 - non_zero
        sample_weights = non_zero * self.weight_nonzero + zero * self.weight_zero
        
        # Apply weights to BCE
        weighted_bce = bce * sample_weights
        
        return tf.reduce_mean(weighted_bce)
    
    def get_config(self):
        return {
            'weight_nonzero': self.weight_nonzero,
            'weight_zero': self.weight_zero
        }

class AdaptiveThresholdCallback(keras.callbacks.Callback):
    """
    Adjusts classification threshold after each epoch to hit a target recall
    or maximize F1 on the validation set. Starts from a user-provided
    initial threshold so the first epoch metrics are meaningful even before
    calibration kicks in.
    """
    def __init__(
        self,
        val_inputs,
        val_targets,
        target_recall=0.35,
        mode='target_recall',
        initial_threshold=0.05
    ):
        super().__init__()
        self.val_inputs = val_inputs
        self.val_targets = val_targets
        self.target_recall = float(target_recall)
        self.mode = mode
        self.initial_threshold = float(initial_threshold)

    def on_train_begin(self, logs=None):
        # Seed metrics with the initial threshold before the first epoch runs.
        try:
            if hasattr(self.model, 'nonzero_precision'):
                self.model.nonzero_precision.thresholds = [self.initial_threshold]
            if hasattr(self.model, 'nonzero_recall'):
                self.model.nonzero_recall.thresholds = [self.initial_threshold]
        except Exception:
            pass

    def on_epoch_end(self, epoch, logs=None):
        # Predict non-zero probabilities on validation set
        outputs = self.model.base_model(self.val_inputs, training=False)
        p = outputs['non_zero_probability'].numpy().reshape(-1)
        y = (self.val_targets > 0).astype(np.float32).reshape(-1)

        # Search very wide threshold range (down to 0.001) to find any positive predictions
        thresholds = np.concatenate([
            np.geomspace(1e-6, 1e-3, 15),
            np.linspace(0.001, 0.05, 50),
            np.linspace(0.05, 0.20, 40),
            np.linspace(0.20, 0.90, 36)
        ])
        thresholds = np.unique(thresholds)
        best_thr = self.initial_threshold
        best_score = -1.0
        best_stats = (0, 0, 0)  # tp, fp, fn

        for thr in thresholds:
            pred = (p >= thr).astype(np.float32)
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            if self.mode == 'target_recall':
                # Prioritize matching target recall; drop precision bias
                score = -abs(recall - self.target_recall)
            else:
                # F1 mode
                f1 = 2 * precision * recall / (precision + recall + 1e-7)
                score = f1
            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_stats = (int(tp), int(fp), int(fn))

        # Fallback: if recall ~ 0 across scanned thresholds, pick a very low threshold
        tp, fp, fn = best_stats
        if tp == 0 and np.max(p) > 0.0:
            high_p = float(np.percentile(p, 99))
            candidate_thr = max(0.001, min(high_p, float(np.max(p))) * 0.5)
            pred = (p >= candidate_thr).astype(np.float32)
            tp = int(np.sum((pred == 1) & (y == 1)))
            fp = int(np.sum((pred == 1) & (y == 0)))
            fn = int(np.sum((pred == 0) & (y == 1)))
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            best_thr = float(candidate_thr)
            best_score = -abs(recall - self.target_recall) if self.mode == 'target_recall' else (2 * precision * recall / (precision + recall + 1e-7))
            best_stats = (tp, fp, fn)

        # Update metric thresholds for the next epoch
        try:
            if hasattr(self.model.nonzero_precision, 'thresholds'):
                self.model.nonzero_precision.thresholds = [best_thr]
            if hasattr(self.model.nonzero_recall, 'thresholds'):
                self.model.nonzero_recall.thresholds = [best_thr]
            tp, fp, fn = best_stats
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            print(f"\n[Adaptive Threshold] Epoch {epoch+1}: threshold set to {best_thr:.3f} (score={best_score:.4f}, recall={recall:.4f}, precision={precision:.4f}, tp={tp}, fp={fp}, fn={fn})")
        except Exception as e:
            print(f"\n[Adaptive Threshold] Warning: could not update metric thresholds ({e})")


class AdaptiveWeightLogger(keras.callbacks.Callback):
    """Logs adaptive weight changes during training."""
    
    def __init__(self, log_freq=1):
        super().__init__()
        self.log_freq = log_freq
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_freq == 0:
            recipe = getattr(self.model, 'loss_recipe', 'legacy')
            if recipe in ('three_term', 'bce_mae', 'spike_aware') or getattr(self.model, 'use_fixed_weights', False):
                print(f"\n[Loss recipe={recipe} - Epoch {epoch+1}] fixed/composite weights (no adaptive log-vars)")
                return
            summary = self.model.adaptive_weighting.get_weights_summary()
            print(f"\n[Adaptive Weights - Epoch {epoch+1}]")
            # Handle fixed weights case
            if summary['weights_pct'] is None:
                print(f"  Using fixed weights in AdaptiveWeightedModel wrapper")
            else:
                bce_w = summary['weights_pct'][0]
                bce_lv = summary['log_vars'][0]
                mae_w = summary['weights_pct'][1]
                mae_lv = summary['log_vars'][1]
                print(f"  BCE weight: {bce_w:.2f}% (log_var={bce_lv:.4f})")
                print(f"  MAE weight: {mae_w:.2f}% (log_var={mae_lv:.4f})")


def main():
    """Train with adaptive loss weighting."""
    
    print("\n" + "="*70)
    print("BALANCED TRAINING (Higher Threshold for Precision)")
    print("="*70)
    print("Using BALANCED 50% classification + 50% forecast weighting")
    print("Focus: Improve precision while keeping recall reasonable")
    print("Threshold: Fixed at 0.55 (higher threshold → fewer false positives)\n")
    # Parse CLI / config
    parser = argparse.ArgumentParser(description="Train lightweight adaptive loss model")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--alpha", type=float, default=None, help="Focal loss alpha (minority weight)")
    parser.add_argument("--gamma", type=float, default=None, help="Focal loss gamma (focus strength)")
    parser.add_argument("--pos_weight", type=float, default=None, help="Positive class weight (class imbalance correction). Use zero_rate/(1-zero_rate) for balanced.")
    parser.add_argument("--target_recall", type=float, default=None, help="Target recall for threshold callback")
    parser.add_argument("--min_bce_weight", type=float, default=None, help="Minimum classification weight share (0-1)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Optimizer learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing train_split.csv and val_split.csv")
    parser.add_argument(
        "--loss_recipe",
        type=str,
        default=None,
        choices=["legacy", "bce_mae", "three_term", "spike_aware"],
        help=(
            "Loss recipe: legacy (nonzero MAE), bce_mae (all-day MAE), "
            "three_term (recommended bake-off), spike_aware (opt-in heavy "
            "positive BCE + positive-only magnitude)"
        ),
    )
    parser.add_argument("--alpha_bce", type=float, default=None, help="BCE weight for three_term/spike_aware (default 0.2 / 1.0)")
    parser.add_argument("--w_gated", type=float, default=None, help="Gated MAE weight for three_term/spike_aware (spike default 0)")
    parser.add_argument("--w_mag", type=float, default=None, help="Nonzero magnitude MAE weight (default 1.0)")
    parser.add_argument("--bce_weight", type=float, default=None, help="BCE weight for bce_mae/legacy fixed mix")
    parser.add_argument("--mae_weight", type=float, default=None, help="MAE weight for bce_mae/legacy fixed mix")
    parser.add_argument(
        "--positive_bce_weight",
        type=float,
        default=None,
        help="Spike-aware: absolute positive-class BCE weight (default boost*zr/(1-zr))",
    )
    parser.add_argument(
        "--positive_bce_boost",
        type=float,
        default=None,
        help="Spike-aware: multiplier on zr/(1-zr) when positive_bce_weight omitted (default 2)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=None,
        help="Spike-aware: focal BCE gamma (0=off)",
    )
    parser.add_argument(
        "--zero_mag_weight",
        type=float,
        default=None,
        help="Spike-aware: relative zero-day magnitude weight on b (default 0.05)",
    )
    args = parser.parse_args([] if hasattr(sys, 'ps1') else None)

    cfg = {}
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                try:
                    cfg = json.load(f)
                except Exception:
                    print(f"WARNING: Could not parse JSON config: {args.config}")
        else:
            print(f"WARNING: Config file not found: {args.config}")

    def cfg_val(key, default):
        v = cfg.get(key, default)
        # CLI overrides config
        cli_v = getattr(args, key, None)
        return cli_v if cli_v is not None else v

    # Load the completed study (optional). If missing, continue with config defaults.
    study_path = 'optuna_study_lightweight.pkl'
    best_params = None
    if not os.path.exists(study_path):
        print(f"WARNING: Study file not found: {study_path}")
        print("Proceeding without Optuna: using configuration defaults.")
    else:
        try:
            study = joblib.load(study_path)
            best_params = study.best_params
        except Exception as e:
            print(f"WARNING: Failed to load study: {e}")
            print("Proceeding without Optuna: using configuration defaults.")
    
    # Override architecture flags for component attention + cross-layer + intermittent
    if best_params is None:
        best_params = {
            'batch_size': int(cfg_val('batch_size', 1024)),
            'hidden_dim': int(cfg_val('hidden_dim', 48)),
            'sku_embedding_dim': int(cfg_val('sku_embedding_dim', 4)),
            'dropout_rate': float(cfg_val('dropout_rate', 0.229909)),
            'learning_rate': float(cfg_val('learning_rate', 0.0025)),
            'loss_type': cfg_val('loss_type', 'sku_aware'),
            'alpha': float(cfg_val('alpha', 0.10)),
            'use_cross_layers': bool(cfg_val('use_cross_layers', False)),
            'use_intermittent': bool(cfg_val('use_intermittent', True)),
        }
    else:
        best_params['use_cross_layers'] = bool(cfg_val('use_cross_layers', False))
        best_params['use_intermittent'] = True
    
    print("\nUsing base hyperparameters from Trial 18 (with architecture overrides):")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Load feature configuration
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    feature_config = load_feature_config()
    
    # Load data from CSV
    data_dir = cfg_val('data_dir', 'data')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_split.csv'), parse_dates=['ds'])
    val_df = pd.read_csv(os.path.join(data_dir, 'val_split.csv'), parse_dates=['ds'])
    
    # Load pre-computed holiday features
    holiday_train = pd.read_csv(os.path.join(data_dir, 'holiday_features_train.csv'))
    holiday_val = pd.read_csv(os.path.join(data_dir, 'holiday_features_val.csv'))
    
    # Create features (causal regressor: train states warm-start val)
    X_train_df, sku_states = feature_config.create_features(
        train_df, holiday_train, return_states=True
    )
    X_val_df, sku_states = feature_config.create_features(
        val_df, holiday_val, prior_states=sku_states, return_states=True
    )
    
    # Prepare inputs
    X_train = X_train_df.values.astype(np.float32)
    X_val = X_val_df.values.astype(np.float32)
    y_train = train_df['Quantity'].values.astype(np.float32)
    y_val = val_df['Quantity'].values.astype(np.float32)
    sku_train = train_df['id_var'].astype('category').cat.codes.values
    sku_val = val_df['id_var'].astype('category').cat.codes.values
    
    print(f"\nData shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, sku={sku_train.shape}")
    print(f"  Val: X={X_val.shape}, y={y_val.shape}, sku={sku_val.shape}")
    
    # Dataset statistics
    zero_rate = np.mean(y_train == 0)
    avg_nonzero_demand = np.mean(y_train[y_train > 0])
    
    # Compute per-SKU non-zero means for adaptive weighting
    train_sku_stats = pd.DataFrame({
        'sku': sku_train,
        'demand': y_train
    })
    # Get mean non-zero demand per SKU
    sku_nonzero_means = train_sku_stats[train_sku_stats['demand'] > 0].groupby('sku')['demand'].mean()
    # Fill missing SKUs (all zeros) with global mean
    sku_nonzero_means = sku_nonzero_means.reindex(range(len(np.unique(sku_train))), fill_value=avg_nonzero_demand)
    
    # Create per-sample weights: log1p(sku_mean) for each sample
    sample_weights_train = np.log1p(sku_nonzero_means[sku_train].values).astype(np.float32)
    sample_weights_val = np.log1p(sku_nonzero_means[sku_val].values).astype(np.float32)
    
    print(f"\nDataset characteristics:")
    print(f"  Zero rate: {zero_rate*100:.2f}%")
    print(f"  Non-zero rate: {(1-zero_rate)*100:.2f}%")
    print(f"  Average non-zero demand: {avg_nonzero_demand:.4f}")
    print(f"  Imbalance ratio: {zero_rate/(1-zero_rate):.2f}:1")
    print(f"\nPer-SKU weighting:")
    print(f"  SKU non-zero means - min: {sku_nonzero_means.min():.2f}, max: {sku_nonzero_means.max():.2f}, median: {sku_nonzero_means.median():.2f}")
    print(f"  Sample weights - min: {sample_weights_train.min():.4f}, max: {sample_weights_train.max():.4f}, mean: {sample_weights_train.mean():.4f}")
    
    # Split features by component
    X_all = X_train
    X_train_trend = X_all[:, feature_config.trend_indices]
    X_train_seasonal = X_all[:, feature_config.seasonal_indices]
    X_train_holiday = X_all[:, feature_config.holiday_indices]
    X_train_regressor = X_all[:, feature_config.regressor_indices]
    
    X_all = X_val
    X_val_trend = X_all[:, feature_config.trend_indices]
    X_val_seasonal = X_all[:, feature_config.seasonal_indices]
    X_val_holiday = X_all[:, feature_config.holiday_indices]
    X_val_regressor = X_all[:, feature_config.regressor_indices]
    
    print(f"\nFeature splits:")
    print(f"  Trend: {X_train_trend.shape[1]} features")
    print(f"  Seasonal: {X_train_seasonal.shape[1]} features")
    print(f"  Holiday: {X_train_holiday.shape[1]} features")
    print(f"  Regressor: {X_train_regressor.shape[1]} features")
    
    # Create base model
    print("\n" + "="*70)
    print("CREATING MODEL WITH ADAPTIVE LOSS WEIGHTING")
    print("="*70)
    
    n_skus = len(np.unique(sku_train))
    
    # Use the new optimized architecture with all improvements:
    # - Bias removed from all layers except trend output
    # - Normalized attention patterns
    # - Gated addition for component combination
    base_model = build_hierarchical_model_lightweight(
        n_temporal_features=X_train_trend.shape[1],
        n_fourier_features=X_train_seasonal.shape[1],
        n_holiday_features=X_train_holiday.shape[1],
        n_lag_features=X_train_regressor.shape[1],
        n_skus=n_skus,
        n_changepoints=10,
        hidden_dim=best_params['hidden_dim'],
        sku_embedding_dim=best_params.get('sku_embedding_dim', 4),
        dropout_rate=best_params['dropout_rate'],
        use_cross_layers=best_params['use_cross_layers'],
        use_intermittent=best_params['use_intermittent']
    )
    
    print(f"\nModel created with optimized architecture:")
    print(f"  Hidden dim: {best_params['hidden_dim']}")
    print(f"  SKU embedding dim: {best_params.get('sku_embedding_dim', 4)}")
    print(f"  Dropout: {best_params['dropout_rate']:.4f}")
    print(f"  Number of SKUs: {n_skus}")
    print(f"  Changepoints: 10")
    print(f"  Total parameters: {base_model.count_params():,}")
    
    # Binary target
    y_train_binary = (y_train > 0).astype(np.float32)
    y_val_binary = (y_val > 0).astype(np.float32)
    
    # Define loss functions
    # Use log-scale MAE to match BCE magnitude range and improve numerical balance
    def log_mae_loss(y_true, y_pred):
        """
        Log-scale mean absolute error.
        Operates on log1p(values) to match BCE's log scale and prevent 
        magnitude differences from dominating the loss.
        
        Args:
            y_true: True demand values
            y_pred: Predicted demand values
        
        Returns:
            MAE on log-transformed values
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to avoid log of negative values
        y_pred = tf.maximum(y_pred, 0.0)
        
        # Add epsilon for numerical stability before log
        epsilon = 1e-7
        y_true_safe = tf.maximum(y_true, epsilon)
        y_pred_safe = tf.maximum(y_pred, epsilon)
        
        y_true_log = tf.math.log1p(y_true_safe)
        y_pred_log = tf.math.log1p(y_pred_safe)
        
        mae = tf.abs(y_true_log - y_pred_log)
        # Filter out NaNs/Infs before reducing
        mae = tf.where(tf.math.is_finite(mae), mae, 0.0)
        return tf.reduce_mean(mae)
    
    # Use composite loss from losses.py
    # This loss automatically handles:
    # - Class weighting (zero=1, non-zero=9) in BCE
    # - MAE/MSE only on non-zero samples
    # - Log1p SKU volume weighting for magnitude loss
    weight_nonzero = float(cfg_val('weight_nonzero', 9.0))
    use_mse = bool(cfg_val('use_mse', False))  # True for MSE, False for MAE
    loss_recipe = str(cfg_val('loss_recipe', 'legacy'))
    # Spike-aware defaults differ from three_term when knobs omitted.
    _default_alpha = 1.0 if loss_recipe == 'spike_aware' else 0.2
    _default_gated = 0.0 if loss_recipe == 'spike_aware' else 1.0
    alpha_bce = float(cfg_val('alpha_bce', _default_alpha))
    w_gated = float(cfg_val('w_gated', _default_gated))
    w_mag = float(cfg_val('w_mag', 1.0))
    fixed_bce_weight = cfg_val('bce_weight', None)
    fixed_mae_weight = cfg_val('mae_weight', None)
    positive_bce_weight = cfg_val('positive_bce_weight', None)
    positive_bce_boost = float(cfg_val('positive_bce_boost', 2.0))
    focal_gamma = float(cfg_val('focal_gamma', 0.0))
    zero_mag_weight = float(cfg_val('zero_mag_weight', 0.05))

    if loss_recipe == 'spike_aware':
        loss_config = spike_aware_loss_config(
            zero_rate=zero_rate,
            alpha_bce=alpha_bce,
            w_mag=w_mag,
            zero_mag_weight=zero_mag_weight,
            w_gated=w_gated,
            positive_bce_weight=positive_bce_weight,
            positive_bce_boost=positive_bce_boost,
            focal_gamma=focal_gamma,
            use_mse=use_mse,
        )
        print(f"\n✓ Using SPIKE-AWARE intermittent recipe (opt-in):")
        print(f"  - heavy positive BCE (pos_weight={loss_config['meta']['positive_bce_weight']:.2f}, "
              f"focal_gamma={focal_gamma})")
        print(f"  - positive-only magnitude on base_forecast "
              f"(zero_mag_weight={zero_mag_weight})")
        print(f"  - optional gated timing w_gated={w_gated}")
        print(f"  - weights: mag={w_mag}, bce={alpha_bce}")
        print(f"  - ŷ=p·b unchanged; bake-off default remains three_term")
    elif loss_recipe == 'three_term':
        loss_config = three_term_loss_config(
            zero_rate=zero_rate,
            alpha_bce=alpha_bce,
            w_gated=w_gated,
            w_mag=w_mag,
            pos_weight=weight_nonzero,
        )
        print(f"\n✓ Using THREE-TERM intermittent recipe (recommended):")
        print(f"  - inverse-weighted gated MAE on ALL days (timing)")
        print(f"  - masked MAE on base_forecast for sale days (size)")
        print(f"  - light weighted BCE (alpha_bce={alpha_bce})")
        print(f"  - weights: gated={w_gated}, mag={w_mag}, bce={alpha_bce}")
        print(f"  - class weights: w_zero={loss_config['meta']['w_zero']:.3f}, "
              f"w_nz={loss_config['meta']['w_nonzero']:.3f}")
    elif loss_recipe == 'bce_mae':
        bw = float(fixed_bce_weight) if fixed_bce_weight is not None else 0.5
        mw = float(fixed_mae_weight) if fixed_mae_weight is not None else 0.5
        loss_config = bce_mae_loss_config(
            zero_rate=zero_rate,
            bce_weight=bw,
            mae_weight=mw,
            pos_weight=weight_nonzero,
            mae_all_days=True,
        )
        print(f"\n✓ Using BCE + all-day MAE recipe:")
        print(f"  - BCE weight={bw}, MAE weight={mw}, pos_weight={weight_nonzero:.2f}")
    else:
        # Create composite loss with data-driven weighting (legacy)
        # Returns separate losses for each output with appropriate weights
        # NOTE: We use weight=1.0 for forecast loss since per-SKU weighting is applied via sample_weight
        loss_config = composite_loss(
            zero_rate=zero_rate,  # Auto-weights BCE by zero rate
            average_nonzero_demand=1.0,  # Set to 1.0 - per-SKU weighting via sample_weight instead
            pos_weight=weight_nonzero,  # 9:1 class weighting
            use_mse=use_mse  # MAE or MSE for magnitude
        )
        print(f"\n✓ Using LEGACY composite_loss (nonzero-only MAE on final):")
        print(f"  - BCE with pos_weight={weight_nonzero:.1f} (zero=1.0, non-zero={weight_nonzero:.1f})")
        print(f"  - {'MSE' if use_mse else 'MAE'} on non-zero samples only")

    bce_weight = loss_config['weights'].get('non_zero_probability', 1.0)
    mae_weight = loss_config['weights'].get('final_forecast', 1.0)
    
    print(f"  - BCE weight: {bce_weight:.4f}")
    print(f"  - final_forecast weight: {mae_weight:.4f}")
    if 'base_forecast' in loss_config['weights']:
        print(f"  - base_forecast (mag) weight: {loss_config['weights']['base_forecast']:.4f}")
    if loss_recipe == 'legacy':
        print(f"  - Per-SKU scaling: log1p(sku_nonzero_mean) applied as sample weights")
    
    # Initial threshold for metrics
    optimal_threshold = 0.55
    
    # Use conservative learning rate for stability with intermittent data
    lr = float(cfg_val('learning_rate', best_params.get('learning_rate', 0.001)))
    # For intermittent data, smaller learning rate helps with stability
    if lr > 0.005:
        lr = 0.005  # Cap at 0.5% per step
        print(f"  WARNING: Reducing learning rate to {lr:.6f} for stability")
    
    # Compile model with separate losses for each output
    compile_metrics = {
        'non_zero_probability': [
            keras.metrics.Precision(name='nonzero_precision', thresholds=[optimal_threshold]),
            keras.metrics.Recall(name='nonzero_recall', thresholds=[optimal_threshold]),
            keras.metrics.AUC(curve='PR', name='nonzero_aucpr'),
            keras.metrics.AUC(curve='ROC', name='nonzero_aucroc')
        ],
        'final_forecast': [
            keras.metrics.MeanAbsoluteError(name='final_forecast_mae')
        ]
    }
    if 'base_forecast' in loss_config['losses']:
        compile_metrics['base_forecast'] = [
            keras.metrics.MeanAbsoluteError(name='base_forecast_mae')
        ]

    base_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=lr,
            global_clipnorm=10.0  # Clip L2 norm of all gradients
        ),
        loss=loss_config['losses'],
        loss_weights=loss_config['weights'],
        metrics=compile_metrics,
    )
    
    # Build the model explicitly to enable weight saving
    dummy_trend = np.zeros((1, X_train_trend.shape[1]), dtype=np.float32)
    dummy_seasonal = np.zeros((1, X_train_seasonal.shape[1]), dtype=np.float32)
    dummy_holiday = np.zeros((1, X_train_holiday.shape[1]), dtype=np.float32)
    dummy_regressor = np.zeros((1, X_train_regressor.shape[1]), dtype=np.float32)
    dummy_sku = np.array([0], dtype=np.int32)
    
    _ = base_model([dummy_trend, dummy_seasonal, dummy_holiday, dummy_regressor, dummy_sku], training=False)
    
    # Use base_model directly
    model = base_model
    
    print("\n✓ Model compiled successfully")
    print(f"\n✓ Model compilation complete:")
    print(f"  Optimizer: Adam with gradient clipping (global_norm=10.0)")
    print(f"  Learning rate: {lr:.6f}")
    print(f"  Loss recipe: {loss_recipe}")
    print(f"  Classification threshold: {optimal_threshold:.2f} (for metrics)")
    
    # Callbacks
    patience = int(cfg_val('patience', 15))
    
    # Callback to compute recall-constrained F1-optimal threshold 
    # Ensures minimum recall (70%) while maximizing F1-score
    class RecallConstrainedF1Callback(keras.callbacks.Callback):
        def __init__(self, val_inputs, val_binary, sample_size=100000, min_recall=0.70):
            super().__init__()
            self.val_inputs = val_inputs
            self.val_binary = val_binary
            self.sample_size = sample_size
            self.min_recall = min_recall  # Minimum acceptable recall
            self.best_threshold = 0.3  # Start with recall-friendly threshold
            
        def on_epoch_end(self, epoch, logs=None):
            try:
                # Sample validation data to avoid memory issues
                n_samples = len(self.val_binary)
                if n_samples > self.sample_size:
                    indices = np.random.choice(n_samples, self.sample_size, replace=False)
                    val_inputs_sample = [x[indices] for x in self.val_inputs]
                    val_binary_sample = self.val_binary[indices]
                else:
                    val_inputs_sample = self.val_inputs
                    val_binary_sample = self.val_binary
                
                # Get predictions
                outputs = self.model.predict(val_inputs_sample, verbose=0)
                y_probs = outputs['non_zero_probability'].flatten()
                y_true = val_binary_sample.flatten()
                
                from sklearn.metrics import precision_recall_curve, f1_score
                precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
                
                # Compute F1 score for each threshold
                f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
                
                # Find thresholds where recall >= min_recall
                valid_mask = recalls[:-1] >= self.min_recall
                
                if np.any(valid_mask):
                    # Among thresholds with sufficient recall, pick best F1
                    valid_f1 = np.where(valid_mask, f1_scores, -np.inf)
                    best_idx = np.argmax(valid_f1)
                    constraint_met = True
                else:
                    # If no threshold meets min_recall, pick highest recall
                    best_idx = np.argmax(recalls[:-1])
                    constraint_met = False
                
                self.best_threshold = thresholds[best_idx]
                
                # Compute metrics at selected threshold
                y_pred = (y_probs >= self.best_threshold).astype(int)
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = f1_scores[best_idx]
                
                status = "✓ MET" if constraint_met else "✗ NOT MET"
                print(f"\n  [Recall-Constrained F1] Threshold: {self.best_threshold:.4f}")
                print(f"    Constraint: Recall >= {self.min_recall*100:.0f}% {status}")
                print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                print(f"    (Sampled {len(y_true):,} validation examples)")
            except Exception as e:
                print(f"\n  [F1-Optimal] Threshold optimization failed: {e}")
    
    # Simple callback that just monitors validation metrics without prediction
    # No on_epoch_end callback to avoid prediction issues
    
    callbacks = [
        # No custom callback - just use standard Keras callbacks
        # AdaptiveThresholdCallback disabled - using fixed threshold=0.90 for high precision
        # AdaptiveThresholdCallback(
        #     val_inputs=[X_val_trend, X_val_seasonal, X_val_holiday, X_val_regressor, sku_val],
        #     val_targets=y_val,
        #     target_recall=float(cfg_val('target_recall', 0.70)),
        #     mode='target_recall',
        #     initial_threshold=initial_threshold
        # ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/lightweight_adaptive_best_weights.weights.h5',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,  # Save only weights to avoid serialization
            verbose=1
        )
    ]

    # Add ScheduledStopGradient ramp callback if present in the model
    class ScheduledStopRampCallback(keras.callbacks.Callback):
        def __init__(self, layer_name='scheduled_stop', total_epochs=10):
            super().__init__()
            self.layer_name = layer_name
            self.total_epochs = max(1, int(total_epochs))
            self._layer = None

        def on_train_begin(self, logs=None):
            try:
                self._layer = self.model.get_layer(self.layer_name)
            except Exception:
                self._layer = None

        def on_epoch_begin(self, epoch, logs=None):
            if self._layer is not None and hasattr(self._layer, 'stop_prob'):
                p = min(1.0, (epoch + 1) / float(self.total_epochs))
                try:
                    self._layer.stop_prob.assign(p)
                except Exception:
                    pass

    ramp_epochs = int(os.environ.get('RAMP_EPOCHS', '10'))
    callbacks.append(ScheduledStopRampCallback(total_epochs=ramp_epochs))
    
    print("\nStarting training...")
    print(f"Loss recipe: {loss_recipe}")
    start_time = time.time()
    
    batch_size = int(cfg_val('batch_size', best_params['batch_size']))
    epochs = int(cfg_val('epochs', 100))

    # three_term / bce_mae / spike_aware: per-SKU BCE imbalance via relative sample weights.
    # legacy keeps per-SKU forecast magnitude weights (log1p mean demand).
    n_skus_fit = int(max(int(sku_train.max()), int(sku_val.max())) + 1)
    sku_zr_info = estimate_zero_rate_by_sku(y_train, sku_train, n_skus=n_skus_fit)
    if loss_recipe in ("three_term", "bce_mae", "spike_aware"):
        fit_sample_weight = multioutput_bce_sample_weight_dict(
            y_train,
            sku_train,
            sku_zr_info["rates"],
            reference_zero_rate=float(zero_rate),
            other_keys=("final_forecast", "base_forecast"),
        )
        val_sample_weight = multioutput_bce_sample_weight_dict(
            y_val,
            sku_val,
            sku_zr_info["rates"],
            reference_zero_rate=float(zero_rate),
            other_keys=("final_forecast", "base_forecast"),
        )
        print(
            f"  Per-SKU BCE sample weights ON "
            f"(panel_zr={float(zero_rate):.3f}, n_skus={n_skus_fit})"
        )
        validation_data = (
            [X_val_trend, X_val_seasonal, X_val_holiday, X_val_regressor, sku_val],
            {
                "non_zero_probability": y_val_binary,
                "final_forecast": y_val,
                "base_forecast": y_val,
            },
            val_sample_weight,
        )
    else:
        fit_sample_weight = {
            "non_zero_probability": None,
            "final_forecast": sample_weights_train,
            "base_forecast": None,
        }
        validation_data = (
            [X_val_trend, X_val_seasonal, X_val_holiday, X_val_regressor, sku_val],
            {
                "non_zero_probability": y_val_binary,
                "final_forecast": y_val,
                "base_forecast": y_val,
            },
        )

    history = model.fit(
        [X_train_trend, X_train_seasonal, X_train_holiday,
         X_train_regressor, sku_train],
        {
            'non_zero_probability': y_train_binary,  # Binary classification target (for metrics only)
            'final_forecast': y_train,  # Final forecast target - composite loss handles both tasks
            'base_forecast': y_train  # Base forecast target (mag term when three_term)
        },
        sample_weight=fit_sample_weight,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")

    # ------------------------------------------------------------------
    # Post-training threshold sweep (validation) to suggest a better cut
    # ------------------------------------------------------------------
    try:
        from sklearn.metrics import precision_recall_curve

        sweep_cap = int(os.environ.get('THRESH_SWEEP_MAX_SAMPLES', '120000'))
        val_count = len(y_val_binary)
        if val_count > sweep_cap:
            idx = np.random.choice(val_count, sweep_cap, replace=False)
            val_inputs_sample = [
                X_val_trend[idx],
                X_val_seasonal[idx],
                X_val_holiday[idx],
                X_val_regressor[idx],
                sku_val[idx]
            ]
            y_val_sample = y_val_binary[idx]
        else:
            val_inputs_sample = [
                X_val_trend,
                X_val_seasonal,
                X_val_holiday,
                X_val_regressor,
                sku_val
            ]
            y_val_sample = y_val_binary

        print(f"\nRunning validation threshold sweep on {len(y_val_sample):,} samples (cap {sweep_cap:,})...")
        preds = model.predict(val_inputs_sample, batch_size=batch_size, verbose=0)
        y_probs = preds['non_zero_probability'].flatten()

        precisions, recalls, thresholds = precision_recall_curve(y_val_sample, y_probs)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
        best_f1 = f1_scores[best_idx]

        # Also capture threshold that meets a recall floor if possible
        recall_floor = float(os.environ.get('THRESH_RECALL_FLOOR', '0.70'))
        meets_floor = recalls[:-1] >= recall_floor
        if np.any(meets_floor):
            floor_idx = np.argmax(f1_scores * meets_floor)
            floor_threshold = thresholds[floor_idx]
            floor_precision = precisions[floor_idx]
            floor_recall = recalls[floor_idx]
            floor_f1 = f1_scores[floor_idx]
        else:
            floor_threshold = None
            floor_precision = None
            floor_recall = None
            floor_f1 = None

        print("Threshold sweep suggestions (validation):")
        print(f"  Best F1 threshold: {best_threshold:.4f} | P={best_precision:.4f}, R={best_recall:.4f}, F1={best_f1:.4f}")
        if floor_threshold is not None:
            print(f"  Recall≥{recall_floor:.0%} threshold: {floor_threshold:.4f} | P={floor_precision:.4f}, R={floor_recall:.4f}, F1={floor_f1:.4f}")
        else:
            print(f"  No threshold met recall floor {recall_floor:.0%}; best recall was {recalls.max():.4f}")
        print("  (Use these thresholds for inference instead of the fixed 0.55 if desired.)")
    except Exception:
        print("\nThreshold sweep failed; keeping fixed threshold. Trace:")
        print(traceback.format_exc())
    
    # Fixed weight summary
    print("\n" + "="*70)
    print("FIXED LOSS WEIGHTING")
    print("="*70)
    print(f"Classification weight: 90.0%")
    print(f"Forecast weight: 10.0%")
    print(f"\nClass weights for imbalance:")
    print(f"  Class 0 (zero) weight: {weight_zero:.2f}")
    print(f"  Class 1 (non-zero) weight: {weight_nonzero:.2f}")
    print("="*70)
    
    # Save base model separately for evaluation
    base_model.save('models/lightweight_adaptive_base.keras')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("Adaptive model saved to: models/lightweight_adaptive_best.keras")
    print("Base model saved to: models/lightweight_adaptive_base.keras")
    print("Run 'python evaluate_trained_model.py' for evaluation")
    print("  (Update model path to use adaptive version)")
    print("="*70)

    # Print key metrics at the end for quick inspection
    print(
        "loss:", history.history.get("loss", [None])[-1] if history else None,
        "non_zero_probability_loss:", history.history.get("non_zero_probability_loss", [None])[-1] if history else None,
        "val_non_zero_probability_nonzero_aucroc:", history.history.get("val_non_zero_probability_nonzero_aucroc", [None])[-1] if history else None,
    )


if __name__ == '__main__':
    main()
