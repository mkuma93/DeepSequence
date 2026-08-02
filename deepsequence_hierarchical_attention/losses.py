"""
Custom loss functions for DeepSequence Hierarchical Attention.

Includes composite loss optimized for intermittent demand forecasting.
"""

from . import _tf_compat  # noqa: F401  # before TensorFlow

import tensorflow as tf


def composite_loss_forecast_only(alpha=0.5, bce_weight=None, zero_rate=None, average_nonzero_demand=None, 
                   pos_weight=9.0, use_mse=False):
    """
    DEPRECATED: This version incorrectly assumes y_pred contains both probability and magnitude.
    Use weighted_bce_loss + masked_mae_loss instead.
    
    For backward compatibility only - does not work correctly with multi-output models.
    """
    raise NotImplementedError(
        "composite_loss_forecast_only is deprecated. "
        "Use separate losses: weighted_bce_loss on 'non_zero_probability' + "
        "masked_mae_loss on 'final_forecast'"
    )


def weighted_bce_loss(pos_weight=9.0):
    """
    Weighted binary cross-entropy loss for intermittent demand classification.
    
    Class weighting:
    - Zero class weight: 1.0
    - Non-zero class weight: pos_weight (default 9.0)
    
    This gives 9x penalty for false negatives (predicting zero when demand is non-zero).

    Returns **per-sample** losses so Keras ``sample_weight`` (e.g. per-SKU
    relative weights from ``bce_sample_weights_from_sku_zero_rates``) applies
    correctly before the framework reduces.
    
    Args:
        pos_weight: Positive class weight (default 9.0)
            Higher = more penalty for false negatives
    
    Returns:
        Loss function for non_zero_probability output (expects probabilities in [0,1])
    """
    pos_weight_tensor = tf.constant(pos_weight, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        """
        Args:
            y_true: True demand values (will be converted to binary)
            y_pred: Predicted non-zero probability in [0, 1]
        """
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Binary target: 1 if demand > 0, else 0
        y_binary = tf.cast(y_true > 0, tf.float32)
        
        # Clip to avoid log(0) errors
        epsilon = 1e-7
        y_pred_safe = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Weighted BCE: -(y*log(p)*pos_weight + (1-y)*log(1-p))
        bce = -(
            y_binary * tf.math.log(y_pred_safe) * pos_weight_tensor +
            (1 - y_binary) * tf.math.log(1 - y_pred_safe)
        )
        
        return bce
    
    return loss_fn


def masked_mae_loss(use_mse=False, use_log_scale=False):
    """
    Masked MAE/MSE loss that only penalizes non-zero demand samples.
    
    This allows the model to learn magnitude prediction independently from
    zero classification, focusing forecast quality where demand actually exists.
    
    Args:
        use_mse: If True, use MSE instead of MAE (default False)
        use_log_scale: If True, compute on log1p scale (default False)
    
    Returns:
        Loss function for final_forecast output (expects demand values >= 0)
    """
    def loss_fn(y_true, y_pred):
        """
        Args:
            y_true: True demand values
            y_pred: Predicted demand values
        """
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Mask: only compute loss on non-zero samples
        non_zero_mask = tf.cast(y_true > 0, tf.float32)
        n_nonzero = tf.reduce_sum(non_zero_mask)
        
        if use_log_scale:
            # Log scale for better balance with BCE
            epsilon = 1e-7
            y_true_log = tf.math.log1p(tf.maximum(y_true, epsilon))
            y_pred_log = tf.math.log1p(tf.maximum(y_pred, epsilon))
            
            if use_mse:
                magnitude_loss = tf.square(y_true_log - y_pred_log) * non_zero_mask
            else:
                magnitude_loss = tf.abs(y_true_log - y_pred_log) * non_zero_mask
        else:
            # Original scale
            if use_mse:
                magnitude_loss = tf.square(y_true - y_pred) * non_zero_mask
            else:
                magnitude_loss = tf.abs(y_true - y_pred) * non_zero_mask
        
        # Average over non-zero samples only
        return tf.reduce_sum(magnitude_loss) / (n_nonzero + 1e-7)
    
    return loss_fn


def composite_loss(alpha=0.5, bce_weight=None, zero_rate=None, average_nonzero_demand=None, 
                   pos_weight=9.0, use_mse=False):
    """
    Factory function that returns SEPARATE losses for multi-output model.
    
    Returns a dictionary of losses:
    - 'non_zero_probability': weighted_bce_loss with pos_weight
    - 'final_forecast': masked_mae_loss (MAE/MSE on non-zero only)
    
    Use like this:
        losses = composite_loss(zero_rate=0.9, average_nonzero_demand=6.53, pos_weight=9.0)
        model.compile(
            optimizer='adam',
            loss=losses['losses'],
            loss_weights=losses['weights']
        )
    
    Args:
        zero_rate: Fraction of zeros (e.g., 0.90) - used to balance loss weights
        average_nonzero_demand: Average non-zero demand - used to scale MAE weight
        pos_weight: Class weight for non-zero samples in BCE (default 9.0)
        use_mse: Use MSE instead of MAE for forecast loss (default False)
    
    Returns:
        Dictionary with 'losses' and 'weights' for model.compile()
    """
    # Compute loss weights from data statistics
    if zero_rate is not None:
        # BCE weight: 1.0 (classification)
        # MAE weight: log1p(avg) for magnitude scaling
        bce_w = 1.0
        if average_nonzero_demand is not None:
            # Scale MAE by demand magnitude using log1p
            import numpy as np
            mae_w = float(np.log1p(average_nonzero_demand))
        else:
            mae_w = 1.0
    else:
        # Default balanced weighting
        bce_w = 0.5
        mae_w = 0.5
    
    # Omit base_forecast from losses: it is an auxiliary output and should not
    # contribute to training unless the caller adds an explicit loss.
    return {
        'losses': {
            'non_zero_probability': weighted_bce_loss(pos_weight=pos_weight),
            'final_forecast': masked_mae_loss(use_mse=use_mse, use_log_scale=False),
        },
        'weights': {
            'non_zero_probability': bce_w,
            'final_forecast': mae_w,
        }
    }


def base_forecast_mse(weight=0.4):
    """
    MSE loss for base_forecast (auxiliary task).
    
    Provides direct supervision for base_forecast to learn proper magnitude,
    independent of zero_probability. Only computed on non-zero samples.
    
    Theory: Multi-task learning to decouple magnitude prediction from classification.
    - Base forecast learns: "What is the demand when non-zero?"
    - Zero probability learns: "Will there be demand?"
    
    Args:
        weight: Weight for the MSE loss (default 0.4)
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    final_weight = tf.constant(weight, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        """
        Compute MSE on base_forecast for non-zero samples only.
        
        Args:
            y_true: True demand values
            y_pred: Base forecast predictions (before zero probability adjustment)
        
        Returns:
            Scalar MSE loss
        """
        # Flatten inputs
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Mask to non-zero samples only
        non_zero_mask = tf.cast(y_true > 0, tf.float32)
        n_nonzero = tf.reduce_sum(non_zero_mask) + 1e-7
        
        # MSE on non-zero samples
        squared_error = tf.square(y_true - y_pred) * non_zero_mask
        mse = tf.reduce_sum(squared_error) / n_nonzero
        
        return final_weight * mse
    
    return loss_fn


def base_forecast_mae(weight=0.4):
    """
    MAE loss for base_forecast (auxiliary task).
    
    Provides direct supervision for base_forecast to learn proper magnitude,
    independent of zero_probability. Only computed on non-zero samples.
    
    Theory: Multi-task learning to decouple magnitude prediction from classification.
    - Base forecast learns: "What is the demand when non-zero?"
    - Zero probability learns: "Will there be demand?"
    
    Args:
        weight: Weight for the MAE loss (default 0.4)
    
    Returns:
        Loss function compatible with Keras model.compile()
    """
    final_weight = tf.constant(weight, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        """
        Compute MAE on base_forecast for non-zero samples only.
        
        Args:
            y_true: True demand values
            y_pred: Base forecast predictions (before zero probability adjustment)
        
        Returns:
            Scalar MAE loss
        """
        # Flatten inputs
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Mask to non-zero samples only
        non_zero_mask = tf.cast(y_true > 0, tf.float32)
        n_nonzero = tf.reduce_sum(non_zero_mask) + 1e-7
        
        # MAE on non-zero samples
        absolute_error = tf.abs(y_true - y_pred) * non_zero_mask
        mae = tf.reduce_sum(absolute_error) / n_nonzero
        
        return final_weight * mae
    
    return loss_fn


def sku_aware_composite_loss(alpha=0.9):
    """
    SKU-aware composite loss with volume-based MAE weighting.
    
    Formula: alpha * BCE + log1p(sku_mean_demand) * MAE
    
    This variant weights MAE by log1p of SKU mean demand, giving more
    importance to forecast accuracy on high-volume SKUs while avoiding
    extreme weights for very high-demand items.
    
    Args:
        alpha: Weight for BCE component (default: 0.9 for 90% zero rate)
    
    Returns:
        Loss function compatible with Keras model.compile()
        
    Note:
        This requires SKU mean demand to be computed externally and passed
        as a constant. For dynamic computation, use a custom training loop.
    """
    def loss_fn(y_true, y_pred):
        """
        Compute SKU-aware composite loss.
        
        Args:
            y_true: True demand values (shape: [batch_size,])
            y_pred: Predicted demand values (shape: [batch_size,])
        
        Returns:
            Scalar loss value
        """
        # Flatten inputs
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Binary target: 1 if demand > 0, else 0
        y_binary = tf.cast(y_true > 0, tf.float32)
        
        # Convert predictions to binary probabilities
        y_pred_binary = tf.nn.sigmoid(y_pred / 10.0)
        
        # Binary cross-entropy loss (zero detection)
        bce_loss = tf.keras.losses.binary_crossentropy(y_binary, y_pred_binary)
        
        # Mean absolute error loss (magnitude prediction)
        mae_loss = tf.abs(y_true - y_pred)
        
        # For this batch, use batch-level NON-ZERO mean as proxy for SKU mean
        # In practice, this should be replaced with actual SKU mean demand
        non_zero_mask = tf.cast(y_true > 0, tf.float32)
        n_nonzero = tf.reduce_sum(non_zero_mask) + 1e-7
        batch_mean_nonzero = tf.reduce_sum(y_true * non_zero_mask) / n_nonzero
        mae_weight = tf.math.log1p(batch_mean_nonzero)
        
        # Combined loss: alpha * BCE + log1p(mean) * MAE
        combined = alpha * bce_loss + mae_weight * mae_loss
        
        return tf.reduce_mean(combined)
    
    return loss_fn


def weighted_composite_loss(alpha=0.5):
    """
    SKU-weighted composite loss for prioritizing high-volume SKUs.
    
    Similar to composite_loss but supports sample weights to emphasize
    important SKUs (e.g., high-revenue or high-volume products).
    
    Args:
        alpha: Weight for BCE component (default: 0.5)
    
    Returns:
        Loss function compatible with Keras model.compile()
    
    Example:
        >>> # Calculate SKU weights
        >>> sku_mean_demand = df.groupby('sku')['demand'].mean()
        >>> sample_weights = np.log1p(sku_mean_demand[sku_ids].values)
        >>> 
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss={'final_forecast': weighted_composite_loss(alpha=0.5)}
        ... )
        >>> model.fit(X, y, sample_weight=sample_weights)
    """
    def loss_fn(y_true, y_pred, sample_weight=None):
        """
        Compute weighted composite loss.
        
        Args:
            y_true: True demand values
            y_pred: Predicted demand values
            sample_weight: Optional weights per sample (e.g., log1p(mean_demand))
        
        Returns:
            Scalar loss value
        """
        # Binary target
        y_binary = tf.cast(y_true > 0, tf.float32)
        y_pred_binary = tf.nn.sigmoid(y_pred / 10.0)
        
        # Binary cross-entropy (no weighting for zero detection)
        bce_loss = tf.keras.losses.binary_crossentropy(y_binary, y_pred_binary)
        
        # MAE with optional weighting
        mae_per_sample = tf.abs(y_true - y_pred)
        
        if sample_weight is not None:
            weighted_mae = mae_per_sample * sample_weight
        else:
            weighted_mae = mae_per_sample
        
        # Combined loss
        combined = alpha * bce_loss + weighted_mae
        
        return tf.reduce_mean(combined)
    
    return loss_fn


def mae_loss():
    """
    Simple MAE loss wrapper for consistency.
    
    Returns:
        MAE loss function
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1, 1])
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    return loss_fn


def inverse_class_weights(zero_rate):
    """Balanced inverse-frequency weights for zero vs non-zero days."""
    zr = float(zero_rate)
    nz = max(1.0 - zr, 1e-6)
    w_zero = 1.0 / (2.0 * max(zr, 1e-6))
    w_nonzero = 1.0 / (2.0 * nz)
    return w_zero, w_nonzero


def inverse_weighted_mae_loss(zero_rate):
    """
    MAE on ALL days with inverse zero/nonzero class weights.

    Keeps quiet-day timing signal while upweighting rare sale days.
    Shapes are forced to [batch, 1] to avoid (N,) vs (N,1) broadcast bugs.
    """
    w_zero, w_nonzero = inverse_class_weights(zero_rate)
    w_zero_t = tf.constant(w_zero, dtype=tf.float32)
    w_nonzero_t = tf.constant(w_nonzero, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1, 1])
        is_nz = tf.cast(y_true > 0, tf.float32)
        w = w_zero_t * (1.0 - is_nz) + w_nonzero_t * is_nz
        return tf.reduce_sum(w * tf.abs(y_true - y_pred)) / tf.reduce_sum(w)

    return loss_fn


def all_days_mae_loss():
    """Unweighted MAE on all days (shape-safe)."""
    def loss_fn(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1, 1])
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    return loss_fn


def three_term_loss_config(
    zero_rate,
    alpha_bce=0.2,
    w_gated=1.0,
    w_mag=1.0,
    pos_weight=None,
):
    """
    Recommended intermittent recipe for model.compile():

      L = w_gated * inverse_weighted_MAE(y, p*softplus)   # all days (timing)
        + w_mag   * masked_MAE(y, softplus)               # sale days (size)
        + alpha_bce * weighted_BCE(y>0, p)                # sale-or-not

    Returns dict with 'losses' and 'weights' keyed by output name.
    """
    if pos_weight is None:
        pos_weight = min(20.0, float(zero_rate) / max(1.0 - float(zero_rate), 1e-6))

    return {
        "recipe": "three_term",
        "losses": {
            "non_zero_probability": weighted_bce_loss(pos_weight=pos_weight),
            "final_forecast": inverse_weighted_mae_loss(zero_rate),
            "base_forecast": masked_mae_loss(use_mse=False, use_log_scale=False),
        },
        "weights": {
            "non_zero_probability": float(alpha_bce),
            "final_forecast": float(w_gated),
            "base_forecast": float(w_mag),
        },
        "meta": {
            "alpha_bce": float(alpha_bce),
            "w_gated": float(w_gated),
            "w_mag": float(w_mag),
            "pos_weight": float(pos_weight),
            "w_zero": inverse_class_weights(zero_rate)[0],
            "w_nonzero": inverse_class_weights(zero_rate)[1],
        },
    }


def bce_mae_loss_config(zero_rate, bce_weight=0.5, mae_weight=0.5, pos_weight=None, mae_all_days=True):
    """
    Two-term recipe: weighted BCE + MAE on final gated forecast.
    mae_all_days=True keeps timing; False matches legacy nonzero-only MAE.
    """
    if pos_weight is None:
        pos_weight = min(20.0, float(zero_rate) / max(1.0 - float(zero_rate), 1e-6))

    forecast_loss = (
        all_days_mae_loss()
        if mae_all_days
        else masked_mae_loss(use_mse=False, use_log_scale=False)
    )
    return {
        "recipe": "bce_mae",
        "losses": {
            "non_zero_probability": weighted_bce_loss(pos_weight=pos_weight),
            "final_forecast": forecast_loss,
        },
        "weights": {
            "non_zero_probability": float(bce_weight),
            "final_forecast": float(mae_weight),
        },
        "meta": {
            "bce_weight": float(bce_weight),
            "mae_weight": float(mae_weight),
            "pos_weight": float(pos_weight),
            "mae_all_days": bool(mae_all_days),
        },
    }


def masked_poisson_nll_loss():
    """
    Poisson NLL on demand days only (hurdle magnitude head).

    Assumes ``y_pred`` is already a positive rate (e.g. softplus base).
    Ignores the factorial term (constant w.r.t. parameters).
    """

    def loss_fn(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        lam = tf.reshape(tf.cast(y_pred, tf.float32), [-1, 1])
        lam = tf.clip_by_value(lam, 1e-5, 1e6)
        # NLL ∝ λ - y log λ  (drop log(y!))
        nll = lam - y_true * tf.math.log(lam)
        mask = tf.cast(y_true > 0, tf.float32)
        return tf.reduce_sum(mask * nll) / (tf.reduce_sum(mask) + 1e-6)

    return loss_fn


def hurdle_poisson_loss_config(
    zero_rate,
    alpha_bce=1.0,
    w_mag=1.0,
    pos_weight=None,
):
    """
    Classic intermittent / hurdle loss for gated models:

      L = α * weighted_BCE(y>0, p) + w_mag * PoissonNLL(y | λ) |_{y>0}

    where ``p = non_zero_probability`` and ``λ = base_forecast`` (softplus rate).
    ``final_forecast = p * λ`` is not trained with MAE — occurrence and size
    are separated the way intermittent demand models usually do.
    """
    if pos_weight is None:
        pos_weight = min(20.0, float(zero_rate) / max(1.0 - float(zero_rate), 1e-6))

    return {
        "recipe": "hurdle_poisson",
        "losses": {
            "non_zero_probability": weighted_bce_loss(pos_weight=pos_weight),
            "base_forecast": masked_poisson_nll_loss(),
            # Keep a tiny final term so Keras still tracks the head; weight ~0
            "final_forecast": all_days_mae_loss(),
        },
        "weights": {
            "non_zero_probability": float(alpha_bce),
            "base_forecast": float(w_mag),
            "final_forecast": 1e-4,
        },
        "meta": {
            "alpha_bce": float(alpha_bce),
            "w_mag": float(w_mag),
            "pos_weight": float(pos_weight),
            "note": "hurdle: BCE occurrence + Poisson size on nonzero days",
        },
    }


def tweedie_deviance_loss(power: float = 1.5):
    """
    Tweedie deviance loss on a positive mean prediction (1 < power < 2).

    Compatible with intermittent / zero-inflated count-like targets; ``y=0``
    is valid in this power range.
    """
    p = float(power)
    if not (1.0 < p < 2.0):
        raise ValueError("tweedie_deviance_loss expects 1 < power < 2")

    def loss_fn(y_true, y_pred):
        y = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        y = tf.maximum(y, 0.0)
        mu = tf.reshape(tf.cast(y_pred, tf.float32), [-1, 1])
        mu = tf.clip_by_value(mu, 1e-5, 1e6)
        # d = 2 * ( y^{2-p}/((1-p)(2-p)) - y μ^{1-p}/(1-p) + μ^{2-p}/(2-p) )
        term_y = tf.pow(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
        term_ymu = y * tf.pow(mu, 1.0 - p) / (1.0 - p)
        term_mu = tf.pow(mu, 2.0 - p) / (2.0 - p)
        deviance = 2.0 * (term_y - term_ymu + term_mu)
        return tf.reduce_mean(deviance)

    return loss_fn


def tweedie_loss_config(power: float = 1.5, bce_weight: float = 0.0, pos_weight: float = 1.0):
    """
    Train gated sequence models with Tweedie deviance on ``final_forecast``.

    Optional light BCE on ``non_zero_probability`` (default off) if you still
    want an explicit occurrence head signal.
    """
    losses = {
        "final_forecast": tweedie_deviance_loss(power=power),
        "base_forecast": all_days_mae_loss(),
        "non_zero_probability": weighted_bce_loss(pos_weight=pos_weight),
    }
    weights = {
        "final_forecast": 1.0,
        "base_forecast": 1e-4,
        "non_zero_probability": float(bce_weight),
    }
    return {
        "recipe": "tweedie",
        "losses": losses,
        "weights": weights,
        "meta": {"power": float(power), "bce_weight": float(bce_weight)},
    }


def resolve_positive_bce_weight(
    zero_rate,
    positive_bce_weight=None,
    *,
    boost=2.0,
    cap=20.0,
):
    """Resolve spike-aware positive-class BCE weight.

    Default bake-off ``three_term`` uses ``zr / (1-zr)`` (capped). Spike-aware
    defaults to ``boost * zr / (1-zr)`` so sale days dominate the gate loss.
    Pass ``positive_bce_weight`` to override explicitly.
    """
    if positive_bce_weight is not None:
        w = float(positive_bce_weight)
    else:
        zr = float(zero_rate)
        w = float(boost) * zr / max(1.0 - zr, 1e-6)
    if cap is None:
        return float(w)
    return float(min(float(cap), w))


def spike_aware_magnitude_loss(zero_mag_weight=0.05, use_mse=False):
    """
    Magnitude loss primarily on ``y > 0``, with optional light zero-day term.

    On sale days the loss matches ``masked_mae_loss`` / MSE. On zero days a
    small ``zero_mag_weight`` keeps the magnitude head ``b`` from drifting
    unbounded when the gate carries quiet-day calibration.
    """
    z_w = tf.constant(float(zero_mag_weight), dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1, 1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1, 1])
        is_nz = tf.cast(y_true > 0, tf.float32)
        err = (
            tf.square(y_true - y_pred)
            if use_mse
            else tf.abs(y_true - y_pred)
        )
        n_nz = tf.reduce_sum(is_nz) + 1e-6
        n_z = tf.reduce_sum(1.0 - is_nz) + 1e-6
        mag_pos = tf.reduce_sum(is_nz * err) / n_nz
        mag_zero = tf.reduce_sum((1.0 - is_nz) * err) / n_z
        return mag_pos + z_w * mag_zero

    return loss_fn


def focal_weighted_bce_loss(pos_weight=9.0, gamma=0.0):
    """
    Weighted BCE with optional focal modulation on the occurrence head.

    ``gamma=0`` recovers ``weighted_bce_loss``. ``gamma>0`` down-weights easy
    examples: ``(1-p)^γ`` on positives, ``p^γ`` on zeros.
    """
    pos_w = tf.constant(float(pos_weight), dtype=tf.float32)
    g = float(gamma)

    def loss_fn(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        y_binary = tf.cast(y_true > 0, tf.float32)
        eps = 1e-7
        p = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        if g > 0.0:
            focal_pos = tf.pow(1.0 - p, g)
            focal_neg = tf.pow(p, g)
        else:
            focal_pos = 1.0
            focal_neg = 1.0
        bce = -(
            y_binary * focal_pos * tf.math.log(p) * pos_w
            + (1.0 - y_binary) * focal_neg * tf.math.log(1.0 - p)
        )
        return bce

    return loss_fn


def spike_aware_loss_config(
    zero_rate,
    *,
    alpha_bce=1.0,
    w_mag=1.0,
    zero_mag_weight=0.05,
    w_gated=0.0,
    positive_bce_weight=None,
    positive_bce_boost=2.0,
    pos_weight_cap=20.0,
    focal_gamma=0.0,
    use_mse=False,
):
    """
    Opt-in **spike-aware** intermittent recipe (default bake-off unchanged).

    Emphasizes sale-day occurrence and positive-only magnitude while keeping
    ``ŷ = p · b``:

      L = α · BCE_{w+}(z, p)               # heavy positive weight / optional focal
        + w_mag · Mag_{nz}(y, b)           # primary magnitude on y>0
          + zero_mag_weight · Mag_z(y, b)  # optional quiet-day b calibration
        + w_gated · MAE_inv(y, ŷ)          # optional timing (default 0)

    Knobs
    -----
    alpha_bce:
        Weight on the occurrence / gate BCE term (default 1.0; three_term uses 0.2).
    positive_bce_weight:
        Absolute positive-class BCE weight. ``None`` →
        ``positive_bce_boost * zr/(1-zr)`` capped at ``pos_weight_cap``.
    positive_bce_boost:
        Multiplier on the standard imbalance weight when
        ``positive_bce_weight`` is omitted (default 2.0).
    focal_gamma:
        Focal focusing parameter; ``0`` disables focal modulation.
    w_mag:
        Weight on the (mostly positive-only) magnitude term against ``b``.
    zero_mag_weight:
        Relative weight of zero-day magnitude vs sale-day magnitude inside
        the mag head (default 0.05; ``0`` = positives only).
    w_gated:
        Optional inverse-class-weighted MAE on ``ŷ=p·b`` (default 0 / off).
    use_mse:
        Use squared error instead of MAE for the magnitude head.

    Default locked bake-off remains ``three_term``; this recipe is opt-in via
    ``loss_recipe='spike_aware'``.
    """
    pos_w = resolve_positive_bce_weight(
        zero_rate,
        positive_bce_weight,
        boost=positive_bce_boost,
        cap=pos_weight_cap,
    )
    losses = {
        "non_zero_probability": focal_weighted_bce_loss(
            pos_weight=pos_w, gamma=focal_gamma
        ),
        "base_forecast": spike_aware_magnitude_loss(
            zero_mag_weight=zero_mag_weight, use_mse=use_mse
        ),
    }
    weights = {
        "non_zero_probability": float(alpha_bce),
        "base_forecast": float(w_mag),
    }
    # Keep final_forecast tracked; weight 0 unless timing term requested.
    if float(w_gated) > 0.0:
        losses["final_forecast"] = inverse_weighted_mae_loss(zero_rate)
        weights["final_forecast"] = float(w_gated)
    else:
        losses["final_forecast"] = all_days_mae_loss()
        weights["final_forecast"] = 1e-4

    return {
        "recipe": "spike_aware",
        "losses": losses,
        "weights": weights,
        "meta": {
            "alpha_bce": float(alpha_bce),
            "w_mag": float(w_mag),
            "zero_mag_weight": float(zero_mag_weight),
            "w_gated": float(w_gated),
            "positive_bce_weight": float(pos_w),
            "positive_bce_boost": float(positive_bce_boost),
            "pos_weight_cap": float(pos_weight_cap) if pos_weight_cap is not None else None,
            "focal_gamma": float(focal_gamma),
            "use_mse": bool(use_mse),
            "w_zero": inverse_class_weights(zero_rate)[0],
            "w_nonzero": inverse_class_weights(zero_rate)[1],
            "note": (
                "spike-aware: heavy positive BCE + positive-only magnitude; "
                "ŷ=p·b unchanged; bake-off default remains three_term"
            ),
        },
    }
