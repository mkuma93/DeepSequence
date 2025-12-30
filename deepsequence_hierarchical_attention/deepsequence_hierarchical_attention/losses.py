"""
Custom loss functions for DeepSequence Hierarchical Attention.

Includes composite loss optimized for intermittent demand forecasting.
"""

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
        
        return tf.reduce_mean(bce)
    
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
    
    return {
        'losses': {
            'non_zero_probability': weighted_bce_loss(pos_weight=pos_weight),
            'final_forecast': masked_mae_loss(use_mse=use_mse, use_log_scale=False),
            'base_forecast': None  # Auxiliary output, no loss
        },
        'weights': {
            'non_zero_probability': bce_w,
            'final_forecast': mae_w,
            'base_forecast': 0.0
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
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    return loss_fn
