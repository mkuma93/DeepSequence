"""
Temporal Calibration Model - Post-hoc refinement using Multi-Head Attention.

Two-stage approach:
1. Base model (hierarchical attention) generates forecasts per timestep
2. Calibration model reshapes forecasts by SKU → applies MHA → outputs calibrated forecasts

Advantages:
- Base model stays flexible (no fixed sequence length)
- Temporal calibration is optional and can be applied offline
- MHA learns temporal patterns and corrections
- Works with any base forecasting model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, MultiHeadAttention, LayerNormalization, Dense, 
    Dropout, Add, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from typing import Tuple, Dict, List, Optional


def prepare_sequences_for_calibration(
    df: pd.DataFrame,
    forecast_col: str = 'forecast',
    actual_col: str = 'actual',
    sku_col: str = 'sku_id',
    date_col: str = 'date',
    window_size: Optional[int] = None,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Reshape forecasts into sequences [n_skus, time_steps, features] for MHA.
    
    Args:
        df: DataFrame with columns [sku_id, date, forecast, actual, ...]
        forecast_col: Column name for base model forecasts
        actual_col: Column name for actual values
        sku_col: Column name for SKU identifier
        date_col: Column name for date/time
        window_size: If None, use all timesteps per SKU. Otherwise, sliding window.
        stride: Stride for sliding window (default: 1)
        
    Returns:
        X_seq: [n_sequences, time_steps, 1] - forecast sequences
        y_seq: [n_sequences, time_steps, 1] - actual sequences
        sku_indices: [n_sequences] - SKU index for each sequence
        metadata: DataFrame with sequence metadata
    """
    # Sort by SKU and date
    df = df.sort_values([sku_col, date_col]).reset_index(drop=True)
    
    # Encode SKU IDs
    unique_skus = df[sku_col].unique()
    sku_to_idx = {sku: idx for idx, sku in enumerate(unique_skus)}
    df['sku_idx'] = df[sku_col].map(sku_to_idx)
    
    sequences_X = []
    sequences_y = []
    sku_indices = []
    metadata_rows = []
    
    for sku_idx, sku_id in enumerate(unique_skus):
        sku_data = df[df[sku_col] == sku_id].sort_values(date_col)
        forecasts = sku_data[forecast_col].values
        actuals = sku_data[actual_col].values
        
        if window_size is None:
            # Use entire sequence for this SKU
            sequences_X.append(forecasts.reshape(-1, 1))
            sequences_y.append(actuals.reshape(-1, 1))
            sku_indices.append(sku_idx)
            
            metadata_rows.append({
                'sku_id': sku_id,
                'sku_idx': sku_idx,
                'start_date': sku_data[date_col].iloc[0],
                'end_date': sku_data[date_col].iloc[-1],
                'n_timesteps': len(forecasts)
            })
        else:
            # Sliding window approach
            for i in range(0, len(forecasts) - window_size + 1, stride):
                window_forecasts = forecasts[i:i + window_size]
                window_actuals = actuals[i:i + window_size]
                
                sequences_X.append(window_forecasts.reshape(-1, 1))
                sequences_y.append(window_actuals.reshape(-1, 1))
                sku_indices.append(sku_idx)
                
                metadata_rows.append({
                    'sku_id': sku_id,
                    'sku_idx': sku_idx,
                    'window_idx': i // stride,
                    'start_date': sku_data[date_col].iloc[i],
                    'end_date': sku_data[date_col].iloc[i + window_size - 1],
                    'n_timesteps': window_size
                })
    
    # Pad sequences to same length if needed
    if window_size is None:
        max_len = max(seq.shape[0] for seq in sequences_X)
        sequences_X_padded = []
        sequences_y_padded = []
        
        for X_seq, y_seq in zip(sequences_X, sequences_y):
            if X_seq.shape[0] < max_len:
                # Pad with zeros
                pad_len = max_len - X_seq.shape[0]
                X_seq = np.vstack([X_seq, np.zeros((pad_len, 1))])
                y_seq = np.vstack([y_seq, np.zeros((pad_len, 1))])
            sequences_X_padded.append(X_seq)
            sequences_y_padded.append(y_seq)
            
        X_seq = np.array(sequences_X_padded)
        y_seq = np.array(sequences_y_padded)
    else:
        X_seq = np.array(sequences_X)
        y_seq = np.array(sequences_y)
    
    sku_indices = np.array(sku_indices)
    metadata = pd.DataFrame(metadata_rows)
    
    return X_seq, y_seq, sku_indices, metadata


def build_temporal_calibration_model(
    time_steps: int,
    n_skus: int,
    num_heads: int = 4,
    key_dim: int = 32,
    ff_dim: int = 64,
    num_transformer_blocks: int = 2,
    dropout: float = 0.1,
    embedding_dim: int = 32
) -> Model:
    """
    Build temporal calibration model using Multi-Head Attention.
    
    Architecture:
        Input: [n_sequences, time_steps, 1] forecasts
        ↓
        Positional Encoding + SKU Embedding
        ↓
        N × Transformer Blocks (MHA + FFN)
        ↓
        Output: [n_sequences, time_steps, 1] calibrated forecasts
        
    Args:
        time_steps: Maximum sequence length
        n_skus: Number of unique SKUs
        num_heads: Number of attention heads
        key_dim: Dimension of attention keys
        ff_dim: Feed-forward network dimension
        num_transformer_blocks: Number of transformer blocks
        dropout: Dropout rate
        embedding_dim: SKU embedding dimension
        
    Returns:
        Keras Model for temporal calibration
    """
    # Inputs
    forecast_input = Input(shape=(time_steps, 1), name='forecast_input')
    sku_input = Input(shape=(1,), dtype='int32', name='sku_input')
    
    # SKU embedding
    sku_embedding = tf.keras.layers.Embedding(
        n_skus, embedding_dim, name='sku_embedding'
    )(sku_input)
    sku_embedding = tf.keras.layers.Flatten()(sku_embedding)
    
    # Broadcast SKU embedding across time steps
    sku_embedding_broadcast = tf.keras.layers.RepeatVector(time_steps)(sku_embedding)
    
    # Positional encoding
    positions = tf.range(start=0, limit=time_steps, delta=1)
    position_embedding = tf.keras.layers.Embedding(
        time_steps, embedding_dim, name='position_embedding'
    )(positions)
    position_embedding = tf.expand_dims(position_embedding, axis=0)  # [1, time_steps, embedding_dim]
    position_embedding = tf.tile(position_embedding, [tf.shape(forecast_input)[0], 1, 1])
    
    # Combine forecast with embeddings
    x = Concatenate(axis=-1)([
        forecast_input,
        sku_embedding_broadcast,
        position_embedding
    ])  # [batch, time_steps, 1 + 2*embedding_dim]
    
    # Project to model dimension
    x = Dense(
        ff_dim,
        activation='relu',
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
        name='input_projection'
    )(x)
    x = LayerNormalization(name='input_norm')(x)
    
    # Transformer blocks
    for i in range(num_transformer_blocks):
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            name=f'mha_block_{i}'
        )(x, x)
        attn_output = Dropout(dropout)(attn_output)
        
        # Add & Norm (residual connection)
        x1 = Add()([x, attn_output])
        x1 = LayerNormalization(name=f'norm1_block_{i}')(x1)
        
        # Feed-Forward Network
        ffn = Dense(
            ff_dim * 2,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
            name=f'ffn1_block_{i}'
        )(x1)
        ffn = Dropout(dropout)(ffn)
        ffn = Dense(
            ff_dim,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
            name=f'ffn2_block_{i}'
        )(ffn)
        ffn = Dropout(dropout)(ffn)
        
        # Add & Norm
        x = Add()([x1, ffn])
        x = LayerNormalization(name=f'norm2_block_{i}')(x)
    
    # Output projection to calibrated forecast
    calibrated_forecast = Dense(
        1,
        activation='linear',
        name='calibrated_forecast'
    )(x)  # [batch, time_steps, 1]
    
    # Build model
    model = Model(
        inputs=[forecast_input, sku_input],
        outputs=calibrated_forecast,
        name='temporal_calibration_model'
    )
    
    return model


class TemporalCalibrationModel:
    """
    Wrapper for temporal calibration workflow.
    
    Usage:
        # 1. Train base model and generate forecasts
        base_model.fit(...)
        train_forecasts = base_model.predict(X_train)
        
        # 2. Prepare sequences for calibration
        calibrator = TemporalCalibrationModel(window_size=30)
        X_seq, y_seq, sku_idx, metadata = calibrator.prepare_data(
            train_df, forecast_col='base_forecast', actual_col='demand'
        )
        
        # 3. Build and train calibration model
        calibrator.build(time_steps=30, n_skus=6099)
        calibrator.fit(X_seq, y_seq, sku_idx, epochs=50)
        
        # 4. Apply calibration to new forecasts
        calibrated_forecasts = calibrator.predict(val_forecasts, val_sku_idx)
    """
    
    def __init__(
        self,
        window_size: Optional[int] = None,
        num_heads: int = 4,
        key_dim: int = 32,
        ff_dim: int = 64,
        num_transformer_blocks: int = 2,
        dropout: float = 0.1,
        embedding_dim: int = 32
    ):
        self.window_size = window_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.model = None
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        forecast_col: str = 'forecast',
        actual_col: str = 'actual',
        sku_col: str = 'sku_id',
        date_col: str = 'date',
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepare sequences for calibration."""
        return prepare_sequences_for_calibration(
            df=df,
            forecast_col=forecast_col,
            actual_col=actual_col,
            sku_col=sku_col,
            date_col=date_col,
            window_size=self.window_size,
            stride=stride
        )
    
    def build(self, time_steps: int, n_skus: int):
        """Build calibration model."""
        self.model = build_temporal_calibration_model(
            time_steps=time_steps,
            n_skus=n_skus,
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            dropout=self.dropout,
            embedding_dim=self.embedding_dim
        )
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
    def fit(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        sku_indices: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """Train calibration model."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        history = self.model.fit(
            [X_seq, sku_indices.reshape(-1, 1)],
            y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return history
    
    def predict(
        self,
        X_seq: np.ndarray,
        sku_indices: np.ndarray
    ) -> np.ndarray:
        """Generate calibrated forecasts."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        calibrated = self.model.predict(
            [X_seq, sku_indices.reshape(-1, 1)]
        )
        
        return calibrated
