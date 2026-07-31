"""
Hybrid temporal DeepSequence: hierarchical experts + causal lookback encoder.

End-to-end design (not freeze-then-residual):
  - Current-row hierarchical experts → structural magnitude prior
  - Causal MHA over lookback (Quantity + causal features) → temporal context
  - Temporal context fused into magnitude ``b`` and occurrence ``p``
  - Decoupled gate: raw regressors + SKU + temporal context (no softplus base /
    component scalars)

Paper path: call ``build_hierarchical_model_lightweight`` with defaults
(``use_temporal_context=False``). This module is the explicit hybrid entrypoint.
"""

from __future__ import annotations

from .components_lightweight import build_hierarchical_model_lightweight


def build_hierarchical_model_hybrid(
    n_temporal_features,
    n_fourier_features,
    n_holiday_features,
    n_lag_features,
    n_skus,
    n_sequence_channels,
    *,
    lookback=14,
    temporal_d_model=32,
    temporal_n_heads=4,
    temporal_n_blocks=1,
    decouple_gate=True,
    **kwargs,
):
    """
    Build hybrid tabular+temporal DeepSequence.

    Model inputs (in order)::

        [temporal, fourier, holiday, lag, sku_id, sequence_history]

    where ``sequence_history`` has shape ``[B, lookback, n_sequence_channels]``
    matching the TST bake-off channels ``[Quantity, X...]`` for past days only.
    """
    return build_hierarchical_model_lightweight(
        n_temporal_features=n_temporal_features,
        n_fourier_features=n_fourier_features,
        n_holiday_features=n_holiday_features,
        n_lag_features=n_lag_features,
        n_skus=n_skus,
        use_temporal_context=True,
        lookback=lookback,
        n_sequence_channels=n_sequence_channels,
        temporal_d_model=temporal_d_model,
        temporal_n_heads=temporal_n_heads,
        temporal_n_blocks=temporal_n_blocks,
        decouple_gate=decouple_gate,
        **kwargs,
    )


__all__ = ["build_hierarchical_model_hybrid"]
