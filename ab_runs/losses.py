"""Report which layers contribute regularization losses to the H>1 graph."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepsequence_hierarchical_attention.components_lightweight import (  # noqa: E402
    build_hierarchical_model_lightweight as build,
)

for horizon in (1, 6):
    model = build(
        n_temporal_features=1,
        n_fourier_features=4,
        n_holiday_features=0,
        n_lag_features=6,
        n_skus=20,
        hidden_dim=16,
        sku_embedding_dim=4,
        dropout_rate=0.1,
        horizon=horizon,
    )
    handler_present = "intermittent" in {layer.name for layer in model.layers}
    loss_layers = sorted(
        layer.name for layer in model.layers if getattr(layer, "losses", None)
    )
    print(f"horizon={horizon}")
    print(f"  IntermittentHandlerLightweight in graph : {handler_present}")
    print(f"  total loss terms                       : {len(model.losses)}")
    print(f"  layers adding losses                   : {loss_layers}")
