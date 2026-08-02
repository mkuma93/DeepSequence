"""TensorFlow / Keras version compatibility for tensorflow-recommenders.

TF >= 2.16 ships Keras 3 by default. ``tensorflow_recommenders`` still requires
the Keras 2 API via ``tf.keras``, so set ``TF_USE_LEGACY_KERAS=1`` before the
first TensorFlow import. Import this module first from any TF entrypoint.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
