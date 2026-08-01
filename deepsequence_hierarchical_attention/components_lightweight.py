import numpy as np
import logging

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Concatenate, Add, Multiply, Dropout,
    LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import UnitNorm
from tensorflow import keras
# TF 2.21 + Keras 3: `tensorflow.keras` may omit `.saving` (present on standalone keras).
if not hasattr(keras, "saving"):
    import keras as _keras3

    keras.saving = _keras3.saving
import tensorflow_recommenders as tfrs

logger = logging.getLogger(__name__)

KERAS_PACKAGE = "deepsequence_hierarchical_attention"

# Fourier periods are expressed in *time steps of the series*, so they only make
# sense together with a sampling frequency. Calendar months and years are not
# whole numbers of days (28-31 day months, leap years), so the daily periods use
# the mean Gregorian year rather than 30/91/365.
DAYS_PER_YEAR = 365.25
DAYS_PER_QUARTER = DAYS_PER_YEAR / 4  # 91.3125
DAYS_PER_MONTH = DAYS_PER_YEAR / 12  # 30.4375
WEEKS_PER_YEAR = DAYS_PER_YEAR / 7  # 52.178571...

FOURIER_PERIODS_BY_FREQUENCY = {
    # daily steps: week, fortnight, mean month, mean quarter, mean year
    "daily": (7.0, 14.0, DAYS_PER_MONTH, DAYS_PER_QUARTER, DAYS_PER_YEAR),
    # weekly steps: mean month, quarter, half-year, year
    "weekly": (
        WEEKS_PER_YEAR / 12,
        WEEKS_PER_YEAR / 4,
        WEEKS_PER_YEAR / 2,
        WEEKS_PER_YEAR,
    ),
    # monthly steps: exact integers, no calendar drift
    "monthly": (3.0, 6.0, 12.0),
    "quarterly": (2.0, 4.0),
}

# Longest learnable period, as a multiple of the longest initial period. Lets an
# annual cycle drift without allowing meaningless multi-decade frequencies.
FOURIER_MAX_PERIOD_SLACK = 2.0

# Nyquist limit: a cycle shorter than two steps is unidentifiable.
FOURIER_MIN_PERIOD = 2.0

# Daily remains the default so existing daily callers keep their contract.
DEFAULT_FOURIER_PERIODS = FOURIER_PERIODS_BY_FREQUENCY["daily"]


def pad_fourier_periods(periods, n_frequencies, min_period=FOURIER_MIN_PERIOD):
    """Grow an initial-period list to ``n_frequencies`` with log-spaced fillers.

    A frequency map entry can be shorter than the requested frequency count
    (monthly has three natural periods), and the learnable layer needs exactly
    one initial period per frequency pair.
    """
    periods = [float(p) for p in periods][: int(n_frequencies)]
    missing = int(n_frequencies) - len(periods)
    if missing <= 0:
        return periods
    longest = max(periods) if periods else DAYS_PER_YEAR
    filler = np.logspace(
        np.log10(min_period), np.log10(longest), missing + 2
    )[1:-1]
    return periods + [float(p) for p in filler]


def fourier_periods_for_frequency(frequency, n_frequencies=None):
    """Initial Fourier periods in units of one time step for a sampling frequency.

    Periods are unit-relative: 12 means "12 steps", so an annual cycle is 12 at
    monthly grain but ~365.25 at daily grain. Pass the panel's frequency rather
    than reusing the daily defaults.

    Alias: ``default_fourier_periods_for_frequency`` in ``frequency_presets``.
    """
    from .frequency_presets import normalize_frequency

    key = normalize_frequency(frequency)
    periods = FOURIER_PERIODS_BY_FREQUENCY[key]
    if n_frequencies is None:
        return list(periods)
    return list(periods[: int(n_frequencies)])


# Discoverability alias (same as frequency_presets.default_fourier_periods_for_frequency)
default_fourier_periods_for_frequency = fourier_periods_for_frequency

# ============================================================================
# GATHER LAYER (for serialization)
# ============================================================================



# InvertProbability restores a named 1-p head (bare 1-x becomes 'subtract' in Keras 3)
@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class InvertProbability(keras.layers.Layer):
    """Named 1 - p transform so Model.output_names keep 'non_zero_probability'."""

    def call(self, inputs):
        return 1.0 - inputs

    def get_config(self):
        return super().get_config()


# Small serializable layer to clip by value (avoids Lambda for model saving)
@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class ClipByValue(keras.layers.Layer):
    def __init__(self, clip_value_min, clip_value_max, **kwargs):
        super().__init__(**kwargs)
        self.clip_value_min = float(clip_value_min)
        self.clip_value_max = float(clip_value_max)

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.clip_value_min, self.clip_value_max)

    def get_config(self):
        config = super().get_config()
        config.update({
            "clip_value_min": self.clip_value_min,
            "clip_value_max": self.clip_value_max,
        })
        return config


# ============================================================================
# COMPONENT ATTENTION LAYERS (serializable, no Lambda)
# ============================================================================

@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class StackComponentsLayer(keras.layers.Layer):
    """Stack component outputs [batch, 4, 1] and squeeze to [batch, 4]"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # inputs is a list of 4 tensors, each [batch, 1]
        # Convert sparse to dense if needed
        dense_inputs = [tf.sparse.to_dense(x) if isinstance(x, tf.SparseTensor) else x for x in inputs]
        stacked = tf.stack(dense_inputs, axis=1)  # [batch, 4, 1]
        return tf.squeeze(stacked, axis=-1)  # [batch, 4]
    
    def compute_output_shape(self, input_shape):
        # input_shape is a list of 4 shapes, each (batch, 1)
        batch_size = input_shape[0][0] if isinstance(input_shape, list) else input_shape[0]
        return (batch_size, 4)
    
    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class ComponentEntropy(keras.layers.Layer):
    """Compute entropy of attention weights for regularization"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # inputs: attention weights [batch, 4]
        return -tf.reduce_sum(inputs * tf.math.log(inputs + 1e-8), axis=-1)
    
    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class ComponentEntropyLoss(keras.layers.Layer):
    """Convert entropy to loss with weight and add to model"""
    def __init__(self, entropy_weight=0.01, **kwargs):
        super().__init__(**kwargs)
        self.entropy_weight = float(entropy_weight)
    
    def call(self, inputs):
        # inputs: entropy values [batch]
        entropy_loss = self.entropy_weight * tf.reduce_mean(inputs)
        # Add loss to model
        self.add_loss(entropy_loss)
        return entropy_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({"entropy_weight": self.entropy_weight})
        return config


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class PrintAttentionWeights(keras.layers.Layer):
    """Print mean attention weights during training"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # inputs: attention weights [batch, 4]
        return inputs
    
    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class SumWeightedComponents(keras.layers.Layer):
    """Sum weighted components along axis -1 with keepdims"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # inputs: weighted components [batch, 4]
        return tf.reduce_sum(inputs, axis=-1, keepdims=True)  # [batch, 1]
    
    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class MultiplicativeComponentCombine(keras.layers.Layer):
    """Prophet-style multiplicative combine of softsign Level-2 experts.

    Stacked experts ``e = [T, S, H, R]`` (post softsign / SKU FiLM) and mixer
    weights ``α`` yield

    .. math::

        b_{\\mathrm{pre}}
        = \\mathrm{softplus}(e_T)
          \\prod_{k \\in \\{S,H,R\\}}
          \\max\\bigl(\\varepsilon,\\, 1 + \\alpha_k e_k\\bigr)

    Softsign experts live in ``(-1, 1)`` and ``α_k ∈ (0, 1]``, so
    ``α_k e_k ∈ (-1, 1)`` and each factor is positive after the ``ε`` floor.
    Inactive experts (via ``component_flags``) are skipped; if trend is off the
    product starts from ones. Mixer entropy / attention still run upstream —
    only the *combine* changes vs additive ``Σ α_k e_k``.
    """

    def __init__(self, component_flags=None, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        flags = list(component_flags) if component_flags is not None else [
            True, True, True, True
        ]
        if len(flags) != 4:
            raise ValueError(
                "component_flags must have length 4 [trend, seasonal, "
                f"holiday, regressor]; got {len(flags)}"
            )
        self.component_flags = [bool(f) for f in flags]
        self.eps = float(eps)

    def call(self, inputs):
        stacked, weights = inputs  # [batch, 4], [batch, 4]
        trend = stacked[:, 0:1]
        if self.component_flags[0]:
            base = tf.nn.softplus(trend)
        else:
            base = tf.ones_like(trend)

        out = base
        for index in (1, 2, 3):
            if not self.component_flags[index]:
                continue
            expert = stacked[:, index : index + 1]
            alpha = weights[:, index : index + 1]
            out = out * tf.maximum(
                tf.cast(self.eps, stacked.dtype),
                1.0 + alpha * expert,
            )
        return out

    def compute_output_shape(self, input_shape):
        stacked_shape = input_shape[0]
        return (stacked_shape[0], 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "component_flags": list(self.component_flags),
            "eps": self.eps,
        })
        return config


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class HorizonGateFromBaseProbability(keras.layers.Layer):
    """Per-horizon occurrence gate anchored on the learned base probability.

    Takes [offsets [batch, H], base_probability [batch, 1]] and returns
    sigmoid(offsets + logit(base_probability)), so the multi-horizon head
    refines the intermittent handler's estimate instead of replacing it.
    """

    PROBABILITY_EPSILON = 1e-6

    def call(self, inputs):
        offsets, base_probability = inputs
        p = tf.clip_by_value(
            base_probability,
            self.PROBABILITY_EPSILON,
            1.0 - self.PROBABILITY_EPSILON,
        )
        base_logit = tf.math.log(p) - tf.math.log(1.0 - p)
        return tf.nn.sigmoid(offsets + base_logit)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class OrthogonalityPenalty(keras.layers.Layer):
    """Add an off-diagonal covariance penalty for component outputs."""

    def __init__(self, weight=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)

    def call(self, inputs):
        components = tf.concat(inputs, axis=-1)
        centered = components - tf.reduce_mean(
            components, axis=0, keepdims=True
        )
        batch_size = tf.cast(tf.shape(centered)[0], centered.dtype)
        gram = tf.matmul(centered, centered, transpose_a=True)
        gram = gram / (batch_size + tf.cast(1e-6, centered.dtype))
        off_diagonal = gram - tf.linalg.diag(tf.linalg.diag_part(gram))
        self.add_loss(self.weight * tf.reduce_mean(tf.square(off_diagonal)))
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"weight": self.weight})
        return config


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
def mish(x):
    """
    Mish activation: x * tanh(softplus(x))
    """
    return x * tf.math.tanh(tf.math.softplus(x))


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
def sparse_amplify(x):
    """
    Sparse amplify: x * 1/(abs(x)+1)
    
    Designed for sparse intermittent demand (90% zeros).
    Dampens large values, maintains small signals.
    """
    return x * (1.0 / (tf.abs(x) + 1.0))


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
def sparse_amplify_exp(x):
    """
    Sparse amplify with exponential: x * exp(1/(abs(x)+1))
    
    More aggressive sparse signal amplification:
    - x ≈ 0: amplifies by ~2.7x (exp(1) ≈ 2.718)
    - x large: no amplification (exp(0) = 1)
    
    Use for extremely sparse data requiring signal boost.
    """
    return x * tf.exp(1.0 / (tf.abs(x) + 1.0))


def _resolve_output_activation(output_activation):
    """Map expert ``output_activation`` name to a Dense/callable activation."""
    if output_activation == 'sparse_amplify_exp':
        return sparse_amplify_exp
    if output_activation == 'sparse_amplify':
        return sparse_amplify
    return output_activation


def _apply_output_activation(x, output_activation):
    """Apply expert output activation (used on mono scalar paths)."""
    fn = _resolve_output_activation(output_activation)
    if fn is None or fn == 'linear':
        return x
    if callable(fn):
        return fn(x)
    return tf.keras.activations.get(fn)(x)


# ============================================================================
# STOP GRADIENT LAYER (for decoupled loss training)
# ============================================================================

@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE
)
class StopGradient(tf.keras.layers.Layer):
    """
    Layer that stops gradients from flowing backward.
    
    Use case: Prevent final_forecast loss from affecting zero_probability
    through multiplication, while still allowing forward pass.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.stop_gradient(inputs)
    
    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE
)
class ScheduledStopGradient(tf.keras.layers.Layer):
    """
    Blends passthrough and stop_gradient based on a non-trainable schedule.
    output = (1 - p) * x + p * stop_gradient(x), with p ∈ [0,1].
    Update `stop_prob` via callbacks to ramp stopping over epochs.
    """
    def __init__(self, initial_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_prob = float(initial_prob)
        self.stop_prob = tf.Variable(self.initial_prob, trainable=False, dtype=tf.float32, name=f'{self.name}_prob')

    def call(self, inputs):
        p = tf.clip_by_value(self.stop_prob, 0.0, 1.0)
        return (1.0 - p) * inputs + p * tf.stop_gradient(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'initial_prob': self.initial_prob})
        return config


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE
)
class TemperatureScale(tf.keras.layers.Layer):
    """
    Applies temperature scaling for sharper sigmoid decisions.
    Transforms: (x - 0.5) / temperature
    """
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    
    def call(self, inputs):
        return (inputs - 0.5) / self.temperature
    
    def get_config(self):
        config = super().get_config()
        config.update({'temperature': self.temperature})
        return config


# ============================================================================
# CUSTOM LAYERS
# ============================================================================

@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class SqueezeLayer(tf.keras.layers.Layer):
    """Serializable layer to squeeze a specific axis."""
    
    def __init__(self, axis=1, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        # Convert to dense tensor if needed, then squeeze
        inputs = tf.convert_to_tensor(inputs)
        return tf.squeeze(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(SqueezeLayer, self).get_config()
        config.update({'axis': self.axis})
        return config


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class ExpandDimsLayer(tf.keras.layers.Layer):
    """Serializable layer to expand dimensions."""
    
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({'axis': self.axis})
        return config


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class ReduceSumLayer(tf.keras.layers.Layer):
    """Serializable layer to reduce sum along an axis."""
    
    def __init__(self, axis=-1, keepdims=True, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
    
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis, keepdims=self.keepdims)
    
    def get_config(self):
        config = super(ReduceSumLayer, self).get_config()
        config.update({
            'axis': self.axis,
            'keepdims': self.keepdims
        })
        return config


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class OneMinusLayer(tf.keras.layers.Layer):
    """Serializable layer to compute 1 - x."""
    
    def call(self, inputs):
        return 1.0 - inputs
    
    def get_config(self):
        return super(OneMinusLayer, self).get_config()


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class GatherLayer(tf.keras.layers.Layer):
    """Serializable layer to gather specific indices from input."""
    
    def __init__(self, indices, **kwargs):
        super(GatherLayer, self).__init__(**kwargs)
        self.indices = indices if isinstance(indices, list) else list(indices)
    
    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=-1)
    
    def get_config(self):
        config = super(GatherLayer, self).get_config()
        config.update({'indices': self.indices})
        return config


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class TemperatureSoftmax(tf.keras.layers.Layer):
    """Serializable temperature-scaled softmax for component attention.

    ``active_mask`` removes disabled component slots from the mixture. Without
    it, softmax spends probability mass on components whose output is forced to
    zero, and the entropy penalty rewards keeping it there because uniform over
    all slots maximizes entropy.
    """

    LOGIT_MASK_PENALTY = 1e9

    def __init__(self, temperature=0.7, active_mask=None, **kwargs):
        super(TemperatureSoftmax, self).__init__(**kwargs)
        self.temperature = float(temperature)
        self.active_mask = (
            None if active_mask is None else [float(m) for m in active_mask]
        )
    
    def call(self, inputs):
        logits = inputs / self.temperature
        if self.active_mask is not None and not all(self.active_mask):
            mask = tf.constant(self.active_mask, dtype=logits.dtype)
            logits = logits + (mask - 1.0) * self.LOGIT_MASK_PENALTY
        return tf.nn.softmax(logits, axis=-1)
    
    def get_config(self):
        config = super(TemperatureSoftmax, self).get_config()
        config.update(
            {'temperature': self.temperature, 'active_mask': self.active_mask}
        )
        return config


@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class ExtractComponentWeight(tf.keras.layers.Layer):
    """Serializable layer to extract a specific component weight."""
    
    def __init__(self, component_index, **kwargs):
        super(ExtractComponentWeight, self).__init__(**kwargs)
        self.component_index = component_index
    
    def call(self, inputs):
        return tf.expand_dims(inputs[:, self.component_index], axis=-1)
    
    def get_config(self):
        config = super(ExtractComponentWeight, self).get_config()
        config.update({'component_index': self.component_index})
        return config


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="MaskedEntropyAttention"
)
class MaskedEntropyAttention(tf.keras.layers.Layer):
    """Feature attention with a fixed-strength entropy penalty.

    ``entropy_weight`` call sites were tuned under the historical init
    ``softplus(0.01) ≈ 0.698``. Keep that factor as a *fixed* scale: a
    trainable multiplier on this penalty collapses toward zero and silently
    disables it; applying the weight at full strength (scale=1) over-sparsifies
    feature attention and pushes the intermittent gate toward high recall /
    high bias on daily panels.
    """

    # softplus(0.01) at the historical Constant(0.01) init — frozen, not trained.
    DEFAULT_ENTROPY_SCALE = float(np.log1p(np.exp(0.01)))  # ≈ 0.69815

    def __init__(
        self,
        units,
        entropy_weight=0.01,
        dropout_rate=0.1,
        temperature=0.7,
        attention_scale=2.0,
        present=1.0,
        entropy_scale=None,
        equal_weights=False,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.entropy_weight = float(entropy_weight)
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.attention_scale = attention_scale
        self.present_value = present
        self.equal_weights = bool(equal_weights)
        self.entropy_scale = (
            self.DEFAULT_ENTROPY_SCALE
            if entropy_scale is None
            else float(entropy_scale)
        )

    def build(self, input_shape):
        n_features = input_shape[-1]
        # Create tf.constant present at build time for correct runtime gating
        self.present = tf.constant(self.present_value, dtype=tf.float32)
        self.attention_dense = Dense(n_features, activation=mish, use_bias=False)
        self.projection = Dense(self.units, use_bias=False)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        x = self.layer_norm(inputs)
        if self.equal_weights:
            # Ablation: uniform 1/n over channels (no learned selection).
            n = tf.cast(tf.shape(inputs)[-1], tf.float32)
            weights = tf.ones_like(inputs) / tf.maximum(n, 1.0)
        else:
            scores = self.attention_dense(x)

            logits = self.attention_scale * tf.tanh(scores)
            # Temperature floor to avoid overly peaked distributions
            temp = tf.maximum(
                tf.constant(0.3, dtype=tf.float32),
                tf.constant(self.temperature, dtype=tf.float32),
            )
            logits = logits / temp

            weights = tf.nn.softmax(logits, axis=-1)
        attended = inputs * weights

        output = self.projection(attended)
        output = self.dropout(output, training=training)

        if not self.equal_weights:
            entropy = -tf.reduce_sum(
                weights * tf.math.log(weights + 1e-8), axis=-1
            )
            present_scalar = tf.cast(self.present_value, tf.float32)
            entropy_loss = (
                present_scalar
                * self.entropy_weight
                * self.entropy_scale
                * tf.reduce_mean(entropy)
            )
            self.add_loss(entropy_loss)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "equal_weights": self.equal_weights,
            "entropy_weight": self.entropy_weight,
            "dropout_rate": self.dropout_rate,
            "temperature": self.temperature,
            "attention_scale": self.attention_scale,
            "present": self.present_value,
            "entropy_scale": self.entropy_scale,
        })
        return config


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="LearnableFourierFeatures"
)
class LearnableFourierFeatures(tf.keras.layers.Layer):
    """
    Learnable Fourier features with adaptive frequencies.
    
    Instead of fixed periods, learns optimal frequencies from data using
    gradient descent. Periods are in time steps of the series, so 12 means a
    yearly cycle on monthly data but a fortnightly one on daily data.
    
    For each frequency k:
        output = [sin(ω_k * t), cos(ω_k * t)]
    where ω_k is learnable.
    """
    
    def __init__(
        self,
        n_frequencies=5,
        initial_periods=None,
        min_period=FOURIER_MIN_PERIOD,
        max_period=None,
        name='learnable_fourier',
        **kwargs
    ):
        """
        Args:
            n_frequencies: Number of frequency pairs (each outputs sin + cos)
            initial_periods: Initial periods in *time steps* of the series
                (e.g. [7, 30.4375, 365.25] daily, [3, 6, 12] monthly).
                If None, uses log-spaced periods.
            min_period: Shortest learnable period in time steps (Nyquist: 2).
            max_period: Longest learnable period in time steps. When None it is
                derived from the initial periods, so an annual cycle is never
                clipped by a hard-coded daily bound.
        """
        super(LearnableFourierFeatures, self).__init__(
            name=name, **kwargs
        )
        self.n_frequencies = n_frequencies
        self.initial_periods = initial_periods
        self.min_period = min_period
        self.max_period = max_period
        
    def build(self, input_shape):
        # Initialize frequencies based on periods
        if self.initial_periods is not None:
            initial_periods = np.array(
                self.initial_periods[:self.n_frequencies],
                dtype=np.float32
            )
        else:
            # Log-spaced periods from min to one year of steps
            span_max = (
                float(self.max_period)
                if self.max_period is not None
                else DAYS_PER_YEAR
            )
            initial_periods = np.logspace(
                np.log10(self.min_period),
                np.log10(span_max),
                self.n_frequencies,
                dtype=np.float32
            )
        # Resolve the clip bound from the actual initial periods so the longest
        # cycle can drift instead of being pinned at its starting value.
        self._max_period = self.resolved_max_period(initial_periods)
        
        # Convert periods to frequencies: ω = 2π / period
        initial_frequencies = 2 * np.pi / initial_periods
        
        # Store log(ω) for unconstrained optimization
        # Then exponentiate to ensure positive frequencies
        self.log_frequencies = self.add_weight(
            name='log_frequencies',
            shape=(self.n_frequencies,),
            initializer=tf.keras.initializers.Constant(
                np.log(initial_frequencies)
            ),
            trainable=True,
            dtype=tf.float32
        )
        
        super(LearnableFourierFeatures, self).build(input_shape)

    def resolved_max_period(self, initial_periods=None):
        """Longest learnable period, derived from initial periods when unset."""
        if self.max_period is not None:
            return float(self.max_period)
        if initial_periods is None or len(initial_periods) == 0:
            return DAYS_PER_YEAR * FOURIER_MAX_PERIOD_SLACK
        return float(np.max(initial_periods)) * FOURIER_MAX_PERIOD_SLACK

    def call(self, inputs):
        """
        Args:
            inputs: [batch, 1] elapsed time in the same unit as the periods
                (days since epoch for daily, month index for monthly, ...)
        
        Returns:
            fourier_features: [batch, 2*n_frequencies]
                [sin(ω₁t), cos(ω₁t), sin(ω₂t), cos(ω₂t), ...]
        """
        # Ensure positive frequencies
        frequencies = tf.exp(self.log_frequencies)  # [n_frequencies]
        
        # Constrain to [min_period, max_period]
        min_freq = 2 * np.pi / self._max_period
        max_freq = 2 * np.pi / self.min_period
        frequencies = tf.clip_by_value(frequencies, min_freq, max_freq)
        
        # Reshape for broadcasting: [1, n_frequencies]
        frequencies = tf.reshape(frequencies, (1, -1))
        
        # Compute ω * t: [batch, 1] * [1, n_frequencies] = [batch, n_freq]
        angles = inputs * frequencies
        
        # Compute sin and cos
        sin_features = tf.sin(angles)  # [batch, n_frequencies]
        cos_features = tf.cos(angles)  # [batch, n_frequencies]
        
        # Interleave: [sin₁, cos₁, sin₂, cos₂, ...] with a static last dim
        # (dynamic tf.shape(...)/-1 reshape breaks Keras Dense units inference)
        features = tf.stack([sin_features, cos_features], axis=-1)
        features = tf.reshape(
            features, (-1, 2 * self.n_frequencies)
        )
        
        return features
    
    def get_config(self):
        config = super(LearnableFourierFeatures, self).get_config()
        config.update({
            'n_frequencies': self.n_frequencies,
            'initial_periods': self.initial_periods,
            'min_period': self.min_period,
            'max_period': self.max_period
        })
        return config
    
    def get_learned_periods(self):
        """Helper to inspect learned periods after training."""
        frequencies = tf.exp(self.log_frequencies).numpy()
        periods = 2 * np.pi / frequencies
        return periods


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="ChangepointReLU"
)
class ChangepointReLU(tf.keras.layers.Layer):
    """Vectorised ReLU hinges with ordered learnable changepoint locations.

    Locations are constrained via softplus+cumsum so ``cp[0] < … < cp[K-1]``.
    When paired with softplus-constrained hinge slopes (see
    ``TrendComponentLightweight(trend_monotonic=True)``), the resulting
    piecewise-linear function of the time feature is monotone.
    """

    def __init__(
        self,
        n_changepoints=10,
        time_min=0.0,
        time_max=1.0,
        name='changepoint_relu',
        **kwargs
    ):
        super(ChangepointReLU, self).__init__(name=name, **kwargs)
        self.n_changepoints = n_changepoints
        self.time_min = time_min
        self.time_max = time_max
        
    def build(self, input_shape):
        # Learn deltas between changepoints instead of absolute positions.
        # softplus+cumsum keeps locations ordered: cp[i] = sum(deltas[0:i]).
        initial_deltas = np.full(
            self.n_changepoints,
            (self.time_max - self.time_min) / self.n_changepoints,
            dtype=np.float32
        )
        
        self.changepoint_deltas = self.add_weight(
            name='changepoint_deltas',
            shape=(self.n_changepoints,),
            initializer=tf.keras.initializers.Constant(initial_deltas),
            trainable=True,
            dtype=tf.float32
        )
        super(ChangepointReLU, self).build(input_shape)

    def call(self, inputs):
        # inputs: [batch, 1] time feature
        
        # 1. Ensure deltas are positive using softplus
        positive_deltas = tf.nn.softplus(self.changepoint_deltas)
        
        # 2. Compute cumulative sum to get ordered changepoints
        changepoints = tf.cumsum(positive_deltas)
        
        # 3. Scale to [time_min, time_max] range
        time_range = self.time_max - self.time_min
        changepoints = (changepoints / changepoints[-1]) * time_range
        changepoints = changepoints + self.time_min
        
        # 4. Reshape and apply ReLU hinges: max(0, t - cp_i)
        cp = tf.reshape(changepoints, (1, -1))  # [1, n_changepoints]
        relu_features = tf.nn.relu(inputs - cp)  # [batch, n_changepoints]
        return relu_features

    def get_config(self):
        config = super(ChangepointReLU, self).get_config()
        config.update({
            'n_changepoints': self.n_changepoints,
            'time_min': self.time_min,
            'time_max': self.time_max
        })
        return config


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="TrendComponentLightweight"
)
class TrendComponentLightweight(tf.keras.layers.Layer):
    """
    Trend component matching hierarchical TabNet:
    Single time feature → ChangepointReLU → (monotone PWL | attention path) → Output

    When ``trend_monotonic=True`` (default), hinge slopes are softplus-constrained
    in magnitude with a learned sign (``softplus(raw) * tanh(raw_sign)``) so the
    trend contribution is monotone in the time feature without a direction
    hyperparameter. Changepoint locations remain ordered via softplus+cumsum in
    ``ChangepointReLU``.
    When ``trend_monotonic=False``, uses the legacy attention + Dense path
    (unconstrained, may be non-monotone).
    """
    
    def __init__(
        self,
        n_changepoints=10,
        hidden_dim=32,
        dropout_rate=0.1,
        time_min=0.0,
        time_max=1.0,
        use_sku_shift_scale=True,
        attention_temperature=0.7,
        attention_entropy_weight=0.01,
        output_activation='softsign',
        present=1.0,
        trend_monotonic=True,
        name='trend_lightweight',
        **kwargs
    ):
        # Ignore legacy direction kwargs from older saved configs.
        kwargs.pop('monotonic_direction', None)
        super(TrendComponentLightweight, self).__init__(
            name=name, **kwargs
        )
        self.n_changepoints = n_changepoints
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.time_min = time_min
        self.time_max = time_max
        self.use_sku_shift_scale = use_sku_shift_scale
        self.attention_temperature = attention_temperature
        self.attention_entropy_weight = attention_entropy_weight
        self.output_activation = output_activation
        self.present_value = present
        self.present = tf.constant(present, dtype=tf.float32)  # 1.0 if enabled, 0.0 if disabled
        self.trend_monotonic = bool(trend_monotonic)
        
    def build(self, input_shape):
        # Learnable ordered changepoint locations on time feature
        self.changepoint_relu = ChangepointReLU(
            n_changepoints=self.n_changepoints,
            time_min=self.time_min,
            time_max=self.time_max,
            name=f'{self.name}_changepoints'
        )

        if self.trend_monotonic:
            # Softplus magnitude × learned sign: g(t) = b + Σ m_i·ReLU(t−cp_i)
            # with m_i = softplus(s_i)·tanh(raw_sign). Direction is not given.
            self.raw_slopes = self.add_weight(
                name='raw_slopes',
                shape=(self.n_changepoints,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
            self.raw_sign = self.add_weight(
                name='raw_sign',
                shape=(1,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
            self.trend_bias = self.add_weight(
                name='trend_bias',
                shape=(1,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
            sku_units = 1
        else:
            # Normalize changepoint features for stable training
            self.changepoint_norm = LayerNormalization(
                name=f'{self.name}_cp_norm'
            )

            # Attention mechanism: learnable weights for each changepoint
            # This learns which changepoints are important
            self.attention_layer = Dense(
                1,  # Single score per changepoint
                activation=None,
                use_bias=False,
                name=f'{self.name}_attention'
            )

            self.dropout_layer = Dropout(self.dropout_rate)

            # Dense transform on attended changepoints
            self.hidden_layer = Dense(
                self.hidden_dim,
                activation=mish,
                use_bias=False,
                name=f'{self.name}_hidden'
            )
            sku_units = self.hidden_dim

            # Final projection — no bias (intermittent/forecast path convention).
            # Activation is configurable: softsign (default), linear, sparse_*, …
            self.output_layer = Dense(
                1,
                activation=_resolve_output_activation(self.output_activation),
                use_bias=False,
                kernel_constraint=UnitNorm(axis=0),
                name=f'{self.name}_output'
            )

        # SKU-specific shift and scale (positive softplus scale preserves
        # monotonicity in time for a fixed SKU when trend_monotonic=True).
        if self.use_sku_shift_scale:
            self.sku_beta = Dense(
                sku_units,
                activation='softplus',
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(1e-5),
                name=f'{self.name}_sku_beta'
            )
            self.sku_alpha = Dense(
                sku_units,
                activation='linear',
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(1e-5),
                name=f'{self.name}_sku_alpha'
            )
            self.sku_multiply = Multiply(name=f'{self.name}_sku_multiply')
            self.sku_add = Add(name=f'{self.name}_sku_add')
        
        super(TrendComponentLightweight, self).build(input_shape)

    def _monotone_slopes(self):
        return tf.nn.softplus(self.raw_slopes) * tf.tanh(self.raw_sign)
    
    def call(self, inputs, sku_embedding=None, training=None):
        """
        Args:
            inputs: [batch, 1] single time feature (date_numeric)
            sku_embedding: [batch, sku_dim] SKU embedding for shift-scale
        
        Returns:
            trend_forecast: [batch, 1]
        """
        # Apply learnable changepoints: [batch, 1] -> [batch, n_changepoints]
        cp_features = self.changepoint_relu(inputs)

        if self.trend_monotonic:
            # Monotone PWL: no LayerNorm/attention (those can break mono in t).
            slopes = self._monotone_slopes()
            output = tf.reduce_sum(
                cp_features * slopes, axis=-1, keepdims=True
            ) + self.trend_bias
            if self.use_sku_shift_scale and sku_embedding is not None:
                beta = self.sku_beta(sku_embedding)
                alpha = self.sku_alpha(sku_embedding)
                output = self.sku_multiply([output, beta])
                output = self.sku_add([output, alpha])
            # softsign (default) bounds signed expert impact; still monotone.
            return _apply_output_activation(output, self.output_activation)

        # Legacy unconstrained path (may be non-monotone in time).
        # Normalize changepoint features
        cp_features = self.changepoint_norm(cp_features)
        
        # Reshape to [batch, n_changepoints, 1] for attention computation
        cp_reshaped = tf.expand_dims(cp_features, axis=-1)
        
        # Compute attention score for each changepoint
        # [batch, n_changepoints, 1] -> [batch, n_changepoints, 1]
        attention_logits = self.attention_layer(cp_reshaped)
        
        # Squeeze and apply softmax with temperature
        attention_logits = tf.squeeze(attention_logits, axis=-1)
        attention_weights = tf.nn.softmax(
            attention_logits / self.attention_temperature, axis=-1
        )
        
        # Entropy regularization: encourage sparse attention
        # Lower entropy = more focused on few changepoints
        # MASKED: only apply entropy loss where self.present = 1.0
        if self.attention_entropy_weight > 0:
            entropy = -tf.reduce_sum(
                attention_weights * tf.math.log(attention_weights + 1e-8),
                axis=-1
            )
            entropy_loss = self.present * self.attention_entropy_weight * tf.reduce_mean(
                entropy
            )
            self.add_loss(entropy_loss)
        
        # Weighted sum of changepoint features: [batch, n_changepoints]
        attended = cp_features * attention_weights
        attended = tf.reduce_sum(attended, axis=-1, keepdims=True)
        attended = self.dropout_layer(attended, training=training)
        
        # Dense transform to hidden representation
        trend_hidden = self.hidden_layer(attended)
        
        # Apply SKU-specific shift and scale
        if self.use_sku_shift_scale and sku_embedding is not None:
            beta = self.sku_beta(sku_embedding)
            alpha = self.sku_alpha(sku_embedding)
            trend_hidden = self.sku_multiply([trend_hidden, beta])
            trend_hidden = self.sku_add([trend_hidden, alpha])
        
        # Final projection
        output = self.output_layer(trend_hidden)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_changepoints': self.n_changepoints,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'time_min': self.time_min,
            'time_max': self.time_max,
            'use_sku_shift_scale': self.use_sku_shift_scale,
            'attention_temperature': self.attention_temperature,
            'attention_entropy_weight': self.attention_entropy_weight,
            'output_activation': self.output_activation,
            'present': self.present_value,
            'trend_monotonic': self.trend_monotonic,
        })
        return config


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="SeasonalComponentLightweight"
)
class SeasonalComponentLightweight(tf.keras.layers.Layer):
    """
    Seasonal component using masked attention for Fourier features.
    
    Lighter alternative to TabNet: learns which seasonal frequencies matter.
    
    Can use either:
    - Fixed Fourier features (pre-computed)
    - Learnable Fourier features (adaptive frequencies)
    """  
    def __init__(
        self,
        hidden_dim=32,
        dropout_rate=0.1,
        use_sku_shift_scale=True,
        use_learnable_fourier=False,
        n_learnable_frequencies=5,
        fourier_periods=None,
        fourier_min_period=FOURIER_MIN_PERIOD,
        fourier_max_period=None,
        activation='mish',
        output_activation='softsign',
        present=1.0,
        name='seasonal_lightweight',
        **kwargs
    ):
        super(SeasonalComponentLightweight, self).__init__(
            name=name, **kwargs
        )
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_sku_shift_scale = use_sku_shift_scale
        self.use_learnable_fourier = use_learnable_fourier
        self.n_learnable_frequencies = n_learnable_frequencies
        self.fourier_periods = fourier_periods
        self.fourier_min_period = fourier_min_period
        self.fourier_max_period = fourier_max_period
        self.activation = activation
        self.output_activation = output_activation
        self.present_value = present
        self.present = tf.constant(present, dtype=tf.float32)  # 1.0 if enabled, 0.0 if disabled
        
    def build(self, input_shape):
        # Optional: Generate Fourier features from time input
        if self.use_learnable_fourier:
            # Expects input_shape = [batch, 1] (time only)
            if self.fourier_periods is None:
                # Daily grain: week, fortnight, mean month/quarter/year. Pass
                # fourier_periods explicitly for any other sampling frequency.
                self.fourier_periods = list(DEFAULT_FOURIER_PERIODS)
            
            # The learnable layer needs exactly one initial period per frequency
            if len(self.fourier_periods) < self.n_learnable_frequencies:
                logger.warning(
                    "Only %d Fourier periods provided for %d frequencies; "
                    "remaining frequencies will use log spacing.",
                    len(self.fourier_periods),
                    self.n_learnable_frequencies,
                )
            initial_periods = pad_fourier_periods(
                self.fourier_periods,
                self.n_learnable_frequencies,
                min_period=self.fourier_min_period,
            )
            
            self.fourier_layer = LearnableFourierFeatures(
                n_frequencies=self.n_learnable_frequencies,
                initial_periods=initial_periods,
                min_period=self.fourier_min_period,
                max_period=self.fourier_max_period,
                name=f'{self.name}_fourier'
            )
            n_features = 2 * self.n_learnable_frequencies
        else:
            # Expects pre-computed Fourier features
            n_features = input_shape[-1]
        
        # Masked attention for Fourier feature selection
        # Higher entropy (0.05): select few key frequencies
        self.attention = MaskedEntropyAttention(
            units=self.hidden_dim,
            temperature=0.5,
            entropy_weight=0.05,
            dropout_rate=self.dropout_rate,
            present=self.present_value,
            name=f'{self.name}_attention'
        )
        
        # Seasonal pattern learning
        self.seasonal_layer = Dense(
            self.hidden_dim // 2,
            activation=mish,
            use_bias=False,
            name=f'{self.name}_pattern'
        )
        
        # SKU-specific shift and scale
        if self.use_sku_shift_scale:
            self.sku_beta = Dense(
                self.hidden_dim // 2,
                activation='softplus',
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(1e-5),
                name=f'{self.name}_sku_beta'
            )
            self.sku_alpha = Dense(
                self.hidden_dim // 2,
                activation='linear',
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(1e-5),
                name=f'{self.name}_sku_alpha'
            )
            self.sku_multiply = Multiply(name=f'{self.name}_sku_multiply')
            self.sku_add = Add(name=f'{self.name}_sku_add')
        
        # Output projection (same activation contract as the other components)
        self.output_layer = Dense(
            1,
            activation=_resolve_output_activation(self.output_activation),
            use_bias=False,
            kernel_constraint=UnitNorm(axis=0),
            name=f'{self.name}_output'
        )
        
        super(SeasonalComponentLightweight, self).build(input_shape)
    
    def call(self, inputs, sku_embedding=None, training=None):
        """
        Args:
            inputs: [batch, n_fourier_features] or [batch, 1] if learnable
            sku_embedding: [batch, sku_dim] SKU embedding for shift-scale
        
        Returns:
            seasonal_forecast: [batch, 1]
        """
        # Generate Fourier features if using learnable frequencies
        if self.use_learnable_fourier:
            fourier_features = self.fourier_layer(inputs)
        else:
            fourier_features = inputs
        
        # Apply masked attention to select important frequencies
        attended_features = self.attention(
            fourier_features, training=training
        )
        
        # Learn seasonal pattern
        seasonal = self.seasonal_layer(attended_features)
        
        # Apply SKU-specific shift and scale
        if self.use_sku_shift_scale and sku_embedding is not None:
            beta = self.sku_beta(sku_embedding)
            alpha = self.sku_alpha(sku_embedding)
            seasonal = self.sku_multiply([seasonal, beta])
            seasonal = self.sku_add([seasonal, alpha])
        
        # Project to output
        output = self.output_layer(seasonal)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'use_learnable_fourier': self.use_learnable_fourier,
            'n_learnable_frequencies': self.n_learnable_frequencies,
            'fourier_periods': self.fourier_periods,
            'fourier_min_period': self.fourier_min_period,
            'fourier_max_period': self.fourier_max_period,
            'use_sku_shift_scale': self.use_sku_shift_scale,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'present': self.present_value,
        })
        return config


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="HolidayComponentLightweight"
)
class HolidayComponentLightweight(tf.keras.layers.Layer):
    """
    Holiday component: per-holiday distance → ChangepointReLU → scalar expert.

    When ``holiday_monotonic=True`` (default), each holiday uses softplus-
    constrained hinge slopes on **absolute** distance ``|days_from_*|``
    (domain [0, 365] for ChangepointReLU), with a learned per-holiday sign
    (``softplus(raw) * tanh(raw_sign)``). Direction is not a hyperparameter.
    Stacked mono scalars then pass through **selection attention** over the
    holiday axis (TemperatureSoftmax + weighted sum). SKU FiLM after the
    attended scalar (softplus scale) preserves mono in |d| for a fixed SKU
    when attention weights are fixed (e.g. single holiday).

    When ``level1_selection_attention=False`` (with mono on), holiday channels
    are combined with uniform ``1/n`` weights (no learned intra-expert
    selection) — ablation for Level-1 hierarchical attention.

    When ``holiday_monotonic=False``, uses the legacy per-holiday attention +
    aggregate Dense path (unconstrained, may be non-monotone in distance).
    """
    
    def __init__(
        self,
        n_changepoints=5,
        hidden_dim=32,
        dropout_rate=0.1,
        use_sku_shift_scale=True,
        attention_temperature=0.7,
        attention_entropy_weight=0.01,
        output_activation='softsign',
        present=1.0,
        holiday_monotonic=True,
        level1_selection_attention=True,
        name='holiday_lightweight',
        **kwargs
    ):
        # Ignore legacy direction kwargs from older saved configs.
        kwargs.pop('holiday_monotonic_direction', None)
        super(HolidayComponentLightweight, self).__init__(
            name=name, **kwargs
        )
        self.n_changepoints = n_changepoints
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_sku_shift_scale = use_sku_shift_scale
        self.attention_temperature = attention_temperature
        self.attention_entropy_weight = attention_entropy_weight
        self.output_activation = output_activation
        self.present_value = present
        self.present = tf.constant(present, dtype=tf.float32)  # 1.0 if enabled, 0.0 if disabled
        self.holiday_monotonic = bool(holiday_monotonic)
        self.level1_selection_attention = bool(level1_selection_attention)
        
    def build(self, input_shape):
        n_holidays = input_shape[-1]
        self.n_holidays = n_holidays
        
        # Per-holiday changepoint layers (hinges on |days_from| ∈ [0, 365])
        self.changepoint_layers = []
        for i in range(n_holidays):
            cp_layer = ChangepointReLU(
                n_changepoints=self.n_changepoints,
                time_min=0.0,
                time_max=365.0,  # |days_from_*| in days
                name=f'{self.name}_cp_{i}'
            )
            self.changepoint_layers.append(cp_layer)

        if self.holiday_monotonic:
            # Softplus magnitude × learned per-holiday sign on |d|.
            self.raw_slopes = self.add_weight(
                name='raw_slopes',
                shape=(n_holidays, self.n_changepoints),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
            self.raw_sign = self.add_weight(
                name='raw_sign',
                shape=(n_holidays,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
            self.holiday_bias = self.add_weight(
                name='holiday_bias',
                shape=(n_holidays,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
            # Selection attention over mono holiday scalars (not XOR with mono).
            # Skipped when level1_selection_attention=False (uniform 1/n).
            if self.level1_selection_attention:
                self.selection_logits = Dense(
                    n_holidays,
                    activation=None,
                    use_bias=False,
                    name=f'{self.name}_selection_logits',
                )
                self.selection_softmax = TemperatureSoftmax(
                    temperature=self.attention_temperature,
                    name=f'{self.name}_selection_softmax',
                )
            sku_units = 1
        else:
            self.changepoint_norms = []
            self.per_holiday_hidden = []
            self.per_holiday_attention = []

            for i in range(n_holidays):
                cp_norm = LayerNormalization(
                    name=f'{self.name}_cp_norm_{i}'
                )
                self.changepoint_norms.append(cp_norm)

                attn_layer = Dense(
                    1,
                    activation=None,
                    use_bias=False,
                    name=f'{self.name}_attn_{i}'
                )
                self.per_holiday_attention.append(attn_layer)

                hidden_layer = Dense(
                    self.hidden_dim // n_holidays,
                    activation=mish,
                    use_bias=False,
                    name=f'{self.name}_hidden_{i}'
                )
                self.per_holiday_hidden.append(hidden_layer)

            self.aggregate_hidden = Dense(
                self.hidden_dim,
                activation=mish,
                use_bias=False,
                name=f'{self.name}_aggregate_hidden'
            )

            self.aggregate_attention = Dense(
                self.hidden_dim,
                activation=None,
                use_bias=False,
                name=f'{self.name}_aggregate_attention'
            )

            self.dropout_layer = Dropout(self.dropout_rate)
            sku_units = self.hidden_dim

            self.output_layer = Dense(
                1,
                activation=_resolve_output_activation(self.output_activation),
                use_bias=False,
                kernel_constraint=UnitNorm(axis=0),
                name=f'{self.name}_output'
            )

        # SKU FiLM after scalar (mono) or hidden (legacy)
        if self.use_sku_shift_scale:
            self.sku_beta = Dense(
                sku_units,
                activation='softplus',
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(1e-5),
                name=f'{self.name}_sku_beta'
            )
            self.sku_alpha = Dense(
                sku_units,
                activation='linear',
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(1e-5),
                name=f'{self.name}_sku_alpha'
            )
            self.sku_multiply = Multiply(name=f'{self.name}_sku_multiply')
            self.sku_add = Add(name=f'{self.name}_sku_add')
        
        super(HolidayComponentLightweight, self).build(input_shape)

    def _monotone_slopes(self, holiday_index):
        return (
            tf.nn.softplus(self.raw_slopes[holiday_index])
            * tf.tanh(self.raw_sign[holiday_index])
        )

    def _mono_channel_scalars(self, inputs):
        """Per-holiday softplus-PWL scalars; each is mono in its ``|d|``."""
        parts = []
        for i in range(self.n_holidays):
            d_abs = tf.abs(inputs[:, i:i + 1])
            cp_features = self.changepoint_layers[i](d_abs)
            slopes = self._monotone_slopes(i)
            part = tf.reduce_sum(
                cp_features * slopes, axis=-1, keepdims=True
            ) + self.holiday_bias[i]
            parts.append(part)
        return tf.concat(parts, axis=-1)

    def call(self, inputs, sku_embedding=None, training=None):
        """
        Args:
            inputs: [batch, n_holiday_distances] signed ``days_from_*``
            sku_embedding: [batch, sku_dim] SKU embedding
        
        Returns:
            holiday_forecast: [batch, 1]
        """
        if self.holiday_monotonic:
            # Mono channel maps ⊕ selection attention over holidays.
            channels = self._mono_channel_scalars(inputs)
            if self.level1_selection_attention:
                attn_logits = self.selection_logits(channels)
                attn_weights = self.selection_softmax(attn_logits)
                if self.attention_entropy_weight > 0 and self.n_holidays > 1:
                    entropy = -tf.reduce_sum(
                        attn_weights * tf.math.log(attn_weights + 1e-8),
                        axis=-1,
                    )
                    entropy_loss = (
                        self.present
                        * self.attention_entropy_weight
                        * tf.reduce_mean(entropy)
                    )
                    self.add_loss(entropy_loss)
            else:
                # Ablation: equal 1/n over holiday mono channels.
                n_h = tf.cast(tf.shape(channels)[-1], tf.float32)
                attn_weights = tf.ones_like(channels) / tf.maximum(n_h, 1.0)
            output = tf.reduce_sum(
                channels * attn_weights, axis=-1, keepdims=True
            )
            if self.use_sku_shift_scale and sku_embedding is not None:
                beta = self.sku_beta(sku_embedding)
                alpha = self.sku_alpha(sku_embedding)
                output = self.sku_multiply([output, beta])
                output = self.sku_add([output, alpha])
            # softsign (default) bounds signed expert impact.
            return _apply_output_activation(output, self.output_activation)

        # Legacy unconstrained path (may be non-monotone in distance).
        attended_holidays = []
        
        for i in range(self.n_holidays):
            holiday_dist = inputs[:, i:i+1]
            cp_features = self.changepoint_layers[i](holiday_dist)
            cp_features = self.changepoint_norms[i](cp_features)
            cp_reshaped = tf.expand_dims(cp_features, axis=-1)
            attn_logits = self.per_holiday_attention[i](cp_reshaped)
            attn_logits = tf.squeeze(attn_logits, axis=-1)

            if self.n_changepoints == 1:
                attn_weights = tf.nn.sigmoid(
                    attn_logits / self.attention_temperature
                )
            else:
                attn_weights = tf.nn.softmax(
                    attn_logits / self.attention_temperature, axis=-1
                )

            if self.attention_entropy_weight > 0 and self.n_changepoints > 1:
                entropy = -tf.reduce_sum(
                    attn_weights * tf.math.log(attn_weights + 1e-8),
                    axis=-1
                )
                entropy_loss = self.present * self.attention_entropy_weight * tf.reduce_mean(
                    entropy
                )
                self.add_loss(entropy_loss)

            attended_cp = cp_features * attn_weights
            hidden = self.per_holiday_hidden[i](attended_cp)
            attended_holidays.append(hidden)
        
        aggregated = tf.concat(attended_holidays, axis=-1)
        agg_hidden = self.aggregate_hidden(aggregated)
        agg_attn_logits = self.aggregate_attention(agg_hidden)
        agg_attn_weights = tf.nn.softmax(agg_attn_logits / self.attention_temperature, axis=-1)
        final_attended = agg_hidden * agg_attn_weights
        final_attended = self.dropout_layer(final_attended, training=training)
        
        if self.use_sku_shift_scale and sku_embedding is not None:
            beta = self.sku_beta(sku_embedding)
            alpha = self.sku_alpha(sku_embedding)
            final_attended = self.sku_multiply([final_attended, beta])
            final_attended = self.sku_add([final_attended, alpha])
        
        output = self.output_layer(final_attended)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_changepoints': self.n_changepoints,
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'use_sku_shift_scale': self.use_sku_shift_scale,
            'attention_temperature': self.attention_temperature,
            'attention_entropy_weight': self.attention_entropy_weight,
            'output_activation': self.output_activation,
            'present': self.present_value,
            'holiday_monotonic': self.holiday_monotonic,
            'level1_selection_attention': self.level1_selection_attention,
        })
        return config


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="RegressorComponentLightweight"
)
class RegressorComponentLightweight(tf.keras.layers.Layer):
    """
    Regressor component for lag / intermittent features.

    When ``regressor_monotonic=True`` (default), each channel uses softplus-
    constrained ChangepointReLU hinge slopes with a learned per-channel sign
    (``softplus(raw) * tanh(raw_sign)``) so each channel map is monotone in
    that feature. Direction is not a hyperparameter. Stacked mono scalars then
    pass through **MaskedEntropyAttention** over lag channels → Dense / FiLM /
    softsign (mono ⊕ lag attention, not XOR).

    When ``level1_selection_attention=False``, lag attention uses uniform
    ``1/n`` channel weights (ablation for Level-1 intra-expert selection).

    When ``regressor_monotonic=False``, uses the legacy masked-attention + Dense
    path (unconstrained) on raw lag features.
    """
    
    def __init__(
        self,
        hidden_dim=32,
        dropout_rate=0.1,
        use_sku_shift_scale=True,
        activation='mish',
        output_activation='softsign',
        present=1.0,
        regressor_monotonic=True,
        level1_selection_attention=True,
        n_changepoints=5,
        feature_min=0.0,
        feature_max=100.0,
        name='regressor_lightweight',
        **kwargs
    ):
        # Ignore legacy direction kwargs from older saved configs.
        kwargs.pop('regressor_monotonic_direction', None)
        super(RegressorComponentLightweight, self).__init__(
            name=name, **kwargs
        )
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_sku_shift_scale = use_sku_shift_scale
        self.activation = activation
        self.output_activation = output_activation
        self.present_value = present
        self.present = tf.constant(present, dtype=tf.float32)
        self.regressor_monotonic = bool(regressor_monotonic)
        self.level1_selection_attention = bool(level1_selection_attention)
        self.n_changepoints = int(n_changepoints)
        self.feature_min = float(feature_min)
        self.feature_max = float(feature_max)
        
    def build(self, input_shape):
        n_features = int(input_shape[-1])
        self.n_features = n_features

        if self.regressor_monotonic:
            # Per-channel ordered hinges on the raw regressor axis.
            self.changepoint_layers = []
            for i in range(n_features):
                self.changepoint_layers.append(
                    ChangepointReLU(
                        n_changepoints=self.n_changepoints,
                        time_min=self.feature_min,
                        time_max=self.feature_max,
                        name=f'{self.name}_cp_{i}',
                    )
                )
            self.raw_slopes = self.add_weight(
                name='raw_slopes',
                shape=(n_features, self.n_changepoints),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
            self.raw_sign = self.add_weight(
                name='raw_sign',
                shape=(n_features,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )
            self.regressor_bias = self.add_weight(
                name='regressor_bias',
                shape=(n_features,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=tf.float32,
            )

        # Lag selection attention (mono path: on softplus-PWL scalars;
        # unconstrained path: on raw lag features).
        self.attention = MaskedEntropyAttention(
            units=self.hidden_dim,
            temperature=0.5,
            entropy_weight=0.01,
            dropout_rate=self.dropout_rate,
            present=self.present_value,
            equal_weights=not self.level1_selection_attention,
            name=f'{self.name}_attention'
        )

        self.ar_layer = Dense(
            self.hidden_dim // 2,
            activation=mish,
            use_bias=False,
            name=f'{self.name}_ar_pattern'
        )
        sku_units = self.hidden_dim // 2

        self.output_layer = Dense(
            1,
            activation=_resolve_output_activation(self.output_activation),
            use_bias=False,
            kernel_constraint=UnitNorm(axis=0),
            name=f'{self.name}_output'
        )

        if self.use_sku_shift_scale:
            self.sku_beta = Dense(
                sku_units,
                activation='softplus',
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(1e-5),
                name=f'{self.name}_sku_beta'
            )
            self.sku_alpha = Dense(
                sku_units,
                activation='linear',
                use_bias=False,
                kernel_regularizer=keras.regularizers.l2(1e-5),
                name=f'{self.name}_sku_alpha'
            )
            self.sku_multiply = Multiply(name=f'{self.name}_sku_multiply')
            self.sku_add = Add(name=f'{self.name}_sku_add')
        
        super(RegressorComponentLightweight, self).build(input_shape)

    def _monotone_slopes(self, feature_index):
        return (
            tf.nn.softplus(self.raw_slopes[feature_index])
            * tf.tanh(self.raw_sign[feature_index])
        )

    def _mono_channel_scalars(self, inputs):
        """Per-lag softplus-PWL scalars; each is mono in its own feature."""
        parts = []
        for i in range(self.n_features):
            x_i = inputs[:, i:i + 1]
            cp_features = self.changepoint_layers[i](x_i)
            slopes = self._monotone_slopes(i)
            part = tf.reduce_sum(
                cp_features * slopes, axis=-1, keepdims=True
            ) + self.regressor_bias[i]
            parts.append(part)
        return tf.concat(parts, axis=-1)
    
    def call(self, inputs, sku_embedding=None, training=None):
        """
        Args:
            inputs: [batch, n_lag_features]
            sku_embedding: [batch, sku_dim] SKU embedding
        
        Returns:
            regressor_forecast: [batch, 1]
        """
        if self.regressor_monotonic:
            # Mono channel maps ⊕ lag attention over those scalars.
            attn_inputs = self._mono_channel_scalars(inputs)
        else:
            attn_inputs = inputs

        attended_features = self.attention(attn_inputs, training=training)
        ar = self.ar_layer(attended_features)
        
        if self.use_sku_shift_scale and sku_embedding is not None:
            beta = self.sku_beta(sku_embedding)
            alpha = self.sku_alpha(sku_embedding)
            ar = self.sku_multiply([ar, beta])
            ar = self.sku_add([ar, alpha])
        
        output = self.output_layer(ar)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'use_sku_shift_scale': self.use_sku_shift_scale,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'present': self.present_value,
            'regressor_monotonic': self.regressor_monotonic,
            'level1_selection_attention': self.level1_selection_attention,
            'n_changepoints': self.n_changepoints,
            'feature_min': self.feature_min,
            'feature_max': self.feature_max,
        })
        return config


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="CrossLayerLightweight"
)
class CrossLayerLightweight(tf.keras.layers.Layer):
    """
    DCN Cross layer using TensorFlow Recommenders.
    
    Learns multiplicative interactions between components using DCN v2.
    """
    
    def __init__(self, projection_dim=None, name='cross_layer', **kwargs):
        super(CrossLayerLightweight, self).__init__(name=name, **kwargs)
        self.projection_dim = projection_dim
        self._cross_layer = None
    
    def build(self, input_shape):
        # input_shape: list of [(batch, 1), (batch, 1), ...]
        n_components = len(input_shape)
        
        # Use TensorFlow Recommenders' Cross layer
        # It implements: x_0 * (W * x_l + b) + x_l
        self._cross_layer = tfrs.layers.dcn.Cross(
            projection_dim=self.projection_dim,
            name=f'{self.name}_dcn'
        )
        
        super(CrossLayerLightweight, self).build(input_shape)
    
    def call(self, component_outputs):
        """
        Args:
            component_outputs: List of [batch, 1] tensors from each component
        
        Returns:
            cross_output: [batch, n_components] - DCN cross features
        """
        # Stack components: [batch, n_components]
        stacked = tf.concat(component_outputs, axis=-1)
        
        # Apply DCN Cross layer
        # Output: x_0 * (W * x + b) + x  where x_0 is the input
        cross_output = self._cross_layer(stacked, stacked)
        
        return cross_output
    
    def compute_output_shape(self, input_shape):
        """DCN Cross maintains input dimension."""
        n_components = len(input_shape)
        batch_size = input_shape[0][0]
        return (batch_size, n_components)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'projection_dim': self.projection_dim,
        })
        return config


def _logit_probability(p: float) -> float:
    """Stable logit for prior probability initialization."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def estimate_zero_rate_by_sku(y, sku_ids, n_skus=None, min_obs=1):
    """Estimate per-SKU zero rates with panel-mean fallback.

    For each SKU with ``count >= min_obs``, the rate is
    ``mean(y ≈ 0)`` on that SKU's rows. Unseen or sparse SKUs
    (``count < min_obs``) receive the **panel mean** so cold-start
    IDs still get a calibrated prior.

    Args:
        y: Demand targets, shape ``[N]`` or ``[N, 1]``.
        sku_ids: Integer SKU indices aligned with ``y``, shape ``[N]``
            or ``[N, 1]``.
        n_skus: Embedding vocabulary size. Defaults to
            ``max(sku_ids) + 1``.
        min_obs: Minimum rows required before using the SKU's own rate.

    Returns:
        dict with:
          - ``rates``: ``np.ndarray`` shape ``[n_skus]`` (float32)
          - ``counts``: ``np.ndarray`` shape ``[n_skus]`` (int64)
          - ``panel_mean``: float panel zero rate
          - ``n_skus``: int
    """
    y = np.asarray(y).reshape(-1)
    sku_ids = np.asarray(sku_ids).reshape(-1).astype(np.int64)
    if y.shape[0] != sku_ids.shape[0]:
        raise ValueError(
            f"y and sku_ids length mismatch: {y.shape[0]} vs {sku_ids.shape[0]}"
        )
    if sku_ids.size == 0:
        raise ValueError("y / sku_ids must be non-empty")
    if np.any(sku_ids < 0):
        raise ValueError("sku_ids must be non-negative integer indices")
    if n_skus is None:
        n_skus = int(sku_ids.max()) + 1
    else:
        n_skus = int(n_skus)
        if int(sku_ids.max()) >= n_skus:
            raise ValueError(
                f"sku_ids max {int(sku_ids.max())} >= n_skus={n_skus}"
            )

    is_zero = np.isclose(y.astype(np.float64), 0.0)
    panel_mean = float(np.mean(is_zero))
    counts = np.bincount(sku_ids, minlength=n_skus).astype(np.int64)
    zero_counts = np.bincount(
        sku_ids, weights=is_zero.astype(np.float64), minlength=n_skus
    )
    rates = np.full(n_skus, panel_mean, dtype=np.float64)
    enough = counts >= int(min_obs)
    rates[enough] = zero_counts[enough] / counts[enough].astype(np.float64)
    return {
        "rates": rates.astype(np.float32),
        "counts": counts,
        "panel_mean": panel_mean,
        "n_skus": n_skus,
    }


def pos_weight_from_zero_rate(zero_rate, cap=20.0):
    """Map a zero rate to BCE positive-class weight ``zr / (1 - zr)``."""
    zr = float(zero_rate)
    weight = zr / max(1.0 - zr, 1e-6)
    if cap is None:
        return float(weight)
    return float(min(float(cap), weight))


def bce_sample_weights_from_sku_zero_rates(
    y,
    sku_ids,
    zero_rates,
    *,
    cap=20.0,
    weight_zero=1.0,
    reference_zero_rate=None,
):
    """Per-sample BCE weights from each row's SKU zero rate.

    Non-zero rows get ``pos_weight_from_zero_rate(rates[sku])``; zero rows
    get ``weight_zero``. Use as ``sample_weight`` for the
    ``non_zero_probability`` head when enabling per-SKU class imbalance.

    When ``reference_zero_rate`` is set (typically the panel mean used to
    compile ``weighted_bce_loss(pos_weight=...)``), non-zero weights are
    **relative** ``sku_pos / panel_pos`` so Keras sample weights compose
    with the compiled scalar instead of double-counting.

    Without ``reference_zero_rate``, weights are absolute SKU pos-weights;
    compile BCE with ``pos_weight=1.0`` in that case.
    """
    y = np.asarray(y).reshape(-1)
    sku_ids = np.asarray(sku_ids).reshape(-1).astype(np.int64)
    rates = np.asarray(zero_rates, dtype=np.float64).reshape(-1)
    if y.shape[0] != sku_ids.shape[0]:
        raise ValueError(
            f"y and sku_ids length mismatch: {y.shape[0]} vs {sku_ids.shape[0]}"
        )
    if np.any(sku_ids < 0) or np.any(sku_ids >= rates.shape[0]):
        raise ValueError("sku_ids out of range for zero_rates")
    is_nonzero = ~np.isclose(y.astype(np.float64), 0.0)
    pos_w = np.array(
        [pos_weight_from_zero_rate(rates[i], cap=cap) for i in sku_ids],
        dtype=np.float32,
    )
    if reference_zero_rate is not None:
        ref = pos_weight_from_zero_rate(reference_zero_rate, cap=cap)
        pos_w = pos_w / max(float(ref), 1e-6)
        # Zero-class stays at 1.0 relative to the compiled zero weight.
        z_w = 1.0
    else:
        z_w = float(weight_zero)
    weights = np.full(y.shape[0], z_w, dtype=np.float32)
    weights[is_nonzero] = pos_w[is_nonzero]
    return weights


def multioutput_bce_sample_weight_dict(
    y,
    sku_ids,
    zero_rates,
    *,
    cap=20.0,
    weight_zero=1.0,
    reference_zero_rate=None,
    bce_key="non_zero_probability",
    other_keys=("final_forecast", "base_forecast"),
):
    """Keras ``sample_weight`` dict: per-SKU BCE weights, ones elsewhere."""
    bce_w = bce_sample_weights_from_sku_zero_rates(
        y,
        sku_ids,
        zero_rates,
        cap=cap,
        weight_zero=weight_zero,
        reference_zero_rate=reference_zero_rate,
    )
    ones = np.ones_like(bce_w, dtype=np.float32)
    out = {str(bce_key): bce_w}
    for k in other_keys:
        out[str(k)] = ones
    return out


def _softplus_inverse(y: float) -> float:
    """Inverse softplus for positive scale initializers: softplus(x)=y."""
    y = float(max(y, 1e-6))
    # softplus(x) = log(1+exp(x)); for y>~20 use y directly
    if y > 20.0:
        return y
    return float(np.log(np.expm1(y)))


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="GateProbabilityScale"
)
class GateProbabilityScale(tf.keras.layers.Layer):
    """Train-time multiplicative cool/heat scale on non-zero probability.

    ``p_cal = clip(p * softplus(raw), eps, 1-eps)``.

    Init ``init_scale=0.85`` matches the post-hoc IWMAE scale that cooled a hot
    gate on the locked daily panel. ``init_scale=1`` + ``trainable=False`` is a
    no-op (paper path can omit this layer entirely).

    Optional ``rate_match_weight`` softly pushes batch-mean ``p_cal`` toward
    ``rate_match_target`` (typically the empirical non-zero rate).
    """

    PROBABILITY_EPSILON = 1e-6

    def __init__(
        self,
        init_scale=1.0,
        trainable_scale=True,
        rate_match_weight=0.0,
        rate_match_target=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.init_scale = float(init_scale)
        self.trainable_scale = bool(trainable_scale)
        self.rate_match_weight = float(rate_match_weight)
        self.rate_match_target = (
            None if rate_match_target is None else float(rate_match_target)
        )

    def build(self, input_shape):
        self._scale_raw = self.add_weight(
            name=f'{self.name}_scale_raw',
            shape=(),
            initializer=tf.keras.initializers.Constant(
                _softplus_inverse(self.init_scale)
            ),
            trainable=self.trainable_scale,
        )
        super().build(input_shape)

    def call(self, inputs):
        scale = tf.nn.softplus(self._scale_raw) + tf.constant(
            1e-6, dtype=inputs.dtype
        )
        p_cal = tf.clip_by_value(
            inputs * scale,
            self.PROBABILITY_EPSILON,
            1.0 - self.PROBABILITY_EPSILON,
        )
        if self.rate_match_weight > 0.0 and self.rate_match_target is not None:
            mean_p = tf.reduce_mean(p_cal)
            target = tf.constant(self.rate_match_target, dtype=p_cal.dtype)
            self.add_loss(
                tf.constant(self.rate_match_weight, dtype=p_cal.dtype)
                * tf.square(mean_p - target)
            )
        return p_cal

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'init_scale': self.init_scale,
                'trainable_scale': self.trainable_scale,
                'rate_match_weight': self.rate_match_weight,
                'rate_match_target': self.rate_match_target,
            }
        )
        return config


@keras.saving.register_keras_serializable(
    package=KERAS_PACKAGE,
    name="IntermittentHandlerLightweight"
)
class IntermittentHandlerLightweight(tf.keras.layers.Layer):
    """
    Lightweight intermittent demand handler.

    Predicts zero probability from component (and optional raw regressor) features.

    Calibration knobs (defaults preserve the historical paper path):
      - ``prior_zero_rate``: bias-init the gate logit toward the panel zero rate
        (fallback when per-SKU rates are unavailable)
      - ``prior_zero_rates``: optional length-``n_skus`` array; when set, call
        expects ``[features, sku_id]`` and adds a **non-trainable** Embedding
        of ``logit(zero_rate_sku)`` to the gate logits (SKU-conditioned prior)
      - ``temperature``: divide logits before sigmoid (legacy; ``>1`` softens)
      - ``learnable_temperature``: legacy softplus *divider* (softens toward 0.5)
      - ``learnable_logit_scale``: softplus *multiplier* on logits before sigmoid.
        Scale ``>1`` sharpens: cools mid-range non-zero probs that sit below 0.5.
        Prefer this over ``learnable_temperature`` for hot-gate reclaim.
      - ``rate_match_weight`` / ``rate_match_target``: soft push of mean
        *non-zero* probability (``1 - zero_prob``) toward the empirical rate.
    """

    def __init__(
        self,
        hidden_dim=16,
        dropout_rate=0.1,
        entropy_weight=1e-5,
        present=1.0,
        prior_zero_rate=None,
        prior_zero_rates=None,
        n_skus=None,
        temperature=1.0,
        learnable_temperature=False,
        learnable_logit_scale=False,
        logit_scale_init=1.0,
        rate_match_weight=0.0,
        rate_match_target=None,
        name='intermittent_lightweight',
        **kwargs,
    ):
        super(IntermittentHandlerLightweight, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.entropy_weight = entropy_weight
        self.present_value = present
        self.present = tf.constant(present, dtype=tf.float32)
        if prior_zero_rates is not None:
            rates = np.asarray(prior_zero_rates, dtype=np.float32).reshape(-1)
            if rates.size == 0:
                raise ValueError("prior_zero_rates must be non-empty when provided")
            self.prior_zero_rates = rates
            self.n_skus = int(n_skus) if n_skus is not None else int(rates.shape[0])
            if self.n_skus != int(rates.shape[0]):
                raise ValueError(
                    f"n_skus={self.n_skus} != len(prior_zero_rates)={rates.shape[0]}"
                )
            # SKU embedding carries the prior; avoid double-counting panel bias.
            self.prior_zero_rate = None
        else:
            self.prior_zero_rates = None
            self.n_skus = None if n_skus is None else int(n_skus)
            self.prior_zero_rate = (
                None if prior_zero_rate is None else float(prior_zero_rate)
            )
        self.temperature = float(temperature)
        self.learnable_temperature = bool(learnable_temperature)
        self.learnable_logit_scale = bool(learnable_logit_scale)
        self.logit_scale_init = float(logit_scale_init)
        self.rate_match_weight = float(rate_match_weight)
        self.rate_match_target = (
            None if rate_match_target is None else float(rate_match_target)
        )
        self.sku_prior_embedding = None

    def _unpack_inputs(self, inputs):
        """Split optional ``[features_or_list, sku_id]`` from component lists."""
        sku_ids = None
        feature_inputs = inputs
        if (
            self.prior_zero_rates is not None
            and isinstance(inputs, (list, tuple))
            and len(inputs) == 2
        ):
            feature_inputs, sku_ids = inputs[0], inputs[1]
        if isinstance(feature_inputs, (list, tuple)):
            features = tf.concat(list(feature_inputs), axis=-1)
        else:
            features = feature_inputs
        return features, sku_ids

    def build(self, input_shape):
        feature_shape = input_shape
        if (
            self.prior_zero_rates is not None
            and isinstance(input_shape, (list, tuple))
            and len(input_shape) == 2
        ):
            feature_shape = input_shape[0]

        if isinstance(feature_shape, (list, tuple)):
            # List of component tensors (each [None, 1]) or a single shape tuple.
            if len(feature_shape) > 0 and hasattr(feature_shape[0], "as_list"):
                n_features = len(feature_shape)
            elif all(isinstance(s, (tuple, list)) for s in feature_shape):
                n_features = len(feature_shape)
            else:
                n_features = feature_shape[-1]
                if hasattr(n_features, "as_list"):
                    n_features = int(n_features) if n_features is not None else None
        else:
            n_features = feature_shape[-1]
            if hasattr(n_features, "value"):
                n_features = n_features.value
            if n_features is not None:
                try:
                    n_features = int(n_features)
                except (TypeError, ValueError):
                    n_features = None

        self.zero_prob_layer1 = Dense(
            self.hidden_dim,
            activation=mish,
            use_bias=False,
            name=f'{self.name}_zero_hidden',
        )
        bias_init = "zeros"
        if self.prior_zero_rate is not None:
            bias_init = tf.keras.initializers.Constant(
                _logit_probability(self.prior_zero_rate)
            )
        self.zero_prob_output = Dense(
            1,
            activation=None,
            use_bias=True,
            bias_initializer=bias_init,
            name=f'{self.name}_zero_logit',
        )
        self.dropout = Dropout(self.dropout_rate)
        self.zero_prob_clip = ClipByValue(
            1e-8,
            1.0 - 1e-8,
            name=f'{self.name}_probability_clip',
        )
        if self.learnable_temperature:
            self._temp_raw = self.add_weight(
                name=f'{self.name}_temp_raw',
                shape=(),
                initializer=tf.keras.initializers.Constant(0.0),
                trainable=True,
            )
        else:
            self._temp_raw = None
        if self.learnable_logit_scale:
            self._logit_scale_raw = self.add_weight(
                name=f'{self.name}_logit_scale_raw',
                shape=(),
                initializer=tf.keras.initializers.Constant(
                    _softplus_inverse(self.logit_scale_init)
                ),
                trainable=True,
            )
        else:
            self._logit_scale_raw = None

        if self.prior_zero_rates is not None:
            prior_logits = np.asarray(
                [_logit_probability(float(r)) for r in self.prior_zero_rates],
                dtype=np.float32,
            ).reshape(-1, 1)
            self.sku_prior_embedding = Embedding(
                input_dim=self.n_skus,
                output_dim=1,
                embeddings_initializer=tf.keras.initializers.Constant(prior_logits),
                trainable=False,
                name=f'{self.name}_sku_zero_prior',
            )
        else:
            self.sku_prior_embedding = None

        if n_features is not None:
            self.zero_prob_layer1.build((None, n_features))
            self.zero_prob_output.build((None, self.hidden_dim))
        if self.sku_prior_embedding is not None:
            self.sku_prior_embedding.build((None, 1))

        super(IntermittentHandlerLightweight, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        feature_shape = input_shape
        if (
            self.prior_zero_rates is not None
            and isinstance(input_shape, (list, tuple))
            and len(input_shape) == 2
        ):
            feature_shape = input_shape[0]
        if isinstance(feature_shape, list):
            batch_size = feature_shape[0][0]
        else:
            batch_size = feature_shape[0]
        return (batch_size, 1)

    def _gate_temperature(self):
        if self.learnable_temperature:
            return tf.nn.softplus(self._temp_raw) + tf.constant(1e-3, dtype=tf.float32)
        return tf.constant(max(self.temperature, 1e-3), dtype=tf.float32)

    def _logit_scale(self):
        if self.learnable_logit_scale:
            return tf.nn.softplus(self._logit_scale_raw) + tf.constant(
                1e-3, dtype=tf.float32
            )
        return tf.constant(1.0, dtype=tf.float32)

    def call(self, inputs, training=None):
        features, sku_ids = self._unpack_inputs(inputs)

        hidden = self.zero_prob_layer1(features)
        hidden = self.dropout(hidden, training=training)
        logits = self.zero_prob_output(hidden)
        if self.sku_prior_embedding is not None:
            if sku_ids is None:
                raise ValueError(
                    "IntermittentHandlerLightweight with prior_zero_rates "
                    "expects inputs [features, sku_id]"
                )
            prior_logit = self.sku_prior_embedding(sku_ids)
            prior_logit = tf.squeeze(prior_logit, axis=1)
            logits = logits + prior_logit
        # Sharpen (multiply) then optional legacy soften (divide).
        logits = logits * self._logit_scale()
        zero_prob = tf.nn.sigmoid(logits / self._gate_temperature())

        zero_prob_safe = self.zero_prob_clip(zero_prob)
        entropy = -zero_prob_safe * tf.math.log(zero_prob_safe) - (
            1.0 - zero_prob_safe
        ) * tf.math.log(1.0 - zero_prob_safe)
        entropy = tf.squeeze(entropy, axis=-1)
        self.add_loss(
            self.present * self.entropy_weight * tf.reduce_mean(entropy)
        )

        if self.rate_match_weight > 0.0 and self.rate_match_target is not None:
            # Match non-zero probability mean to empirical occurrence rate.
            mean_p = tf.reduce_mean(1.0 - zero_prob_safe)
            target = tf.constant(self.rate_match_target, dtype=zero_prob_safe.dtype)
            self.add_loss(
                self.present
                * tf.constant(self.rate_match_weight, dtype=zero_prob_safe.dtype)
                * tf.square(mean_p - target)
            )
        return zero_prob

    def get_config(self):
        config = super(IntermittentHandlerLightweight, self).get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'entropy_weight': self.entropy_weight,
            'present': self.present_value,
            'prior_zero_rate': self.prior_zero_rate,
            'prior_zero_rates': (
                None
                if self.prior_zero_rates is None
                else [float(x) for x in self.prior_zero_rates]
            ),
            'n_skus': self.n_skus,
            'temperature': self.temperature,
            'learnable_temperature': self.learnable_temperature,
            'learnable_logit_scale': self.learnable_logit_scale,
            'logit_scale_init': self.logit_scale_init,
            'rate_match_weight': self.rate_match_weight,
            'rate_match_target': self.rate_match_target,
        })
        return config


def _resolve_component_flags(
    n_temporal_features,
    n_fourier_features,
    n_holiday_features,
    n_lag_features,
    enable_trend,
    enable_seasonal,
    enable_holiday,
    enable_regressor,
    use_learnable_fourier,
):
    """Resolve optional component switches from available feature counts."""
    if enable_trend is None:
        enable_trend = bool(n_temporal_features and n_temporal_features > 0)
    if enable_seasonal is None:
        enable_seasonal = (
            True
            if use_learnable_fourier
            else bool(n_fourier_features and n_fourier_features > 0)
        )
    if enable_holiday is None:
        enable_holiday = bool(n_holiday_features and n_holiday_features > 0)
    if enable_regressor is None:
        enable_regressor = bool(n_lag_features and n_lag_features > 0)
    return enable_trend, enable_seasonal, enable_holiday, enable_regressor


def _build_sku_path(sku_input, n_skus, sku_embedding_dim, use_sku):
    """Build ID personalization only when SKU pooling is enabled."""
    if not use_sku:
        return None
    embedding = Embedding(
        input_dim=n_skus,
        output_dim=sku_embedding_dim,
        name='sku_embedding',
    )(sku_input)
    return SqueezeLayer(axis=1, name='sku_embedding_squeeze')(embedding)


def _build_components(
    temporal_input,
    fourier_input,
    holiday_input,
    lag_input,
    sku_embedding,
    *,
    n_temporal_features,
    n_fourier_features,
    n_holiday_features,
    n_lag_features,
    n_changepoints,
    hidden_dim,
    dropout_rate,
    activation,
    output_activation,
    use_sku,
    use_learnable_fourier,
    n_learnable_frequencies,
    fourier_periods,
    fourier_min_period,
    fourier_max_period,
    enable_trend,
    enable_seasonal,
    enable_holiday,
    enable_regressor,
    trend_monotonic=True,
    holiday_monotonic=True,
    regressor_monotonic=True,
    level1_selection_attention=True,
):
    """Build the four structural components and their presence flags."""
    sku_arg = sku_embedding if use_sku else None

    trend_present = (
        1.0
        if (enable_trend and n_temporal_features and n_temporal_features > 0)
        else 0.0
    )
    trend = TrendComponentLightweight(
        n_changepoints=n_changepoints,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        output_activation=output_activation,
        use_sku_shift_scale=use_sku,
        present=trend_present,
        trend_monotonic=trend_monotonic,
        name='trend',
    )(temporal_input, sku_embedding=sku_arg)
    trend = trend * tf.constant(1.0 if enable_trend else 0.0, dtype=tf.float32)

    seasonal_present = (
        1.0
        if (
            enable_seasonal
            and (
                use_learnable_fourier
                or (n_fourier_features and n_fourier_features > 0)
            )
        )
        else 0.0
    )
    seasonal = SeasonalComponentLightweight(
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        activation=activation,
        output_activation=output_activation,
        use_sku_shift_scale=use_sku,
        present=seasonal_present,
        use_learnable_fourier=use_learnable_fourier,
        n_learnable_frequencies=n_learnable_frequencies,
        fourier_periods=fourier_periods,
        fourier_min_period=fourier_min_period,
        fourier_max_period=fourier_max_period,
        name='seasonal',
    )(fourier_input, sku_embedding=sku_arg)
    seasonal = seasonal * tf.constant(
        1.0 if enable_seasonal else 0.0, dtype=tf.float32
    )

    holiday_present = (
        1.0
        if (enable_holiday and n_holiday_features and n_holiday_features > 0)
        else 0.0
    )
    if n_holiday_features and n_holiday_features > 0:
        holiday = HolidayComponentLightweight(
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            output_activation=output_activation,
            use_sku_shift_scale=use_sku,
            present=holiday_present,
            holiday_monotonic=holiday_monotonic,
            level1_selection_attention=level1_selection_attention,
            name='holiday',
        )(holiday_input, sku_embedding=sku_arg)
    else:
        holiday = Dense(
            1,
            use_bias=False,
            trainable=False,
            kernel_initializer='zeros',
            name='holiday',
        )(holiday_input)
    holiday = holiday * tf.constant(
        1.0 if (enable_holiday and holiday_present == 1.0) else 0.0,
        dtype=tf.float32,
    )

    regressor_present = (
        1.0
        if (enable_regressor and n_lag_features and n_lag_features > 0)
        else 0.0
    )
    regressor = RegressorComponentLightweight(
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        activation=activation,
        output_activation=output_activation,
        use_sku_shift_scale=use_sku,
        present=regressor_present,
        regressor_monotonic=regressor_monotonic,
        level1_selection_attention=level1_selection_attention,
        name='regressor',
    )(lag_input, sku_embedding=sku_arg)
    regressor = regressor * tf.constant(
        1.0 if enable_regressor else 0.0, dtype=tf.float32
    )

    presence = {
        'trend': trend_present,
        'seasonal': seasonal_present,
        'holiday': holiday_present,
        'regressor': regressor_present,
    }
    flags = [
        bool(enable_trend and trend_present == 1.0),
        bool(enable_seasonal and seasonal_present == 1.0),
        bool(enable_holiday and holiday_present == 1.0),
        bool(enable_regressor and regressor_present == 1.0),
    ]
    return [trend, seasonal, holiday, regressor], presence, flags


# softplus(x + SOFTPLUS_ONE) = 1 when x = 0 (identity FiLM scale at zeros init).
_SOFTPLUS_ONE = float(np.log(np.e - 1.0))


def _context_film_modulate(expert, context, name):
    """Apply lag-conditioned FiLM: softplus scale (≈1 at init) + softsign shift."""
    scale_raw = Dense(
        1,
        use_bias=False,
        kernel_initializer='zeros',
        name=f'context_film_{name}_scale',
    )(context)
    shift_raw = Dense(
        1,
        use_bias=False,
        kernel_initializer='zeros',
        name=f'context_film_{name}_shift',
    )(context)
    scale = tf.nn.softplus(scale_raw + _SOFTPLUS_ONE)
    shift = tf.nn.softsign(shift_raw)
    scaled = Multiply(name=f'context_film_{name}_mul')([expert, scale])
    return Add(name=f'context_film_{name}_add')([scaled, shift])


def _apply_context_film_seasonal_holiday(
    seasonal,
    holiday,
    context_features,
    *,
    enabled,
    hidden_dim,
    context_dim=None,
):
    """Modulate seasonal/holiday scalars with lag-derived FiLM (post softsign/SKU).

    Conditioner is a small projection of ``lag_input`` — the same context tensor
    the mixer sees — not the regressor expert, so seasonal/holiday conditioning
    does not create a hard dependency cycle through mixer weights. Trend and
    regressor are left unchanged. When ``enabled`` is False or context is
    unavailable, returns the experts unchanged (legacy identity).
    """
    if not enabled or context_features is None:
        return seasonal, holiday

    proj_dim = (
        int(context_dim)
        if context_dim is not None
        else max(4, int(hidden_dim) // 4)
    )
    context = Dense(
        proj_dim,
        activation=mish,
        use_bias=False,
        name='context_film_proj',
    )(context_features)
    seasonal = _context_film_modulate(seasonal, context, 'seasonal')
    holiday = _context_film_modulate(holiday, context, 'holiday')
    return seasonal, holiday


def _build_component_mixer_source(
    sku_embedding,
    context_features,
    *,
    use_sku,
    context_aware_component_mixer,
    hidden_dim,
    context_dim=None,
):
    """Build the query tensor for component-attention mixing.

    Context is lag / intermittent (and related regressor) regime signals already
    present as model inputs — not calendar, Fourier, or holiday distances
    (those stay inside their experts).

    When ``context_aware_component_mixer`` is True and ``context_features`` is
    available, projects context through a small Dense (``use_bias=False``) and
    combines with the SKU embedding via concat. SKU-off ⇒ context-only.
    When False (legacy / paper protocol), returns the SKU embedding only, or
    ``None`` so ``_build_component_attention`` falls back to stacked experts.
    """
    if (
        not context_aware_component_mixer
        or context_features is None
    ):
        return sku_embedding if use_sku else None

    proj_dim = (
        int(context_dim)
        if context_dim is not None
        else max(4, int(hidden_dim) // 4)
    )
    context = Dense(
        proj_dim,
        activation=mish,
        use_bias=False,
        name='component_mixer_context',
    )(context_features)

    if use_sku and sku_embedding is not None:
        return Concatenate(name='component_mixer_source')(
            [sku_embedding, context]
        )
    return context


def _build_component_attention(
    component_outputs,
    attention_source,
    hidden_dim,
    temperature,
    entropy_weight,
    l2_weight,
    component_flags=None,
    component_combine='additive',
    multiplicative_eps=1e-3,
):
    """Combine component outputs with learned entropy-regularized attention.

    ``component_combine``:
      - ``'additive'`` (default / locked): ``Σ_k α_k e_k``
      - ``'multiplicative'``: Prophet-like
        ``softplus(e_T) Π_{k∈{S,H,R}} max(ε, 1 + α_k e_k)``
    """
    combine = str(component_combine).lower().strip()
    if combine not in ('additive', 'multiplicative'):
        raise ValueError(
            "component_combine must be 'additive' or 'multiplicative', "
            f"got {component_combine!r}"
        )
    stacked_components = StackComponentsLayer(
        name='stack_components'
    )(component_outputs)
    # When SKU pooling is off (and no context mixer source), reuse the stack.
    if attention_source is None:
        attention_source = stacked_components
    attention_hidden = Dense(
        hidden_dim,
        activation=mish,
        use_bias=False,
        name='component_attention_hidden',
    )(attention_source)
    attention_hidden = LayerNormalization(
        name='component_attention_norm'
    )(attention_hidden)
    attention_logits = Dense(
        len(component_outputs),
        activation=None,
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(l2_weight),
        name='component_attention_logits',
    )(attention_hidden)
    active_mask = (
        None
        if component_flags is None
        else [1.0 if flag else 0.0 for flag in component_flags]
    )
    attention_weights = TemperatureSoftmax(
        temperature=temperature,
        active_mask=active_mask,
        name='component_attention_softmax',
    )(attention_logits)
    attention_weights = PrintAttentionWeights(
        name='print_attention_weights'
    )(attention_weights)
    entropy = ComponentEntropy(name='component_entropy')(attention_weights)
    ComponentEntropyLoss(
        entropy_weight=entropy_weight,
        name='component_entropy_loss',
    )(entropy)
    if combine == 'multiplicative':
        flags = (
            list(component_flags)
            if component_flags is not None
            else [True] * len(component_outputs)
        )
        return MultiplicativeComponentCombine(
            component_flags=flags,
            eps=multiplicative_eps,
            name='multiplicative_component_combine',
        )([stacked_components, attention_weights])
    weighted_components = Multiply(name='apply_attention_weights')(
        [stacked_components, attention_weights]
    )
    return SumWeightedComponents(
        name='sum_weighted_components'
    )(weighted_components)


def _combine_component_forecasts(
    attention_forecast,
    component_outputs,
    use_cross_layers,
):
    """Optionally add DCN interactions to the attention-weighted forecast."""
    if not use_cross_layers:
        return attention_forecast
    interaction = CrossLayerLightweight(
        name='cross_layer_shared'
    )(component_outputs)
    interaction = LayerNormalization(name='cross_layer_norm')(interaction)
    cross_forecast = Dense(
        1,
        activation='linear',
        use_bias=False,
        name='base_forecast_cross',
    )(interaction)
    return Add(name='base_forecast_add')(
        [attention_forecast, cross_forecast]
    )


def _add_orthogonality_regularization(
    component_outputs,
    component_flags,
    weight,
):
    """Attach orthogonality loss to active components without swallowing errors."""
    if weight <= 0:
        return component_outputs
    active_indices = [
        index for index, enabled in enumerate(component_flags) if enabled
    ]
    if len(active_indices) < 2:
        return component_outputs
    active_outputs = [component_outputs[index] for index in active_indices]
    regularized = OrthogonalityPenalty(
        weight=weight,
        name='component_orthogonality',
    )(active_outputs)
    outputs = list(component_outputs)
    for index, tensor in zip(active_indices, regularized):
        outputs[index] = tensor
    return outputs



@keras.saving.register_keras_serializable(package=KERAS_PACKAGE)
class TakeLastTimestepLayer(tf.keras.layers.Layer):
    """Read out the last timestep from a sequence tensor."""

    def call(self, inputs):
        return inputs[:, -1, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()


def _build_causal_lookback_encoder(
    sequence_input,
    sku_embedding,
    *,
    lookback,
    d_model=32,
    n_heads=4,
    n_blocks=1,
    use_sku=True,
    name='temporal_encoder',
):
    """Causal MHA encoder over lookback; returns last-step context ``[B, d_model]``."""
    x = sequence_input
    if use_sku and sku_embedding is not None:
        emb = tf.keras.layers.Flatten(name=f'{name}_sku_flat')(sku_embedding)
        emb_t = tf.keras.layers.RepeatVector(lookback, name=f'{name}_sku_repeat')(emb)
        x = Concatenate(axis=-1, name=f'{name}_concat_sku')([x, emb_t])
    x = Dense(d_model, use_bias=False, name=f'{name}_in_proj')(x)
    x = LayerNormalization(name=f'{name}_in_norm')(x)
    key_dim = max(1, int(d_model) // max(1, int(n_heads)))
    for b in range(max(1, int(n_blocks))):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=int(n_heads),
            key_dim=key_dim,
            name=f'{name}_mha_{b}',
        )(x, x, use_causal_mask=True)
        x = LayerNormalization(name=f'{name}_ln_attn_{b}')(x + attn)
        ff = Dense(
            d_model * 2,
            activation='relu',
            use_bias=False,
            name=f'{name}_ff_up_{b}',
        )(x)
        ff = Dense(d_model, use_bias=False, name=f'{name}_ff_down_{b}')(ff)
        x = LayerNormalization(name=f'{name}_ln_ff_{b}')(x + ff)
    # Last lookback step (most recent past day).
    return TakeLastTimestepLayer(name=f'{name}_readout')(x)


def _build_intermittent_heads(
    component_outputs,
    base_forecast,
    sku_embedding,
    *,
    use_intermittent,
    use_cross_layers,
    use_sku,
    horizon,
    hidden_dim,
    dropout_rate,
    intermittent_hidden_dim,
    intermittent_entropy_weight,
    lag_features=None,
    gate_use_raw_regressors=False,
    intermittent_prior_zero_rate=None,
    intermittent_prior_zero_rates=None,
    n_skus=None,
    sku_input=None,
    intermittent_gate_temperature=1.0,
    intermittent_learnable_temperature=False,
    intermittent_learnable_logit_scale=False,
    intermittent_logit_scale_init=1.0,
    gate_prob_scale=False,
    gate_prob_scale_init=1.0,
    gate_prob_scale_trainable=True,
    gate_rate_match_weight=0.0,
    gate_rate_match_target=None,
    gate_raw_regressor_proj_dim=3,
    decouple_gate=False,
    temporal_context=None,
    gate_temporal_proj_dim=2,
):
    """Build gated intermittent heads for 1-step or direct multi-horizon.

    ``decouple_gate=True``: occurrence branch sees raw regressors / SKU /
    temporal context only — not softplus magnitude or component scalars.
    ``temporal_context`` (optional ``[B, d]``) is fused into magnitude and,
    when present, into the gate.

    Gate prior:
      - ``intermittent_prior_zero_rates`` (length ``n_skus``): SKU-conditioned
        non-trainable logit Embedding when ``use_sku`` and ``sku_input`` given.
      - else ``intermittent_prior_zero_rate``: panel-level Dense bias prior.
    """
    # Magnitude path: fuse optional temporal context before softplus.
    if temporal_context is not None:
        mag_ctx = Concatenate(name='magnitude_temporal_concat')(
            [base_forecast, temporal_context]
        )
        mag_ctx = Dense(
            hidden_dim,
            activation=mish,
            use_bias=False,
            name='magnitude_temporal_hidden',
        )(mag_ctx)
        mag_ctx = Dropout(dropout_rate, name='magnitude_temporal_dropout')(mag_ctx)
        base_level = Dense(
            1,
            activation='softplus',
            use_bias=True,
            name='base_level' if horizon > 1 else 'base_forecast',
        )(mag_ctx)
    else:
        base_level = Dense(
            1,
            activation='softplus',
            use_bias=True,
            name='base_level' if horizon > 1 else 'base_forecast',
        )(base_forecast)

    if not use_intermittent:
        if horizon == 1:
            return {'final_forecast': base_level}
        multi_horizon = Dense(
            horizon,
            activation='softplus',
            use_bias=True,
            name='base_forecast',
        )(base_level)
        return {'final_forecast': multi_horizon}

    if decouple_gate:
        # Occurrence must not see magnitude / component expert scalars.
        intermittent_components = []
        use_raw = True  # always feed raw regressors when decoupled
    else:
        intermittent_components = list(component_outputs) + [base_level]
        use_raw = bool(gate_use_raw_regressors)

    if use_sku:
        sku_signal = Dense(
            1,
            activation='tanh',
            use_bias=False,
            name='sku_signal_for_intermittent',
        )(sku_embedding)
        intermittent_components.append(sku_signal)

    # Direct intermittent-state / lag path into the gate.
    if (
        use_raw
        and lag_features is not None
        and int(gate_raw_regressor_proj_dim) > 0
    ):
        n_proj = int(gate_raw_regressor_proj_dim)
        lag_proj = Dense(
            n_proj,
            activation='tanh',
            use_bias=False,
            name='gate_raw_regressor_proj',
        )(lag_features)
        for i in range(n_proj):
            intermittent_components.append(
                GatherLayer([i], name=f'gate_raw_regressor_{i}')(lag_proj)
            )

    if temporal_context is not None and int(gate_temporal_proj_dim) > 0:
        n_t = int(gate_temporal_proj_dim)
        t_proj = Dense(
            n_t,
            activation='tanh',
            use_bias=False,
            name='gate_temporal_proj',
        )(temporal_context)
        for i in range(n_t):
            intermittent_components.append(
                GatherLayer([i], name=f'gate_temporal_{i}')(t_proj)
            )

    if not intermittent_components:
        raise ValueError(
            "Intermittent gate has no inputs. With decouple_gate=True provide "
            "lag_features and/or temporal_context (and optionally use_sku)."
        )

    rate_target = gate_rate_match_target
    if rate_target is None and intermittent_prior_zero_rate is not None:
        rate_target = max(1e-6, 1.0 - float(intermittent_prior_zero_rate))

    # Prefer rate-match on the calibrated non-zero path when prob-scale is on.
    handler_rate_w = (
        0.0 if gate_prob_scale else float(gate_rate_match_weight)
    )
    scale_rate_w = (
        float(gate_rate_match_weight) if gate_prob_scale else 0.0
    )

    sku_prior_rates = None
    if intermittent_prior_zero_rates is not None:
        if not use_sku:
            raise ValueError(
                "intermittent_prior_zero_rates requires use_sku=True"
            )
        if sku_input is None:
            raise ValueError(
                "intermittent_prior_zero_rates requires sku_input"
            )
        sku_prior_rates = np.asarray(
            intermittent_prior_zero_rates, dtype=np.float32
        ).reshape(-1)
        if n_skus is None:
            n_skus = int(sku_prior_rates.shape[0])
        elif int(n_skus) != int(sku_prior_rates.shape[0]):
            raise ValueError(
                f"n_skus={n_skus} != len(intermittent_prior_zero_rates)="
                f"{sku_prior_rates.shape[0]}"
            )

    handler_kwargs = dict(
        hidden_dim=intermittent_hidden_dim,
        dropout_rate=dropout_rate,
        entropy_weight=intermittent_entropy_weight,
        present=1.0,
        # Panel bias only when SKU prior Embedding is not used.
        prior_zero_rate=(
            None if sku_prior_rates is not None else intermittent_prior_zero_rate
        ),
        prior_zero_rates=sku_prior_rates,
        n_skus=int(n_skus) if sku_prior_rates is not None else None,
        temperature=intermittent_gate_temperature,
        learnable_temperature=intermittent_learnable_temperature,
        learnable_logit_scale=intermittent_learnable_logit_scale,
        logit_scale_init=intermittent_logit_scale_init,
        rate_match_weight=handler_rate_w,
        rate_match_target=rate_target if handler_rate_w > 0.0 else None,
        name='intermittent',
    )

    if use_cross_layers:
        intermittent_features = CrossLayerLightweight(
            name='cross_layer_intermittent'
        )(intermittent_components)
        intermittent_features = LayerNormalization(
            name='cross_layer_intermittent_norm'
        )(intermittent_features)
    else:
        intermittent_features = Concatenate(name='intermittent_concat')(
            intermittent_components
        )

    handler = IntermittentHandlerLightweight(**handler_kwargs)
    if sku_prior_rates is not None:
        zero_prob = handler([intermittent_features, sku_input])
    else:
        zero_prob = handler(intermittent_features)

    scheduled_stop = ScheduledStopGradient(
        initial_prob=0.0, name='scheduled_stop'
    )

    zero_prob_safe = ClipByValue(
        1e-7, 1.0 - 1e-7, name='zero_prob_clip'
    )(zero_prob)
    invert_name = (
        'non_zero_probability'
        if (horizon == 1 and not gate_prob_scale)
        else 'non_zero_base'
    )
    non_zero_base = InvertProbability(name=invert_name)(zero_prob_safe)

    if horizon == 1:
        non_zero_prob = non_zero_base
        if gate_prob_scale:
            non_zero_prob = GateProbabilityScale(
                init_scale=gate_prob_scale_init,
                trainable_scale=gate_prob_scale_trainable,
                rate_match_weight=scale_rate_w,
                rate_match_target=rate_target if scale_rate_w > 0.0 else None,
                name='non_zero_probability',
            )(non_zero_prob)
        base_forecast_out = base_level
        final_forecast = Multiply(name='final_forecast')(
            [base_forecast_out, scheduled_stop(non_zero_prob)]
        )
    else:
        mh_parts = [base_level, intermittent_features, non_zero_base]
        if use_sku:
            mh_parts.append(sku_embedding)
        head_ctx = Concatenate(name='mh_head_context')(mh_parts)
        head_ctx = Dense(
            hidden_dim,
            activation=mish,
            use_bias=False,
            name='mh_head_hidden',
        )(head_ctx)
        head_ctx = Dropout(dropout_rate, name='mh_head_dropout')(head_ctx)
        base_forecast_out = Dense(
            horizon,
            activation='softplus',
            use_bias=True,
            name='base_forecast',
        )(head_ctx)
        # Per-horizon deviations around the handler's occurrence probability,
        # so the intermittent head is on the multi-horizon gradient path.
        gate_offsets = Dense(
            horizon,
            activation=None,
            use_bias=True,
            kernel_initializer='zeros',
            name='mh_gate_offsets',
        )(head_ctx)
        non_zero_prob = HorizonGateFromBaseProbability(
            name='non_zero_base_mh' if gate_prob_scale else 'non_zero_probability'
        )([gate_offsets, non_zero_base])
        if gate_prob_scale:
            non_zero_prob = GateProbabilityScale(
                init_scale=gate_prob_scale_init,
                trainable_scale=gate_prob_scale_trainable,
                rate_match_weight=scale_rate_w,
                rate_match_target=rate_target if scale_rate_w > 0.0 else None,
                name='non_zero_probability',
            )(non_zero_prob)
        final_forecast = Multiply(name='final_forecast')(
            [base_forecast_out, scheduled_stop(non_zero_prob)]
        )

    return {
        'final_forecast': final_forecast,
        'non_zero_probability': non_zero_prob,
        'base_forecast': base_forecast_out,
    }


def build_hierarchical_model_lightweight(
    n_temporal_features,
    n_fourier_features,
    n_holiday_features,
    n_lag_features,
    n_skus,
    n_changepoints=25,
    hidden_dim=32,
    sku_embedding_dim=8,
    dropout_rate=0.1,
    use_cross_layers=False,
    use_intermittent=True,
    enable_trend=None,
    enable_seasonal=None,
    enable_holiday=None,
    enable_regressor=None,
    activation='mish',
    output_activation='softsign',
    use_learnable_fourier=False,
    n_learnable_frequencies=5,
    fourier_periods=None,
    fourier_frequency=None,
    fourier_min_period=FOURIER_MIN_PERIOD,
    fourier_max_period=None,
    horizon=1,
    use_sku=True,
    component_attention_temperature=0.7,
    component_entropy_weight=0.01,
    component_attention_l2=0.001,
    intermittent_hidden_dim=16,
    intermittent_entropy_weight=1e-5,
    orthogonality_weight=1e-4,
    gate_use_raw_regressors=False,
    intermittent_prior_zero_rate=None,
    intermittent_prior_zero_rates=None,
    intermittent_gate_temperature=1.0,
    intermittent_learnable_temperature=False,
    intermittent_learnable_logit_scale=False,
    intermittent_logit_scale_init=1.0,
    gate_prob_scale=False,
    gate_prob_scale_init=1.0,
    gate_prob_scale_trainable=True,
    gate_rate_match_weight=0.0,
    gate_rate_match_target=None,
    gate_raw_regressor_proj_dim=3,
    use_temporal_context=False,
    lookback=14,
    n_sequence_channels=None,
    temporal_d_model=32,
    temporal_n_heads=4,
    temporal_n_blocks=1,
    decouple_gate=None,
    trend_monotonic=True,
    holiday_monotonic=True,
    regressor_monotonic=True,
    level1_selection_attention=True,
    context_aware_component_mixer=True,
    component_mixer_context_dim=None,
    context_film_seasonal_holiday=False,
    component_combine='additive',
    multiplicative_eps=1e-3,
):
    """
    Build lightweight hierarchical model with masked entropy attention.
    
    Parameters much smaller than TabNet version:
    - TabNet version: ~320K parameters
    - Lightweight version: ~30K parameters (10x reduction)
    
    Args:
        n_temporal_features: Number of time features (day, month, cyclical, etc.)
        n_fourier_features: Number of seasonal Fourier features
            When ``use_learnable_fourier=True``, pass 1 (raw time in days).
        n_holiday_features: Number of holiday distance features
        n_lag_features: Number of lag features
        n_skus: Number of unique SKUs for embedding
        n_changepoints: Number of trend changepoints (default: 25)
        hidden_dim: Hidden dimension for components (default: 32)
        sku_embedding_dim: SKU embedding dimension (default: 8)
        dropout_rate: Dropout rate (default: 0.1)
        use_cross_layers: Opt-in DCN cross on component outputs (default False;
            locked A/Bs favor off). Pass True for ablation.
        use_intermittent: Whether to use intermittent demand handling
        activation: Hidden layer activation (default: 'mish')
        output_activation: Signed expert scalar activation (default: 'softsign').
            Bounds magnitude milder than tanh while keeping sign. Applied on
            legacy Dense ``output_layer`` and mono PWL scalar returns.
            Options: 'softsign', 'linear', 'sparse_amplify', 'sparse_amplify_exp',
            'relu', 'mish', 'tanh'. Not used for base softplus, gate, FiLM, or DCN.
        use_learnable_fourier: If True, seasonal ω is trainable; fourier input
            must be ``[batch, 1]`` elapsed time in the same unit as the periods
            (not precomputed sin/cos).
        n_learnable_frequencies: Number of learnable (sin, cos) frequency pairs
        fourier_periods: Optional initial periods for learnable ω, in *time
            steps of the series*: [7, 30.44, 365.25] at daily grain but
            [3, 6, 12] at monthly grain. Overrides ``fourier_frequency``.
        fourier_frequency: Sampling frequency ('daily', 'weekly', 'monthly',
            'quarterly') used to pick calendar-correct default periods when
            ``fourier_periods`` is None. Defaults to daily.
        fourier_min_period: Shortest learnable period in steps (Nyquist: 2).
        fourier_max_period: Longest learnable period in steps. None derives it
            from the initial periods.
        horizon: Forecast horizon length. ``1`` keeps the classic 1-step head;
            ``H>1`` emits ``[B, H]`` gated outputs (direct multi-horizon).
        use_sku: If False, disable SKU embedding personalization (shift-scale,
            SKU-conditioned component attention, intermittent SKU signal).
            Long-format shared trunk remains; useful as a no-ID-pooling ablation.
        component_attention_temperature: Softmax temperature for component weights.
        component_entropy_weight: Sparsity loss weight for component attention.
        component_attention_l2: L2 weight for component-attention logits.
        intermittent_hidden_dim: Hidden width of the zero-demand classifier.
        intermittent_entropy_weight: Confidence regularizer for zero probability.
        orthogonality_weight: Off-diagonal covariance penalty for active components.
        intermittent_prior_zero_rate: Optional panel-level gate prior (Dense bias).
            Used when per-SKU rates are not provided; also seeds rate-match target.
        intermittent_prior_zero_rates: Optional length-``n_skus`` zero rates for a
            non-trainable SKU Embedding prior on gate logits. Prefers this over
            the panel bias when ``use_sku=True``. Sparse/unseen SKUs should use
            panel-mean fallback via ``estimate_zero_rate_by_sku``.
        trend_monotonic: If True (default), trend hinge slopes use softplus
            magnitude with a learned sign so the trend expert is monotone in
            the time feature. Set False for the legacy unconstrained attention path.
        holiday_monotonic: If True (default), holiday hinge slopes use softplus
            magnitude with a learned per-holiday sign on ``|days_from_*|``,
            then selection attention over holiday channels. Set False for the
            legacy unconstrained CP-attention path.
        regressor_monotonic: If True (default), each lag/regressor channel uses
            softplus magnitude with a learned per-channel sign, then lag
            attention over those mono scalars. Set False for unconstrained
            masked-attention on raw lag features.
        level1_selection_attention: If True (default), holiday/regressor mono
            paths use learned intra-expert selection attention. Set False for
            uniform ``1/n`` channel weights (novelty ablation).
        context_aware_component_mixer: If True (default), component attention
            also sees a projected lag/intermittent context vector so the same
            SKU can reweight experts by demand regime. Set False for the legacy
            SKU-only (or stacked-expert) mixer used by paper-protocol ablations.
        component_mixer_context_dim: Optional width of the context Dense. None
            uses ``max(4, hidden_dim // 4)``.
        context_film_seasonal_holiday: If True, after seasonal and holiday
            expert scalars (post softsign / SKU FiLM), apply a lag-derived FiLM
            (softplus scale near 1 at init + softsign shift). Trend and
            regressor are unchanged. Default False (preferred softsign + mono
            + mixer stack); set True to enable calendar FiLM.
        component_combine: How Level-2 mixes expert scalars after attention.
            ``'additive'`` (default, locked bake-off) is ``Σ α_k e_k``.
            ``'multiplicative'`` is Prophet-like
            ``softplus(e_T) Π max(ε, 1 + α_k e_k)`` over seasonal/holiday/
            regressor (see ``MultiplicativeComponentCombine``). Does not change
            L1 / FiLM / gate; cross-layers stay off by default.
        multiplicative_eps: Positive floor for multiplicative factors
            (default ``1e-3``). Ignored when ``component_combine='additive'``.
    
    Returns:
        model: Keras Model
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    use_sku = bool(use_sku)
    level1_selection_attention = bool(level1_selection_attention)
    context_aware_component_mixer = bool(context_aware_component_mixer)
    context_film_seasonal_holiday = bool(context_film_seasonal_holiday)
    component_combine = str(component_combine).lower().strip()
    if component_combine not in ('additive', 'multiplicative'):
        raise ValueError(
            "component_combine must be 'additive' or 'multiplicative', "
            f"got {component_combine!r}"
        )
    multiplicative_eps = float(multiplicative_eps)
    if multiplicative_eps <= 0.0 or multiplicative_eps >= 1.0:
        raise ValueError(
            "multiplicative_eps must be in (0, 1), "
            f"got {multiplicative_eps}"
        )
    if fourier_periods is None and fourier_frequency is not None:
        fourier_periods = fourier_periods_for_frequency(fourier_frequency)
    (
        enable_trend,
        enable_seasonal,
        enable_holiday,
        enable_regressor,
    ) = _resolve_component_flags(
        n_temporal_features,
        n_fourier_features,
        n_holiday_features,
        n_lag_features,
        enable_trend,
        enable_seasonal,
        enable_holiday,
        enable_regressor,
        use_learnable_fourier,
    )

    if use_learnable_fourier and n_fourier_features != 1:
        raise ValueError(
            "use_learnable_fourier=True expects n_fourier_features=1 "
            "(raw time in days). Got n_fourier_features="
            f"{n_fourier_features}."
        )

    # Inputs
    temporal_input = Input(shape=(n_temporal_features,), name='temporal_features')
    fourier_input = Input(shape=(n_fourier_features,), name='fourier_features')
    # Keras rejects Input(shape=(0,)); disabled holidays use a 1-d zero dummy.
    holiday_input_dim = int(n_holiday_features) if (n_holiday_features and n_holiday_features > 0) else 1
    holiday_input = Input(shape=(holiday_input_dim,), name='holiday_features')
    lag_input = Input(shape=(n_lag_features,), name='lag_features')
    sku_input = Input(shape=(1,), name='sku_id', dtype=tf.int32)
    
    # Input-level normalization before component splitting
    # Temporal: NO normalization (changepoints need raw time values)
    # Fourier: NO normalization (sin/cos already bounded [-1, 1])
    # Holiday & Lag: NO normalization (MaskedEntropyAttention handles it internally)
    
    sku_embedding = _build_sku_path(
        sku_input, n_skus, sku_embedding_dim, use_sku
    )

    use_temporal_context = bool(use_temporal_context)
    if decouple_gate is None:
        decouple_gate = use_temporal_context
    else:
        decouple_gate = bool(decouple_gate)
    if use_temporal_context and decouple_gate is False:
        # Explicit paper-style coupled gate with temporal is allowed but rare.
        pass
    sequence_input = None
    temporal_context = None
    if use_temporal_context:
        if n_sequence_channels is None or int(n_sequence_channels) < 1:
            raise ValueError(
                "use_temporal_context=True requires n_sequence_channels >= 1 "
                "(Quantity + causal feature channels, same as TST windows)."
            )
        lookback = int(lookback)
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        sequence_input = Input(
            shape=(lookback, int(n_sequence_channels)),
            name='sequence_history',
        )
        temporal_context = _build_causal_lookback_encoder(
            sequence_input,
            sku_embedding,
            lookback=lookback,
            d_model=int(temporal_d_model),
            n_heads=int(temporal_n_heads),
            n_blocks=int(temporal_n_blocks),
            use_sku=use_sku,
            name='temporal_encoder',
        )

    component_outputs, presence, component_flags = _build_components(
        temporal_input,
        fourier_input,
        holiday_input,
        lag_input,
        sku_embedding,
        n_temporal_features=n_temporal_features,
        n_fourier_features=n_fourier_features,
        n_holiday_features=n_holiday_features,
        n_lag_features=n_lag_features,
        n_changepoints=n_changepoints,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        activation=activation,
        output_activation=output_activation,
        use_sku=use_sku,
        use_learnable_fourier=use_learnable_fourier,
        n_learnable_frequencies=n_learnable_frequencies,
        fourier_periods=fourier_periods,
        fourier_min_period=fourier_min_period,
        fourier_max_period=fourier_max_period,
        enable_trend=enable_trend,
        enable_seasonal=enable_seasonal,
        enable_holiday=enable_holiday,
        enable_regressor=enable_regressor,
        trend_monotonic=trend_monotonic,
        holiday_monotonic=holiday_monotonic,
        regressor_monotonic=regressor_monotonic,
        level1_selection_attention=level1_selection_attention,
    )

    # Lag/intermittent block already on the graph; do not pull holiday/Fourier.
    mixer_context = (
        lag_input
        if (n_lag_features and int(n_lag_features) > 0)
        else None
    )
    # Post softsign/SKU FiLM: lag-conditioned FiLM on seasonal + holiday only.
    trend_c, seasonal_c, holiday_c, regressor_c = component_outputs
    seasonal_c, holiday_c = _apply_context_film_seasonal_holiday(
        seasonal_c,
        holiday_c,
        mixer_context,
        enabled=context_film_seasonal_holiday,
        hidden_dim=hidden_dim,
        context_dim=component_mixer_context_dim,
    )
    component_outputs = [trend_c, seasonal_c, holiday_c, regressor_c]

    logger.debug(
        "Component presence: trend=%s seasonal=%s holiday=%s regressor=%s; "
        "cross=%s intermittent=%s horizon=%d sku=%s context_film=%s "
        "component_combine=%s",
        presence['trend'],
        presence['seasonal'],
        presence['holiday'],
        presence['regressor'],
        use_cross_layers,
        use_intermittent,
        horizon,
        use_sku,
        context_film_seasonal_holiday,
        component_combine,
    )
    if use_learnable_fourier:
        periods = (
            fourier_periods
            if fourier_periods is not None
            else DEFAULT_FOURIER_PERIODS
        )
        logger.debug(
            "Learnable Fourier enabled: frequencies=%d initial_periods=%s "
            "(steps, frequency=%s)",
            n_learnable_frequencies,
            pad_fourier_periods(
                periods, n_learnable_frequencies, min_period=fourier_min_period
            ),
            fourier_frequency or 'daily',
        )

    component_outputs = _add_orthogonality_regularization(
        component_outputs,
        component_flags,
        orthogonality_weight,
    )
    attention_source = _build_component_mixer_source(
        sku_embedding,
        mixer_context,
        use_sku=use_sku,
        context_aware_component_mixer=context_aware_component_mixer,
        hidden_dim=hidden_dim,
        context_dim=component_mixer_context_dim,
    )
    base_forecast_attention = _build_component_attention(
        component_outputs,
        attention_source,
        hidden_dim,
        component_attention_temperature,
        component_entropy_weight,
        component_attention_l2,
        component_flags=component_flags,
        component_combine=component_combine,
        multiplicative_eps=multiplicative_eps,
    )
    base_forecast = _combine_component_forecasts(
        base_forecast_attention,
        component_outputs,
        use_cross_layers,
    )
    # When decoupled, always feed raw regressors into the gate.
    gate_raw = bool(gate_use_raw_regressors) or bool(decouple_gate)
    outputs = _build_intermittent_heads(
        component_outputs,
        base_forecast,
        sku_embedding,
        use_intermittent=use_intermittent,
        use_cross_layers=use_cross_layers,
        use_sku=use_sku,
        horizon=horizon,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        intermittent_hidden_dim=intermittent_hidden_dim,
        intermittent_entropy_weight=intermittent_entropy_weight,
        lag_features=lag_input,
        gate_use_raw_regressors=gate_raw,
        intermittent_prior_zero_rate=intermittent_prior_zero_rate,
        intermittent_prior_zero_rates=intermittent_prior_zero_rates,
        n_skus=n_skus,
        sku_input=sku_input,
        intermittent_gate_temperature=intermittent_gate_temperature,
        intermittent_learnable_temperature=intermittent_learnable_temperature,
        intermittent_learnable_logit_scale=intermittent_learnable_logit_scale,
        intermittent_logit_scale_init=intermittent_logit_scale_init,
        gate_prob_scale=gate_prob_scale,
        gate_prob_scale_init=gate_prob_scale_init,
        gate_prob_scale_trainable=gate_prob_scale_trainable,
        gate_rate_match_weight=gate_rate_match_weight,
        gate_rate_match_target=gate_rate_match_target,
        gate_raw_regressor_proj_dim=gate_raw_regressor_proj_dim,
        decouple_gate=decouple_gate,
        temporal_context=temporal_context,
    )

    inputs = [temporal_input, fourier_input, holiday_input, lag_input, sku_input]
    if sequence_input is not None:
        inputs.append(sequence_input)
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=(
            'hierarchical_attention_hybrid'
            if use_temporal_context
            else 'hierarchical_attention_lightweight'
        ),
    )

    return model


COMPONENT_EXPERT_NAMES = ("trend", "seasonal", "holiday", "regressor")


def build_component_readout_model(model):
    r"""Wrap a trained lightweight model to expose interpretable components.

    Returns a Keras ``Model`` with the same inputs and a **dict** of tensors:

    | Key | Meaning |
    |-----|---------|
    | ``trend``, ``seasonal``, ``holiday``, ``regressor`` | Expert scalars **after** Level-1 selection / softsign / SKU FiLM (and optional calendar FiLM on seasonal+holiday)—the values mixed by Level-2. |
    | ``component_alpha`` | Level-2 softmax weights \(\alpha_k\) over the four experts. |
    | ``base_forecast`` | Magnitude head \(b\) (softplus). |
    | ``non_zero_probability`` | Occurrence gate \(p\). |
    | ``final_forecast`` | \(\hat{y}=p\cdot b\). |

    Call :func:`predict_with_components` to also get ``mixed_contribution_*``
    (\(\alpha_k\cdot e_k\)) as NumPy arrays. Training API is unchanged: the
    original ``model`` still outputs only gate heads.

    **Multi-horizon note.** Recursive MH rollouts call the one-step model
    repeatedly; probe at **each** step to record components. Direct MH heads
    (``horizon>1``) expose ``p``/``b``/``yhat`` shaped ``[B, H]``, but expert
    scalars remain one-step (shared base)—document that when dumping MH tables.
    """
    layer_names = {layer.name for layer in model.layers}
    required = set(COMPONENT_EXPERT_NAMES) | {"component_attention_softmax"}
    missing = sorted(required - layer_names)
    if missing:
        raise ValueError(
            f"Model is missing component layers {missing}; "
            "build_component_readout_model expects the lightweight hierarchy."
        )

    outs = {
        name: model.get_layer(name).output for name in COMPONENT_EXPERT_NAMES
    }
    outs["component_alpha"] = model.get_layer("component_attention_softmax").output

    if isinstance(model.output, dict):
        outs["final_forecast"] = model.output["final_forecast"]
        if "base_forecast" in model.output:
            outs["base_forecast"] = model.output["base_forecast"]
        elif "base_forecast" in layer_names:
            outs["base_forecast"] = model.get_layer("base_forecast").output
        else:
            outs["base_forecast"] = model.output["final_forecast"]
        if "non_zero_probability" in model.output:
            outs["non_zero_probability"] = model.output["non_zero_probability"]
    else:
        outs["final_forecast"] = model.output
        outs["base_forecast"] = model.output

    return Model(model.inputs, outs, name="component_readout")


def predict_with_components(model, x, *, batch_size=1024, verbose=0):
    r"""Run component readout → numpy dict including ``mixed_contribution_*``.

    Keys: expert scalars, ``component_alpha``, ``base_forecast`` (\(b\)),
    ``non_zero_probability`` (\(p\)), ``final_forecast`` (\(\hat y=p\cdot b\)),
    and ``mixed_contribution_{trend,seasonal,holiday,regressor}`` =
    \(\alpha_k \cdot e_k\).
    """
    probe = build_component_readout_model(model)
    raw = probe.predict(x, batch_size=batch_size, verbose=verbose)
    if isinstance(raw, dict):
        out = {k: np.asarray(v) for k, v in raw.items()}
    else:
        names = list(probe.output_names)
        out = {n: np.asarray(v) for n, v in zip(names, raw)}
    alpha = np.asarray(out["component_alpha"], dtype=np.float64)
    for i, name in enumerate(COMPONENT_EXPERT_NAMES):
        e = np.asarray(out[name], dtype=np.float64).reshape(-1, 1)
        a = alpha[:, i : i + 1]
        out[f"mixed_contribution_{name}"] = (a * e).astype(np.float32)
    return out


def create_model_from_features(
    X_train,
    sku_train,
    feature_indices,
    n_skus,
    hidden_dim=64,
    sku_embedding_dim=8,
    dropout_rate=0.3,
    use_cross_layers=False,
    use_intermittent=True,
    learning_rate=0.001,
    loss_weights=None,
    zero_rate=None,
    y_train=None,
    use_sku=True,
    use_sku_gate_prior=True,
    pos_weight=None,
):
    """
    Wrapper function that creates and compiles a model ready for training.
    Takes 2 inputs (features + SKU) and handles the internal splitting.
    
    Args:
        X_train: Training features array [n_samples, n_features]
        sku_train: SKU IDs array [n_samples]
        feature_indices: Dict with keys 'trend', 'seasonal', 'holiday', 'regressor'
                        Each containing list of column indices
        n_skus: Number of unique SKUs
        hidden_dim: Hidden dimension for components
        sku_embedding_dim: SKU embedding dimension
        dropout_rate: Dropout rate
        use_cross_layers: Opt-in DCN cross on component outputs (default False)
        use_intermittent: Whether to use intermittent handling
        learning_rate: Learning rate for Adam optimizer
        loss_weights: Optional override for output loss weights. If None, uses
                     data-driven weights from composite_loss().
        zero_rate: Optional known panel zero rate. Estimated from ``y_train``
            when omitted. **Required** (directly or via ``y_train``) — there is
            no silent 0.9 default.
        y_train: Optional targets used to estimate zero_rate / avg non-zero demand
            and (with ``sku_train``) per-SKU rates for the gate prior.
        use_sku: Enable SKU embedding personalization (default True).
        use_sku_gate_prior: When True and ``y_train`` + ``sku_train`` are given,
            wire per-SKU zero-rate priors into the intermittent gate. Sparse /
            unseen SKUs fall back to the panel mean via
            ``estimate_zero_rate_by_sku``.
        pos_weight: Optional BCE positive-class weight. Defaults to
            ``pos_weight_from_zero_rate(panel_zero_rate)``. This remains a
            **panel-level** scalar compiled into BCE. When ``y_train`` +
            ``sku_train`` are available, the model also exposes
            ``sku_zero_rates`` and ``make_fit_sample_weights(y, sku)`` —
            pass the returned dict as ``sample_weight`` on ``fit`` so
            imbalance is per-SKU (weights are relative to the panel
            ``pos_weight`` to avoid double-counting).

    Returns:
        model: Compiled Keras model ready for training
        split_fn: Function to split input features for model.fit()
    """
    from tensorflow import keras
    
    # Get feature counts
    n_temporal = len(feature_indices['trend'])
    n_fourier = len(feature_indices['seasonal'])
    n_holiday = len(feature_indices['holiday'])
    n_lag = len(feature_indices['regressor'])

    sku_rate_info = None
    if y_train is not None and sku_train is not None and use_sku:
        sku_rate_info = estimate_zero_rate_by_sku(
            y_train, sku_train, n_skus=n_skus
        )

    # Fail fast: never silently hardcode 0.9.
    if zero_rate is None:
        if y_train is None:
            raise ValueError(
                "zero_rate is required when y_train is not provided. "
                "Pass zero_rate=... or y_train=... so the intermittent loss "
                "can be calibrated (no default of 0.9)."
            )
        if sku_rate_info is not None:
            zero_rate_value = float(sku_rate_info["panel_mean"])
        else:
            zeros = np.sum(np.isclose(np.asarray(y_train).reshape(-1), 0.0))
            total = int(np.asarray(y_train).reshape(-1).shape[0])
            zero_rate_value = float(zeros) / float(max(total, 1))
        logger.info("Estimated zero_rate from y_train: %.4f", zero_rate_value)
    else:
        zero_rate_value = float(zero_rate)

    sku_prior_rates = None
    if (
        use_sku
        and use_sku_gate_prior
        and sku_rate_info is not None
        and use_intermittent
    ):
        sku_prior_rates = sku_rate_info["rates"]
        logger.info(
            "Using per-SKU gate priors (panel_mean=%.4f, n_skus=%d)",
            sku_rate_info["panel_mean"],
            sku_rate_info["n_skus"],
        )

    # Build the model
    model = build_hierarchical_model_lightweight(
        n_temporal_features=n_temporal,
        n_fourier_features=n_fourier,
        n_holiday_features=n_holiday,
        n_lag_features=n_lag,
        n_skus=n_skus,
        hidden_dim=hidden_dim,
        sku_embedding_dim=sku_embedding_dim,
        dropout_rate=dropout_rate,
        use_cross_layers=use_cross_layers,
        use_intermittent=use_intermittent,
        use_sku=use_sku,
        intermittent_prior_zero_rate=zero_rate_value,
        intermittent_prior_zero_rates=sku_prior_rates,
    )
    
    # Define split function for features
    def split_features(X, sku):
        """Split features into component inputs"""
        X_temporal = X[:, feature_indices['trend']]
        X_fourier = X[:, feature_indices['seasonal']]
        X_holiday = X[:, feature_indices['holiday']]
        X_lag = X[:, feature_indices['regressor']]
        return [X_temporal, X_fourier, X_holiday, X_lag, sku]
    
    from .losses import composite_loss

    avg_nonzero = None
    if y_train is not None:
        y_flat = np.asarray(y_train).reshape(-1)
        nonzero = y_flat[~np.isclose(y_flat, 0.0)]
        if len(nonzero) > 0:
            avg_nonzero = float(np.mean(nonzero))

    if pos_weight is None:
        pos_weight = pos_weight_from_zero_rate(zero_rate_value)
    # Global pos_weight stays panel-level; per-SKU imbalance via sample_weight.
    loss_config = composite_loss(
        zero_rate=zero_rate_value,
        average_nonzero_demand=avg_nonzero,
        pos_weight=float(pos_weight),
    )
    compile_losses = dict(loss_config['losses'])
    compile_weights = dict(loss_config['weights']) if loss_weights is None else loss_weights

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=compile_losses,
        loss_weights=compile_weights,
        metrics={
            'final_forecast': ['mae'],
            'base_forecast': ['mae'],
            'non_zero_probability': ['accuracy', 'binary_crossentropy']
        }
    )

    # Attach per-SKU rates + fit helper when available (backward-compat API).
    model.panel_zero_rate = float(zero_rate_value)
    model.sku_zero_rates = (
        None if sku_rate_info is None else sku_rate_info["rates"]
    )
    model.compiled_pos_weight = float(pos_weight)
    if sku_rate_info is not None:
        _rates = sku_rate_info["rates"]
        _panel = float(zero_rate_value)

        def make_fit_sample_weights(y, sku, **kwargs):
            """Relative per-SKU BCE sample weights for ``model.fit``."""
            return multioutput_bce_sample_weight_dict(
                y,
                sku,
                _rates,
                reference_zero_rate=kwargs.pop("reference_zero_rate", _panel),
                other_keys=("final_forecast",),
                **kwargs,
            )

        model.make_fit_sample_weights = make_fit_sample_weights
        logger.info(
            "Per-SKU BCE sample weights available via "
            "model.make_fit_sample_weights(y, sku) "
            "(relative to panel pos_weight=%.3f)",
            float(pos_weight),
        )
    
    return model, split_features


# Example usage and parameter comparison
def compare_model_sizes(
    n_temporal=10,
    n_fourier=10,
    n_holiday=15,
    n_lag=3,
    n_skus=6099,
    hidden_dim=32,
    use_cross_layers=False,
    use_intermittent=True,
    tabnet_params=322359
):
    """
    Compare parameter counts: TabNet vs Lightweight
    
    Args:
        n_temporal: Number of temporal features
        n_fourier: Number of Fourier features
        n_holiday: Number of holiday features
        n_lag: Number of lag features
        n_skus: Number of SKUs
        hidden_dim: Hidden dimension for components
        use_cross_layers: Whether to use cross-layers
        use_intermittent: Whether to use intermittent handling
        tabnet_params: Reference TabNet parameter count for comparison
    """
    print("=" * 80)
    print("PARAMETER COMPARISON: TabNet vs Lightweight Masked Attention")
    print("=" * 80)
    
    print("\nConfiguration:")
    print(f"  Temporal features: {n_temporal}")
    print(f"  Fourier features: {n_fourier}")
    print(f"  Holiday features: {n_holiday}")
    print(f"  Lag features: {n_lag}")
    print(f"  Number of SKUs: {n_skus}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Cross-layers: {use_cross_layers}")
    print(f"  Intermittent handling: {use_intermittent}")
    
    # Build lightweight model
    model_light = build_hierarchical_model_lightweight(
        n_temporal_features=n_temporal,
        n_fourier_features=n_fourier,
        n_holiday_features=n_holiday,
        n_lag_features=n_lag,
        n_skus=n_skus,
        hidden_dim=hidden_dim,
        use_cross_layers=use_cross_layers,
        use_intermittent=use_intermittent
    )
    
    total_params = model_light.count_params()
    trainable_params = sum([
        tf.size(w).numpy() for w in model_light.trainable_weights
    ])
    
    print("\n✓ Lightweight Model (Masked Entropy Attention):")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n✓ TabNet Model (reference):")
    print(f"  Total parameters: {tabnet_params:,}")
    print(f"  Trainable parameters: {tabnet_params:,}")
    
    print("\n✓ Parameter Reduction:")
    reduction = ((tabnet_params - total_params) / tabnet_params) * 100
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Lightweight params: {total_params/tabnet_params:.1%} of TabNet")
    
    return {
        'lightweight_params': total_params,
        'tabnet_params': tabnet_params,
        'reduction_percent': reduction
    }
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES:")
    print("=" * 80)
    print("TabNet:")
    print("  - Sequential attention blocks (multiple steps)")
    print("  - Feature transformer + attentive transformer")
    print("  - Ghost batch normalization")
    print("  - ~20K params per component × 4 components = ~80K")
    print("  - Heavy but powerful")
    
    print("\nLightweight Masked Attention:")
    print("  - Single attention layer per component")
    print("  - Entropy regularization for sparsity")
    print("  - Simple dense projections")
    print("  - ~2K params per component × 4 components = ~8K")
    print("  - Fast and interpretable")
    
    print("\n" + "=" * 80)
    
    return model_light


if __name__ == "__main__":
    # Show model comparison
    model = compare_model_sizes()
    
    # Show architecture
    print("\nModel Architecture:")
    print(model.summary())
