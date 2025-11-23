"""
Compare LightGBM vs Two-Stage Intermittent Handler (Optimized DeepSequence)

Tests both models on the same sparse/intermittent demand dataset.

ARCHITECTURE CHANGES (from best/ baseline):
- Regressor component: REMOVED (set to None)
- Weight initializers: he_normal (mish activation), glorot_normal (linear/sigmoid)
- Transformer FFN dimension: Matches DeepSequence output (3 components * 32 = 96)
- DeepSequence components: Trend, Seasonal, Holiday (3 components only)

DEFAULT CONFIGURATION (Optimized for Intermittent Demand):
- Loss: Composite (BCE + weighted MAE with SKU-aware log1p weights)
  * alpha=0.2 balances zero detection (BCE) with magnitude (MAE)
  * SKU weights = log1p(mean_demand) prioritize high-volume SKUs
- Activation: Mish (x * tanh(softplus(x)))
  * Smooth, unbounded, superior gradient flow vs sparse_amplify
- Baseline Results: 74.7% per-SKU win rate, 82.2% non-zero recall, 23% aggregate MAE improvement
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import tensorflow as tf
from tf_keras.layers import Layer, LayerNormalization, MultiHeadAttention

# Check Metal GPU availability
print("\n" + "="*80)
print("GPU/METAL CONFIGURATION")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Available devices: {[d.name for d in tf.config.list_physical_devices()]}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU devices found: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu.name}")
else:
    print("⚠️  WARNING: No GPU devices found! Running on CPU (SLOW)")
    print("   Make sure tensorflow-metal is installed: pip install tensorflow-metal")
print("="*80 + "\n")

# XLA JIT is NOT compatible with Metal GPU - causes platform errors
# tf.config.optimizer.set_jit(True)
print("✓ XLA JIT disabled (not compatible with Metal)")

# Disable mixed precision to avoid dtype conflicts with PWLCalibration
from tf_keras import mixed_precision
mixed_precision.set_global_policy('float32')
print("✓ Using float32 policy (required for PWLCalibration)\n")

# Custom activations for TF 2.17 compatibility
@tf.function
def mish(x):
    """Mish activation: x * tanh(softplus(x))"""
    return x * tf.math.tanh(tf.math.softplus(x))

@tf.function
def sparse_amplify(x):
    """Sparse amplify: x * 1/(abs(x)+1) - designed for sparse data"""
    return x * (1.0 / (tf.abs(x) + 1.0))

# Global SKU weight lookup (will be populated after computing statistics)
sku_weight_lookup = None


def create_composite_loss_with_sku_weights(sku_weights_dict, alpha=0.2):
    """
    Create composite loss function with SKU-aware weighting:
    
    1. Binary cross-entropy: Predict if y_true > 0 (no weighting)
    2. Weighted MAE: Magnitude prediction (weight = log1p(sku_mean))
    
    This explicitly models intermittent demand (zero vs non-zero) while
    weighting errors by SKU demand level.
    
    Args:
        sku_weights_dict: Dict mapping sku_id -> log1p(mean_demand)
        alpha: Balance coefficient for BCE relative to MAE (default 0.2)
        
    Returns:
        Composite loss function that takes (y_true, y_pred, sample_weight)
    """
    # Convert to numpy array for fast lookup
    max_sku = max(sku_weights_dict.keys()) + 1
    sku_weights_array = np.ones(max_sku, dtype=np.float32)
    for sku_id, weight in sku_weights_dict.items():
        sku_weights_array[sku_id] = weight
    
    def composite_loss(y_true, y_pred, sample_weight=None):
        """
        Composite loss with SKU-aware weighting.
        
        sample_weight contains SKU IDs (passed as weights for lookup)
        """
        # Binary target: 1 if y_true > 0, else 0
        y_binary = tf.cast(y_true > 0, tf.float32)
        
        # Predict binary class from magnitude
        # Clip predictions to avoid log(0) in BCE
        y_pred_clipped = tf.clip_by_value(y_pred, 0.0, 1e6)
        y_pred_binary = tf.nn.sigmoid(y_pred_clipped / 10.0)  # Scale down for sigmoid
        
        # Binary cross-entropy loss (no SKU weighting)
        bce_loss = tf.keras.losses.binary_crossentropy(y_binary, y_pred_binary)
        
        # MAE loss with SKU weighting
        mae_per_sample = tf.abs(y_true - y_pred)
        
        # Apply SKU weights if sample_weight provided
        if sample_weight is not None:
            # sample_weight contains the actual weight values (not SKU IDs)
            weighted_mae = mae_per_sample * sample_weight
        else:
            weighted_mae = mae_per_sample
        
        # Combined loss: alpha * BCE + weighted MAE
        # alpha scales BCE to be comparable to MAE
        combined = alpha * bce_loss + weighted_mae
        
        return tf.reduce_mean(combined)
    
    return composite_loss

# Tweedie loss for zero-inflated count data
@tf.function
def tweedie_loss(y_true, y_pred, p=1.5):
    """
    Tweedie loss for compound Poisson-Gamma distribution.
    
    p = 1.5 is typical for zero-inflated count data:
    - p=1: Poisson (count data)
    - p=2: Gamma (continuous positive)
    - 1 < p < 2: Compound Poisson-Gamma (zero-inflated)
    
    Formula: -y * (y_pred^(1-p)) / (1-p) + (y_pred^(2-p)) / (2-p)
    """
    # Clip predictions to avoid numerical issues
    y_pred = tf.maximum(y_pred, 1e-6)
    
    # Tweedie deviance
    a = -y_true * tf.pow(y_pred, 1.0 - p) / (1.0 - p)
    b = tf.pow(y_pred, 2.0 - p) / (2.0 - p)
    
    return tf.reduce_mean(a + b)

print("="*80)
print("LIGHTGBM vs TWO-STAGE INTERMITTENT HANDLER COMPARISON")
print("="*80)

# Load fixed splits
print("\n[1/6] Loading data...")
train = pd.read_csv('data/train_split.csv')
val = pd.read_csv('data/val_split.csv')
test = pd.read_csv('data/test_split.csv')

train['ds'] = pd.to_datetime(train['ds'])
val['ds'] = pd.to_datetime(val['ds'])
test['ds'] = pd.to_datetime(test['ds'])

print(f"Train: {len(train):,} records")
print(f"Val:   {len(val):,} records")
print(f"Test:  {len(test):,} records")

# Check sparsity
train_zeros = (train['Quantity'] == 0).sum()
val_zeros = (val['Quantity'] == 0).sum()
test_zeros = (test['Quantity'] == 0).sum()

print(f"\nSparsity (zeros):")
print(f"Train: {100*train_zeros/len(train):.1f}%")
print(f"Val:   {100*val_zeros/len(val):.1f}%")
print(f"Test:  {100*test_zeros/len(test):.1f}%")

# Use larger sample for evaluation (full dataset takes too long)
SAMPLE_SIZE = 500000  # Balanced: large enough for accuracy, fast enough to complete
if len(train) > SAMPLE_SIZE:
    print(f"\n⚠ Using {SAMPLE_SIZE:,} training samples (full dataset would take hours)")
    train = train.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    val = val.sample(n=min(100000, len(val)), random_state=42).reset_index(drop=True)
    test = test.sample(n=min(100000, len(test)), random_state=42).reset_index(drop=True)
else:
    print(f"\n✓ Using full dataset")

# Create features
print("\n[2/6] Creating features...")

# Define major US holidays for distance calculation
from pandas.tseries.holiday import Holiday, AbstractHolidayCalendar, Easter
from dateutil.relativedelta import MO, SU, TH


class ExtendedUSHolidayCalendar(AbstractHolidayCalendar):
    """
    Extended US Holiday calendar including retail/commercial events.
    Calculates distance from these reference dates.
    """
    rules = [
        Holiday("NewYearsDay", month=1, day=1),
        Holiday("MLKDay", month=1, day=1,
                offset=pd.DateOffset(weekday=MO(3))),
        Holiday("PresidentsDay", month=2, day=1,
                offset=pd.DateOffset(weekday=MO(3))),
        Holiday("ValentinesDay", month=2, day=14),
        Holiday("Easter", month=1, day=1, offset=[Easter()]),
        Holiday("MothersDay", month=5, day=1,
                offset=pd.DateOffset(weekday=SU(2))),
        Holiday("MemorialDay", month=5, day=31,
                offset=pd.DateOffset(weekday=MO(-1))),
        Holiday("FathersDay", month=6, day=1,
                offset=pd.DateOffset(weekday=SU(3))),
        Holiday("IndependenceDay", month=7, day=4),
        Holiday("LaborDay", month=9, day=1,
                offset=pd.DateOffset(weekday=MO(1))),
        Holiday("Halloween", month=10, day=31),
        Holiday("Thanksgiving", month=11, day=1,
                offset=pd.DateOffset(weekday=TH(4))),
        Holiday("BlackFriday", month=11, day=1,
                offset=[pd.DateOffset(weekday=TH(4)),
                        pd.DateOffset(days=1)]),
        Holiday("Christmas", month=12, day=25),
        Holiday("NewYearsEve", month=12, day=31),
    ]


def get_nearest_holiday_distance(dates):
    """
    Calculate days from nearest holiday using holiday calendar.
    Returns signed distance: negative = before holiday, positive = after.
    """
    cal = ExtendedUSHolidayCalendar()
    # Get holidays for the full date range with buffer
    start_date = dates.min() - pd.Timedelta(days=90)
    end_date = dates.max() + pd.Timedelta(days=90)
    holidays = cal.holidays(start=start_date, end=end_date)

    distances = []
    for date in dates:
        # Calculate days to each holiday
        days_to_holidays = (holidays - date).days.values

        if len(days_to_holidays) > 0:
            # Find nearest holiday by absolute distance
            abs_distances = np.abs(days_to_holidays)
            nearest_idx = abs_distances.argmin()
            nearest_distance = days_to_holidays[nearest_idx]
            distances.append(nearest_distance)
        else:
            # No holidays found - default to far distance
            distances.append(90)

    return np.array(distances)

for df in [train, val, test]:
    df['week_of_year'] = df['ds'].dt.isocalendar().week
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['day_of_week'] = df['ds'].dt.dayofweek
    
    # Holiday distance feature (replaces binary is_holiday)
    df['holiday_distance'] = get_nearest_holiday_distance(df['ds'])
    
    # Fourier features for seasonality (52 weeks/year, 7 days/week)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Lag features
    for lag in [1, 2, 4]:
        df[f'lag_{lag}'] = df.groupby('id_var')['Quantity'].shift(lag).fillna(0)
    
    # Rolling mean
    df['rolling_mean_4'] = df.groupby('id_var')['Quantity'].transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    )

print("✓ Features created (including holiday distance and Fourier components)")

# Encode SKU IDs
sku_to_id = {sku: idx for idx, sku in enumerate(train['id_var'].unique())}
train['sku_id'] = train['id_var'].map(sku_to_id).fillna(0).astype(int)
val['sku_id'] = val['id_var'].map(sku_to_id).fillna(0).astype(int)
test['sku_id'] = test['id_var'].map(sku_to_id).fillna(0).astype(int)

# ============================================================================
# LIGHTGBM BASELINE
# ============================================================================
print("\n" + "="*80)
print("LIGHTGBM BASELINE")
print("="*80)

lgb_features = ['sku_id', 'week_of_year', 'month', 'year', 'day_of_week',
                'lag_1', 'lag_2', 'lag_4', 'rolling_mean_4',
                'week_sin', 'week_cos', 'day_sin', 'day_cos',
                'holiday_distance']

X_train_lgb = train[lgb_features].fillna(0)
y_train_lgb = train['Quantity']
X_val_lgb = val[lgb_features].fillna(0)
y_val_lgb = val['Quantity']
X_test_lgb = test[lgb_features].fillna(0)
y_test_lgb = test['Quantity']

print("\n[3/6] Training LightGBM...")
lgb_start = time.time()

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'verbose': -1
}

lgb_train = lgb.Dataset(X_train_lgb, y_train_lgb)
lgb_valid = lgb.Dataset(X_val_lgb, y_val_lgb)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)

lgb_time = time.time() - lgb_start

# Predictions
lgb_train_pred = lgb_model.predict(X_train_lgb)
lgb_val_pred = lgb_model.predict(X_val_lgb)
lgb_test_pred = lgb_model.predict(X_test_lgb)

# Metrics
lgb_train_mae = mean_absolute_error(y_train_lgb, lgb_train_pred)
lgb_val_mae = mean_absolute_error(y_val_lgb, lgb_val_pred)
lgb_test_mae = mean_absolute_error(y_test_lgb, lgb_test_pred)

print(f"\n✓ LightGBM trained in {lgb_time:.1f}s")
print(f"  Train MAE: {lgb_train_mae:.4f}")
print(f"  Val MAE:   {lgb_val_mae:.4f}")
print(f"  Test MAE:  {lgb_test_mae:.4f}")

# ============================================================================
# TWO-STAGE INTERMITTENT HANDLER
# ============================================================================
print("\n" + "="*80)
print("TWO-STAGE INTERMITTENT HANDLER")
print("="*80)

try:
    from tf_keras.layers import Input, Dense
    from tf_keras.models import Model
    from tf_keras.optimizers import Adam
    from tf_keras.callbacks import EarlyStopping
    print("Using tf_keras (Keras 2)")
except ImportError:
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    print("Using tensorflow.keras")

print("\n[4/6] Preparing neural network features...")

# Simple features for neural network (include holiday_distance + Fourier)
nn_features = ['sku_id', 'week_of_year', 'month', 'day_of_week',
               'lag_1', 'lag_2', 'lag_4', 'rolling_mean_4',
               'week_sin', 'week_cos', 'day_sin', 'day_cos',
               'holiday_distance']

X_train_nn = train[nn_features].fillna(0).values.astype(np.float32)
X_val_nn = val[nn_features].fillna(0).values.astype(np.float32)
X_test_nn = test[nn_features].fillna(0).values.astype(np.float32)

# Extract SKU IDs for embedding (before normalization)
sku_train = X_train_nn[:, 0].astype(np.int32)
sku_val = X_val_nn[:, 0].astype(np.int32)
sku_test = X_test_nn[:, 0].astype(np.int32)
num_skus = int(train['sku_id'].max()) + 1

print(f"  Number of unique SKUs: {num_skus}")

# Normalize features (including sku_id for feature input)
feature_mean = X_train_nn.mean(axis=0)
feature_std = X_train_nn.std(axis=0) + 1e-8
X_train_nn = (X_train_nn - feature_mean) / feature_std
X_val_nn = (X_val_nn - feature_mean) / feature_std
X_test_nn = (X_test_nn - feature_mean) / feature_std

# Prepare targets for two-stage model
y_train_nn = train['Quantity'].values.astype(np.float32)
y_val_nn = val['Quantity'].values.astype(np.float32)
y_test_nn = test['Quantity'].values.astype(np.float32)

# Zero probability targets
y_train_zero = (y_train_nn == 0).astype(np.float32)
y_val_zero = (y_val_nn == 0).astype(np.float32)

# Candidate forecast targets (non-zero only)
y_train_candidate = np.where(y_train_nn > 0, y_train_nn, 1.0)
y_val_candidate = np.where(y_val_nn > 0, y_val_nn, 1.0)

print(f"✓ Features prepared: {X_train_nn.shape}")

# Compute SKU-specific weights from training data
print("\nComputing SKU-specific weights (log1p of mean demand)...")
sku_weights = {}
for sku_id in range(num_skus):
    sku_mask = sku_train == sku_id
    if sku_mask.sum() > 0:
        sku_demand = y_train_nn[sku_mask]
        # Use mean demand across all samples (including zeros)
        sku_mean = float(sku_demand.mean())
        # Weight = log1p(mean) for logarithmic scaling
        sku_weights[sku_id] = float(np.log1p(sku_mean))
    else:
        sku_weights[sku_id] = 0.0  # log1p(0) = 0 for unseen SKUs

# Ensure minimum weight of 0.1 to avoid zero weights
sku_weights = {k: max(v, 0.1) for k, v in sku_weights.items()}

# Statistics
weight_values = list(sku_weights.values())
print(f"  SKU weights computed: {len(sku_weights)} SKUs")
print(f"  Weight range: [{min(weight_values):.4f}, {max(weight_values):.4f}]")
print(f"  Weight mean: {np.mean(weight_values):.4f}, median: {np.median(weight_values):.4f}")

# Create weight lookup tensor for efficient batch lookups
max_sku_id = max(sku_weights.keys())
sku_weight_array = np.ones(max_sku_id + 1, dtype=np.float32)
for sku_id, weight in sku_weights.items():
    sku_weight_array[sku_id] = weight
sku_weight_tensor = tf.constant(sku_weight_array, dtype=tf.float32)

# Create sample weights for training/validation
# These will be used in model.fit() to weight each sample by its SKU
train_sample_weights = np.array([sku_weights[int(sid)] for sid in sku_train])
val_sample_weights = np.array([sku_weights.get(int(sid), 1.0) for sid in sku_val])

print(f"  Train sample weights: mean={train_sample_weights.mean():.2f}, "
      f"std={train_sample_weights.std():.2f}")

print("\n[5/6] Building two-stage model...")

# Hyperparameters - tunable for optimization
ID_EMBEDDING_DIM = 16                # ID embedding dimension
COMPONENT_HIDDEN_UNITS = 32          # Component hidden layer size
COMPONENT_DROPOUT = 0.2              # Component dropout rate
TRANSFORMER_NUM_HEADS = 4            # Number of attention heads
TRANSFORMER_KEY_DIM = 32             # Attention key dimension
# Transformer FFN dimension = 3 components * COMPONENT_HIDDEN_UNITS (trend, seasonal, holiday)
TRANSFORMER_FFN_UNITS = 3 * COMPONENT_HIDDEN_UNITS  # 96 units to match DeepSequence output
TRANSFORMER_DROPOUT = 0.1            # Transformer dropout rate
ZERO_PROB_HIDDEN_UNITS = 128         # Zero probability network units
ZERO_PROB_HIDDEN_LAYERS = 3          # Zero probability network layers
ZERO_PROB_DROPOUT = 0.2              # Zero probability dropout rate
# Loss Configuration (Default: Composite Loss with SKU-aware weighting)
LOSS_TYPE = 'composite'              # 'composite' (BCE + weighted MAE) - BEST
LOSS_ALPHA = 0.2                     # Balance: alpha*BCE + MAE
                                     # Tuning guide:
                                     # - Lower (0.1): Focus on magnitude
                                     # - Higher (0.3): Focus on zero detection
                                     # - Default 0.2: Balanced performance
# Activation Function (Default: Mish)
ACTIVATION = 'mish'                  # 'mish' (x*tanh(softplus(x))) - BEST
INITIAL_LEARNING_RATE = 0.001        # Start with higher LR, will decay
BATCH_SIZE = 512                     # Adam: larger batch size
EARLY_STOPPING_PATIENCE = 20         # Early stopping patience

print(f"\n  Architecture configuration:")
print(f"    ID embedding dimension: {ID_EMBEDDING_DIM}")
print(f"    Component hidden units: {COMPONENT_HIDDEN_UNITS}")
print(f"    Transformer heads: {TRANSFORMER_NUM_HEADS}")
print(f"    Transformer key dim: {TRANSFORMER_KEY_DIM}")
print(f"    Transformer FFN units: {TRANSFORMER_FFN_UNITS}")
print(f"    Zero prob hidden units: {ZERO_PROB_HIDDEN_UNITS}")
print(f"  Training configuration:")
print(f"    Loss function: {LOSS_TYPE} (alpha={LOSS_ALPHA})")
print(f"    Activation: {ACTIVATION}")
print(f"    Initial learning rate: {INITIAL_LEARNING_RATE}")
print(f"    Batch size: {BATCH_SIZE} (Adam)")
print(f"    Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"    Zero prob hidden layers: {ZERO_PROB_HIDDEN_LAYERS}")

# Create mock component outputs (in practice, these would be real components)
n_features = X_train_nn.shape[1]

# Inputs
main_input = Input(shape=(n_features,), name='main_input')
sku_input = Input(shape=(1,), dtype='int32', name='sku_input')

# ID Embedding layer - shared across all components
from tensorflow.keras.layers import Embedding, Flatten, Reshape
from tensorflow.keras.layers import Dropout, Add, Multiply
id_embedding = Embedding(
    input_dim=num_skus,
    output_dim=ID_EMBEDDING_DIM,
    embeddings_initializer='glorot_normal',
    name='id_embedding'
)(sku_input)
id_embedding = Flatten(name='id_embedding_flat')(id_embedding)

# Component feature processing - each produces a forecast
# Trend = base forecast (with bias)
# Seasonal/Holiday = deviations from trend (no bias)
# Regressor component removed (set to None)

# Trend component: base forecast with bias + ID interaction
trend_out = Dense(COMPONENT_HIDDEN_UNITS, activation=mish, use_bias=True,
                  kernel_initializer='he_normal',
                  name='trend_hidden')(main_input)
# ID-specific scaling via element-wise multiplication
id_trend_scale = Dense(COMPONENT_HIDDEN_UNITS, activation='sigmoid',
                       use_bias=False, kernel_initializer='glorot_normal',
                       name='id_trend_scale')(id_embedding)
trend_out = Multiply(name='trend_id_interaction')([trend_out, id_trend_scale])
trend_out = Dropout(COMPONENT_DROPOUT)(trend_out)
trend_forecast = Dense(1, activation='linear', use_bias=True,
                       kernel_initializer='glorot_normal',
                       name='trend_forecast')(trend_out)

# Seasonal component: deviation from trend (no bias) + ID interaction
seasonal_out = Dense(COMPONENT_HIDDEN_UNITS, activation=mish, use_bias=False,
                     kernel_initializer='he_normal',
                     name='seasonal_hidden')(main_input)
# ID-specific seasonal residual: output = feature + α*embedding
id_seasonal_residual = Dense(COMPONENT_HIDDEN_UNITS, activation='linear',
                              use_bias=False, kernel_initializer='glorot_normal',
                              name='id_seasonal_residual')(id_embedding)
seasonal_out = Add(name='seasonal_id_interaction')([
    seasonal_out, id_seasonal_residual])
seasonal_out = Dropout(COMPONENT_DROPOUT)(seasonal_out)
seasonal_forecast = Dense(1, activation='linear', use_bias=False,
                          kernel_initializer='glorot_normal',
                          name='seasonal_forecast')(seasonal_out)

# Holiday component: deviation from trend using PWL + Lattice
# Extract holiday_distance feature (index 11 in features)
import tensorflow_lattice as tfl
from tensorflow.keras.layers import Lambda, Concatenate as Concat

holiday_distance_input = Lambda(
    lambda x: x[:, 8:9],  # holiday_distance is now at index 8
    name='holiday_distance_extract'
)(main_input)

# PWL calibration: adapt range based on data granularity
# Daily, Weekly, Monthly, or Quarterly data
DATA_FREQUENCY = 'daily'  # Options: 'daily', 'weekly', 'monthly', 'quarterly'

if DATA_FREQUENCY == 'daily':
    # Daily data: use ±365 days (1 year) to capture full yearly patterns
    keypoint_range = 365  # ±1 year
    num_keypoints = 37  # ~10 days per keypoint
elif DATA_FREQUENCY == 'weekly':
    # Weekly data: 52 weeks = 364 days
    keypoint_range = 364  # ±52 weeks (1 year)
    num_keypoints = 27  # ~2 weeks per keypoint
elif DATA_FREQUENCY == 'monthly':
    # Monthly data: 12 months = 365 days
    keypoint_range = 365  # ±12 months (1 year)
    num_keypoints = 13  # 1 month per keypoint
elif DATA_FREQUENCY == 'quarterly':
    # Quarterly data: 4 quarters = 365 days
    keypoint_range = 365  # ±4 quarters (1 year)
    num_keypoints = 9  # 1 quarter per keypoint
else:
    keypoint_range = 365
    num_keypoints = 37

# Cast to float32 explicitly for PWLCalibration compatibility
holiday_distance_float32 = tf.cast(holiday_distance_input, tf.float32)

holiday_pwl = tfl.layers.PWLCalibration(
    input_keypoints=np.linspace(-keypoint_range, keypoint_range, num_keypoints).astype(np.float32),
    output_min=-2.0,
    output_max=2.0,
    monotonicity='none',  # Holidays can increase/decrease demand
    kernel_regularizer=('hessian', 0.0, 1e-3),
    dtype='float32',  # Explicitly set dtype
    name='holiday_pwl'
)(holiday_distance_float32)

# Lattice layer: capture non-linear holiday effects
holiday_lattice = tfl.layers.Lattice(
    lattice_sizes=[num_keypoints],
    output_min=-2.0,
    output_max=2.0,
    kernel_regularizer=('torsion', 0.0, 1e-4),
    name='holiday_lattice'
)(holiday_pwl)

# Combine with other features through dense layer + ID interaction
holiday_combined = Concat(name='holiday_combined')([
    main_input, holiday_lattice
])
holiday_out = Dense(COMPONENT_HIDDEN_UNITS, activation=mish,
                    use_bias=False, kernel_initializer='he_normal',
                    name='holiday_hidden')(holiday_combined)
# ID-specific holiday residual: output = feature + α*embedding
id_holiday_residual = Dense(COMPONENT_HIDDEN_UNITS, activation='linear',
                            use_bias=False, kernel_initializer='glorot_normal',
                            name='id_holiday_residual')(id_embedding)
holiday_out = Add(name='holiday_id_interaction')([
    holiday_out, id_holiday_residual])
holiday_out = Dropout(COMPONENT_DROPOUT)(holiday_out)
holiday_forecast = Dense(1, activation='linear', use_bias=False,
                         kernel_initializer='glorot_normal',
                         name='holiday_forecast')(holiday_out)

# Regressor component: REMOVED (set to None per requirements)
# Architecture now uses only 3 components: trend, seasonal, holiday

# Additive combination: base_forecast = trend + seasonal + holiday
base_forecast = Add(name='base_forecast')([
    trend_forecast,
    seasonal_forecast,
    holiday_forecast
])

# Component identifiability penalties: DISABLED for TF 2.17/Metal compatibility
# TODO: Re-enable when using TF 2.20+ or fix Keras version conflict
# Zero-mean penalty would improve identifiability of seasonal/holiday

# Now build intermittent handler to predict zero probability
# Use the hidden representations (not forecasts) for zero probability prediction
# DeepSequence architecture: combined output from 3 components
from tensorflow.keras.layers import Concatenate
combined_features = Concatenate(name='combined_features')([
    trend_out, seasonal_out, holiday_out
])

# Add Transformer block on DeepSequence combined features
# Transformer input dim = 3 components * 32 units = 96 (matches TRANSFORMER_FFN_UNITS)
print("\n  Adding Transformer block on DeepSequence component outputs...")
print(f"    Combined features dimension: {3 * COMPONENT_HIDDEN_UNITS}")
print(f"    Transformer FFN dimension: {TRANSFORMER_FFN_UNITS}")
# Reshape for transformer: (batch, seq_len=1, features)
transformer_input = tf.expand_dims(combined_features, axis=1)

# Multi-head self-attention
attention_output = MultiHeadAttention(
    num_heads=TRANSFORMER_NUM_HEADS,
    key_dim=TRANSFORMER_KEY_DIM,
    name='transformer_attention'
)(transformer_input, transformer_input)

# Add & Norm
attention_output = Add(name='transformer_add1')([
    transformer_input, attention_output
])
attention_output = LayerNormalization(
    name='transformer_norm1'
)(attention_output)

# Feed-forward network (dimension matches DeepSequence output: 96)
ffn_output = Dense(TRANSFORMER_FFN_UNITS, activation=mish,
                   kernel_initializer='he_normal',
                   name='transformer_ffn1')(attention_output)
ffn_output = Dropout(TRANSFORMER_DROPOUT)(ffn_output)
ffn_output = Dense(TRANSFORMER_FFN_UNITS, activation=mish,
                   kernel_initializer='he_normal',
                   name='transformer_ffn2')(ffn_output)

# Add & Norm
transformer_output = Add(name='transformer_add2')([
    attention_output, ffn_output
])
transformer_output = LayerNormalization(
    name='transformer_norm2'
)(transformer_output)

# Flatten back to (batch, features)
transformer_features = tf.squeeze(transformer_output, axis=1)

# Intermittent handler: predict zero probability from transformer output
zero_hidden = transformer_features
for i in range(ZERO_PROB_HIDDEN_LAYERS):
    zero_hidden = Dense(ZERO_PROB_HIDDEN_UNITS, activation=mish,
                        kernel_initializer='he_normal',
                        name=f'zero_prob_hidden_{i+1}')(zero_hidden)
    zero_hidden = Dropout(ZERO_PROB_DROPOUT)(zero_hidden)

zero_probability = Dense(1, activation='sigmoid',
                        kernel_initializer='glorot_normal',
                        name='zero_probability')(zero_hidden)
# Cast to float32 for compatibility with float32 constants
zero_probability = tf.cast(zero_probability, tf.float32)

# Final forecast = base_forecast * (1 - zero_probability)
one_minus_zero = tf.subtract(1.0, zero_probability, name='non_zero_prob')
final = Multiply(name='final_forecast')([base_forecast, one_minus_zero])

# Create individual component models (share weights with main model)
# These can be used after training to extract component-specific predictions
print("\n  Creating component models for decomposition...")
trend_model = Model(
    inputs=[main_input, sku_input],
    outputs=trend_forecast,
    name='TrendComponent'
)

seasonal_model = Model(
    inputs=[main_input, sku_input],
    outputs=seasonal_forecast,
    name='SeasonalComponent'
)

holiday_model = Model(
    inputs=[main_input, sku_input],
    outputs=holiday_forecast,
    name='HolidayComponent'
)

# Regressor component removed (set to None)
regressor_model = None

# Create model with dual inputs (features + SKU ID)
two_stage_model = Model(
    inputs=[main_input, sku_input],
    outputs={
        'zero_probability': zero_probability,
        'base_forecast': base_forecast,
        'final_forecast': final
    },
    name='TwoStage_Additive_Transformer_Model'
)

print(f"✓ Model built: {two_stage_model.count_params():,} parameters")
print("  Component models created: Trend, Seasonal, Holiday (Regressor=None)")
print("  Component penalties: DISABLED (TF compatibility)")
print(f"  DeepSequence output dimension: {3 * COMPONENT_HIDDEN_UNITS} = Transformer FFN: {TRANSFORMER_FFN_UNITS}")

# Compile with Adam optimizer and learning rate schedule
import tf_keras
lr_schedule_callback = tf_keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

optimizer = tf_keras.optimizers.Adam(
    learning_rate=INITIAL_LEARNING_RATE
)

# Create composite loss with SKU-aware weighting
# Use LOSS_ALPHA from configuration (tunable hyperparameter)
composite_loss_fn = create_composite_loss_with_sku_weights(
    sku_weights_dict=sku_weights,
    alpha=LOSS_ALPHA
)

two_stage_model.compile(
    optimizer=optimizer,
    loss={
        'zero_probability': 'binary_crossentropy',
        'base_forecast': composite_loss_fn,
        'final_forecast': composite_loss_fn
    },
    loss_weights={
        'zero_probability': 1.0,
        'base_forecast': 0.5,
        'final_forecast': 1.0
    },
    metrics={'final_forecast': ['mae']},
    weighted_metrics=[]  # Silence sample_weight warning
)

print(f"\n[6/6] Training two-stage model with {LOSS_TYPE} loss + {ACTIVATION}...")
print(f"  Composite: BCE (y>0) + weighted MAE with log1p(sku_mean)")
print(f"  Activation: {ACTIVATION} (x * tanh(softplus(x)))")
print(f"  Balance coefficient alpha = {LOSS_ALPHA}")
nn_start = time.time()

history = two_stage_model.fit(
    [X_train_nn, sku_train],
    {
        'zero_probability': y_train_zero,
        'base_forecast': y_train_nn,
        'final_forecast': y_train_nn
    },
    sample_weight={
        'zero_probability': np.ones_like(y_train_zero),  # No weighting for zero prob
        'base_forecast': train_sample_weights,
        'final_forecast': train_sample_weights
    },
    validation_data=(
        [X_val_nn, sku_val],
        {
            'zero_probability': y_val_zero,
            'base_forecast': y_val_nn,
            'final_forecast': y_val_nn
        },
        {
            'zero_probability': np.ones_like(y_val_zero),
            'base_forecast': val_sample_weights,
            'final_forecast': val_sample_weights
        }
    ),
    batch_size=BATCH_SIZE,
    epochs=100,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        lr_schedule_callback
    ],
    verbose=1
)

nn_time = time.time() - nn_start

# Predictions with dual inputs
train_pred = two_stage_model.predict([X_train_nn, sku_train],
                                     batch_size=1024, verbose=0)
val_pred = two_stage_model.predict([X_val_nn, sku_val],
                                   batch_size=1024, verbose=0)
test_pred = two_stage_model.predict([X_test_nn, sku_test],
                                    batch_size=1024, verbose=0)

# Extract final forecasts
nn_train_pred = train_pred['final_forecast'].flatten()
nn_val_pred = val_pred['final_forecast'].flatten()
nn_test_pred = test_pred['final_forecast'].flatten()

# Extract component predictions using component models
print("\n[6/6] Extracting component predictions...")
trend_test = trend_model.predict([X_test_nn, sku_test],
                                 batch_size=1024, verbose=0).flatten()
seasonal_test = seasonal_model.predict([X_test_nn, sku_test],
                                       batch_size=1024, verbose=0).flatten()
holiday_test = holiday_model.predict([X_test_nn, sku_test],
                                     batch_size=1024, verbose=0).flatten()
# Regressor component removed (set to None)
regressor_test = np.zeros_like(trend_test)

# Metrics
nn_train_mae = mean_absolute_error(y_train_nn, nn_train_pred)
nn_val_mae = mean_absolute_error(y_val_nn, nn_val_pred)
nn_test_mae = mean_absolute_error(y_test_nn, nn_test_pred)

print(f"\n✓ Two-stage model trained in {nn_time:.1f}s")
print(f"  Epochs: {len(history.history['loss'])}")
print(f"  Train MAE: {nn_train_mae:.4f}")
print(f"  Val MAE:   {nn_val_mae:.4f}")
print(f"  Test MAE:  {nn_test_mae:.4f}")

# Component statistics
print(f"\nComponent contributions (test set):")
print(f"  Trend:     mean={trend_test.mean():7.2f}, std={trend_test.std():7.2f}")
print(f"  Seasonal:  mean={seasonal_test.mean():7.2f}, std={seasonal_test.std():7.2f}")
print(f"  Holiday:   mean={holiday_test.mean():7.2f}, std={holiday_test.std():7.2f}")
print(f"  Regressor: REMOVED (set to None)")

# ============================================================================
# COMPARISON RESULTS
# ============================================================================
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print("\nModel Performance:")
print(f"{'Model':<30} {'Train MAE':<12} {'Val MAE':<12} {'Test MAE':<12} {'Time (s)':<10}")
print("-"*80)
print(f"{'LightGBM Baseline':<30} {lgb_train_mae:<12.4f} {lgb_val_mae:<12.4f} {lgb_test_mae:<12.4f} {lgb_time:<10.1f}")
print(f"{'Two-Stage Intermittent':<30} {nn_train_mae:<12.4f} {nn_val_mae:<12.4f} {nn_test_mae:<12.4f} {nn_time:<10.1f}")

# Calculate improvement
val_improvement = 100 * (lgb_val_mae - nn_val_mae) / lgb_val_mae
test_improvement = 100 * (lgb_test_mae - nn_test_mae) / lgb_test_mae

print(f"\nImprovement over LightGBM:")
print(f"  Val:  {val_improvement:+.2f}%")
print(f"  Test: {test_improvement:+.2f}%")

# Zero probability and component analysis
print("\n" + "="*80)
print("COMPONENT & INTERMITTENT ANALYSIS")
print("="*80)

zero_probs_val = val_pred['zero_probability'].flatten()
base_forecast_val = val_pred['base_forecast'].flatten()
actual_zeros_val = (y_val_nn == 0)

print(f"\nValidation Set:")
print(f"  Actual zeros: {actual_zeros_val.sum()}/{len(actual_zeros_val)} ({100*actual_zeros_val.mean():.1f}%)")
print(f"  Predicted zero prob (mean): {zero_probs_val.mean():.3f}")
print(f"  Predicted zero prob (median): {np.median(zero_probs_val):.3f}")
print(f"  Predicted zero prob (range): [{zero_probs_val.min():.3f}, {zero_probs_val.max():.3f}]")
print(f"\n  Base forecast (additive components):")
print(f"    Mean: {base_forecast_val.mean():.3f}")
print(f"    Range: [{base_forecast_val.min():.3f}, {base_forecast_val.max():.3f}]")
print(f"\n  Final forecast = base * (1 - zero_prob):")
print(f"    Mean: {nn_val_pred.mean():.3f}")
print(f"    Range: [{nn_val_pred.min():.3f}, {nn_val_pred.max():.3f}]")

# Correlation between predicted zero prob and actual zeros
from scipy.stats import pearsonr
if zero_probs_val.std() > 0:
    corr, pval = pearsonr(zero_probs_val, actual_zeros_val)
    print(f"\n  Zero prob correlation with actual: {corr:.3f} (p={pval:.4f})")
else:
    print(f"\n  Zero prob correlation: N/A (constant predictions)")

# Save results
results_df = pd.DataFrame({
    'Model': ['LightGBM', 'TwoStage_Intermittent'],
    'Train_MAE': [lgb_train_mae, nn_train_mae],
    'Val_MAE': [lgb_val_mae, nn_val_mae],
    'Test_MAE': [lgb_test_mae, nn_test_mae],
    'Training_Time_s': [lgb_time, nn_time],
    'Val_Improvement_%': [0.0, val_improvement],
    'Test_Improvement_%': [0.0, test_improvement]
})

results_df.to_csv('lgb_vs_twostage_results.csv', index=False)
print(f"\n✓ Results saved to lgb_vs_twostage_results.csv")

# Save component decomposition for analysis and explainability
print("\nSaving component decomposition...")
component_df = pd.DataFrame({
    'actual': y_test_nn,
    'final_forecast': nn_test_pred,
    'base_forecast': test_pred['base_forecast'].flatten(),
    'zero_probability': test_pred['zero_probability'].flatten(),
    'trend': trend_test,
    'seasonal': seasonal_test,
    'holiday': holiday_test,
    'regressor': regressor_test,
    'sku_id': test.iloc[:len(y_test_nn)]['sku_id'].values
})
component_df.to_csv('component_decomposition_test.csv', index=False)
print("✓ Component decomposition saved to component_decomposition_test.csv")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)

if nn_val_mae < lgb_val_mae:
    print("\n🎉 Two-Stage Intermittent Handler WINS!")
    print(f"   {val_improvement:.2f}% better on validation set")
else:
    print("\n⚠ LightGBM performs better")
    print(f"   Two-stage needs tuning or more data")
