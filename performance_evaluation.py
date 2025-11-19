"""
Performance Evaluation: DeepSequence vs LightGBM
Compare models on actual data with comprehensive metrics
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PERFORMANCE EVALUATION: DeepSequence vs LightGBM Baselines")
print("="*80)

# Load data
print("\n📊 Loading data...")
data = pd.read_csv('data/cleaned_data_week.csv')
print(f"   Total records: {len(data):,}")
print(f"   Columns: {list(data.columns)}")
print(f"   Date range: {data['ds'].min()} to {data['ds'].max()}")

# Data analysis
print("\n🔍 Data Analysis:")
print(f"   Unique SKUs: {data['id_var'].nunique():,}")
print(f"   Zero demand records: {(data['Quantity'] == 0).sum():,} ({(data['Quantity'] == 0).mean()*100:.1f}%)")
print(f"   Non-zero demand records: {(data['Quantity'] > 0).sum():,} ({(data['Quantity'] > 0).mean()*100:.1f}%)")
print(f"   Mean quantity: {data['Quantity'].mean():.2f}")
print(f"   Median quantity: {data['Quantity'].median():.2f}")
print(f"   Max quantity: {data['Quantity'].max():.0f}")

# Feature engineering
print("\n⚙️  Feature Engineering...")
data['ds'] = pd.to_datetime(data['ds'])
data = data.sort_values(['id_var', 'ds'])

# Time features
data['year'] = data['ds'].dt.year
data['month'] = data['ds'].dt.month
data['week'] = data['ds'].dt.isocalendar().week
data['dayofweek'] = data['ds'].dt.dayofweek
data['dayofyear'] = data['ds'].dt.dayofyear

# Lag features (for comparison with LightGBM)
for lag in [1, 7, 14, 28]:
    data[f'lag_{lag}'] = data.groupby('id_var')['Quantity'].shift(lag)

# Rolling features
for window in [7, 14, 28]:
    data[f'rolling_mean_{window}'] = data.groupby('id_var')['Quantity'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    data[f'rolling_std_{window}'] = data.groupby('id_var')['Quantity'].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )

# Drop rows with NaN from lag features
data = data.dropna()
print(f"   Records after feature engineering: {len(data):,}")

# Split data
print("\n📦 Splitting data...")
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size+val_size]
test_data = data.iloc[train_size+val_size:]

print(f"   Train: {len(train_data):,} records")
print(f"   Validation: {len(val_data):,} records")
print(f"   Test: {len(test_data):,} records")

# Prepare features
seasonal_features = ['year', 'month', 'week', 'dayofweek', 'dayofyear']
regressor_features = ['cumsum', 'cumdist', 'holiday'] + \
                     [f'lag_{lag}' for lag in [1, 7, 14, 28]] + \
                     [f'rolling_mean_{window}' for window in [7, 14, 28]] + \
                     [f'rolling_std_{window}' for window in [7, 14, 28]]

all_features = seasonal_features + regressor_features

print(f"\n📋 Features:")
print(f"   Seasonal: {seasonal_features}")
print(f"   Regressor: {len(regressor_features)} features")

# Evaluate metrics
def evaluate_metrics(y_true, y_pred, prefix=""):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE (handle zeros)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = 0.0
    
    # Zero prediction accuracy
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    zero_acc = (y_true_binary == y_pred_binary).mean() * 100
    
    # Metrics by demand type
    zero_mask = y_true == 0
    nonzero_mask = y_true > 0
    
    mae_zero = mean_absolute_error(y_true[zero_mask], y_pred[zero_mask]) if zero_mask.sum() > 0 else 0
    mae_nonzero = mean_absolute_error(y_true[nonzero_mask], y_pred[nonzero_mask]) if nonzero_mask.sum() > 0 else 0
    
    print(f"\n{prefix} Performance:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Zero Prediction Accuracy: {zero_acc:.2f}%")
    print(f"   MAE (Zero demand): {mae_zero:.4f}")
    print(f"   MAE (Non-zero demand): {mae_nonzero:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'zero_acc': zero_acc,
        'mae_zero': mae_zero,
        'mae_nonzero': mae_nonzero
    }

# ============================================================================
# 1. NAIVE BASELINE (7-day shift)
# ============================================================================
print("\n" + "="*80)
print("1️⃣  NAIVE BASELINE (7-day lag)")
print("="*80)

y_test = test_data['Quantity'].values
y_pred_naive = test_data['lag_7'].values

naive_metrics = evaluate_metrics(y_test, y_pred_naive, "Naive (lag-7)")

# ============================================================================
# 2. LIGHTGBM BASELINE (from existing results)
# ============================================================================
print("\n" + "="*80)
print("2️⃣  LIGHTGBM BASELINE")
print("="*80)

try:
    # Load existing LightGBM predictions if available
    lgb_test = pd.read_csv('data/test_lgb.csv')
    print(f"   Loaded LightGBM predictions: {len(lgb_test):,} records")
    
    if 'predictions' in lgb_test.columns:
        y_pred_lgb = lgb_test['predictions'].values[:len(y_test)]
        lgb_metrics = evaluate_metrics(y_test[:len(y_pred_lgb)], y_pred_lgb, "LightGBM")
    else:
        print("   ⚠️  No predictions column found, skipping LightGBM evaluation")
        lgb_metrics = None
except FileNotFoundError:
    print("   ⚠️  LightGBM test file not found, skipping comparison")
    lgb_metrics = None

# ============================================================================
# 3. DEEPSEQUENCE MODEL (Quick evaluation)
# ============================================================================
print("\n" + "="*80)
print("3️⃣  DEEPSEQUENCE MODEL (with TabNet + UnitNorm)")
print("="*80)

print("\n📦 Building DeepSequence model...")
from src.deepsequence import (
    DeepSequenceModel, 
    SeasonalComponent, 
    RegressorComponent
)

# Prepare data for DeepSequence
X_train_seasonal = train_data[seasonal_features].values
X_train_regressor = train_data[regressor_features].values
y_train = train_data['Quantity'].values

X_val_seasonal = val_data[seasonal_features].values
X_val_regressor = val_data[regressor_features].values
y_val = val_data['Quantity'].values

X_test_seasonal = test_data[seasonal_features].values
X_test_regressor = test_data[regressor_features].values

print(f"   Train shapes: Seasonal {X_train_seasonal.shape}, Regressor {X_train_regressor.shape}")
print(f"   Test shapes: Seasonal {X_test_seasonal.shape}, Regressor {X_test_regressor.shape}")

# Build components
print("\n🏗️  Building model components...")
seasonal_comp = SeasonalComponent(
    seasonal_features=X_train_seasonal,
    output=y_train,
    seasonal_input_dim=len(seasonal_features)
)

regressor_comp = RegressorComponent(
    training_data=X_train_regressor,
    output=y_train,
    regressor_input_dim=len(regressor_features)
)

# Build DeepSequence model with TabNet and UnitNorm
deep_model = DeepSequenceModel(
    mode='additive',
    use_intermittent=True,
    use_tabnet=True
)

deep_model.build(
    seasonal_component=seasonal_comp,
    regressor_component=regressor_comp,
    tabnet_config={
        'feature_dim': 32,
        'output_dim': 32,
        'num_steps': 3,
        'relaxation_factor': 1.5
    },
    intermittent_config={
        'hidden_layers': [64, 32]
    }
)

model = deep_model.full_model

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mse']
)

print("\n🏗️  Model Architecture:")
model.summary()

# Train model (quick training for evaluation)
print("\n🚀 Training DeepSequence model...")
start_time = time.time()

history = model.fit(
    [X_train_seasonal, X_train_regressor],
    y_train,
    validation_data=([X_val_seasonal, X_val_regressor], y_val),
    epochs=50,
    batch_size=256,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
)

training_time = time.time() - start_time
print(f"\n✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")

# Predict on test set
print("\n🔮 Generating predictions...")
y_pred_deep = model.predict([X_test_seasonal, X_test_regressor], verbose=0).flatten()
y_pred_deep = np.maximum(y_pred_deep, 0)  # Ensure non-negative predictions

deep_metrics = evaluate_metrics(y_test, y_pred_deep, "DeepSequence")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📊 FINAL PERFORMANCE COMPARISON")
print("="*80)

comparison_data = {
    'Model': ['Naive (lag-7)', 'LightGBM', 'DeepSequence'],
    'MAE': [naive_metrics['mae'], 
            lgb_metrics['mae'] if lgb_metrics else np.nan,
            deep_metrics['mae']],
    'RMSE': [naive_metrics['rmse'],
             lgb_metrics['rmse'] if lgb_metrics else np.nan,
             deep_metrics['rmse']],
    'MAPE (%)': [naive_metrics['mape'],
                 lgb_metrics['mape'] if lgb_metrics else np.nan,
                 deep_metrics['mape']],
    'Zero Accuracy (%)': [naive_metrics['zero_acc'],
                          lgb_metrics['zero_acc'] if lgb_metrics else np.nan,
                          deep_metrics['zero_acc']],
    'MAE (Zero)': [naive_metrics['mae_zero'],
                   lgb_metrics['mae_zero'] if lgb_metrics else np.nan,
                   deep_metrics['mae_zero']],
    'MAE (Non-zero)': [naive_metrics['mae_nonzero'],
                       lgb_metrics['mae_nonzero'] if lgb_metrics else np.nan,
                       deep_metrics['mae_nonzero']]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + str(comparison_df.to_string(index=False)))

# Improvement analysis
if lgb_metrics:
    print("\n🎯 DeepSequence vs LightGBM Improvements:")
    mae_improvement = (lgb_metrics['mae'] - deep_metrics['mae']) / lgb_metrics['mae'] * 100
    zero_improvement = (deep_metrics['zero_acc'] - lgb_metrics['zero_acc'])
    mae_zero_improvement = (lgb_metrics['mae_zero'] - deep_metrics['mae_zero']) / lgb_metrics['mae_zero'] * 100
    
    print(f"   MAE improvement: {mae_improvement:+.1f}%")
    print(f"   Zero accuracy improvement: {zero_improvement:+.1f} percentage points")
    print(f"   MAE (zero demand) improvement: {mae_zero_improvement:+.1f}%")

print("\n🎯 DeepSequence vs Naive Improvements:")
mae_improvement_naive = (naive_metrics['mae'] - deep_metrics['mae']) / naive_metrics['mae'] * 100
zero_improvement_naive = (deep_metrics['zero_acc'] - naive_metrics['zero_acc'])

print(f"   MAE improvement: {mae_improvement_naive:+.1f}%")
print(f"   Zero accuracy improvement: {zero_improvement_naive:+.1f} percentage points")

# Save results
print("\n💾 Saving results...")
comparison_df.to_csv('performance_comparison_results.csv', index=False)
print("   ✅ Saved to: performance_comparison_results.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'actual': y_test,
    'naive': y_pred_naive,
    'deepsequence': y_pred_deep
})
predictions_df.to_csv('test_predictions_comparison.csv', index=False)
print("   ✅ Saved to: test_predictions_comparison.csv")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE")
print("="*80)
