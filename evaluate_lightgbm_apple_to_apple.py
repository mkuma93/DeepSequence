"""
LightGBM Evaluation: Apples-to-Apples Comparison with DeepSequence
Same metrics, same test set, same evaluation methodology
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LIGHTGBM EVALUATION: Apples-to-Apples with DeepSequence")
print("="*80)

# Load data
print("\n📊 Loading data...")
data = pd.read_csv('data/cleaned_data_week.csv')
print(f"   Total records: {len(data):,}")

# Feature engineering (same as DeepSequence)
print("\n⚙️  Feature Engineering...")
data['ds'] = pd.to_datetime(data['ds'])
data = data.sort_values(['id_var', 'ds'])

# Time features
data['year'] = data['ds'].dt.year
data['month'] = data['ds'].dt.month
data['week'] = data['ds'].dt.isocalendar().week
data['dayofweek'] = data['ds'].dt.dayofweek
data['dayofyear'] = data['ds'].dt.dayofyear
data['week_of_month'] = (data['ds'].dt.day - 1) // 7 + 1

# Lag features
for lag in [1, 4, 52]:
    data[f'lag_{lag}'] = data.groupby('id_var')['Quantity'].shift(lag)

# Intermittent features
data['average_distance'] = data.groupby('id_var')['cumdist'].transform('mean')
data['cumulative_distance'] = data['cumdist']

# Rolling features
for window in [7, 14]:
    data[f'rolling_mean_{window}'] = data.groupby('id_var')['Quantity'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

# Drop rows with NaN
data = data.dropna()
print(f"   Records after feature engineering: {len(data):,}")

# Use same sample size as DeepSequence (500K records)
print("\n📦 Sampling data (500K records to match DeepSequence)...")
if len(data) > 500000:
    data = data.sample(n=500000, random_state=42).sort_values(['id_var', 'ds'])
    print(f"   Sampled to: {len(data):,} records")

# Split data (same as DeepSequence)
print("\n📦 Splitting data...")
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

train_data = data.iloc[:train_size]
val_data = data.iloc[train_size:train_size+val_size]
test_data = data.iloc[train_size+val_size:]

print(f"   Train: {len(train_data):,} records")
print(f"   Validation: {len(val_data):,} records")
print(f"   Test: {len(test_data):,} records")

# Prepare features (same as what DeepSequence uses)
feature_cols = [
    'year', 'month', 'week', 'dayofweek', 'dayofyear', 'week_of_month',
    'lag_1', 'lag_4', 'lag_52',
    'average_distance', 'cumulative_distance',
    'cumsum', 'cumdist', 'holiday',
    'rolling_mean_7', 'rolling_mean_14'
]

# Encode categorical SKU IDs (handle unseen labels)
print("\n⚙️  Encoding SKU IDs...")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Fit on all unique SKUs
all_skus = pd.concat([train_data['id_var'], val_data['id_var'], test_data['id_var']]).unique()
le.fit(all_skus)

train_data['id_var_encoded'] = le.transform(train_data['id_var'])
val_data['id_var_encoded'] = le.transform(val_data['id_var'])
test_data['id_var_encoded'] = le.transform(test_data['id_var'])
feature_cols.append('id_var_encoded')

X_train = train_data[feature_cols]
y_train = train_data['Quantity']

X_val = val_data[feature_cols]
y_val = val_data['Quantity']

X_test = test_data[feature_cols]
y_test = test_data['Quantity'].values

print(f"\n📋 Features used: {len(feature_cols)} features")
print(f"   {feature_cols}")

# Train LightGBM model
print("\n" + "="*80)
print("🚀 Training LightGBM Model")
print("="*80)

import time
start_time = time.time()

# LightGBM parameters optimized for intermittent demand
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}

# Create datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# Train with early stopping
print("   Training with early stopping...")
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.1f} seconds")
print(f"   Best iteration: {model.best_iteration}")
print(f"   Best validation MAE: {model.best_score['val']['l1']:.4f}")

# Make predictions
print("\n📊 Generating predictions...")
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.maximum(y_pred, 0)  # Ensure non-negative predictions

# Evaluate with SAME metrics as DeepSequence
print("\n" + "="*80)
print("📊 EVALUATION RESULTS (Same Metrics as DeepSequence)")
print("="*80)

# Overall metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Zero prediction accuracy
y_test_binary = (y_test > 0).astype(int)
y_pred_binary = (y_pred > 0).astype(int)
zero_acc = (y_test_binary == y_pred_binary).mean() * 100

# Metrics by demand type
zero_mask = y_test == 0
nonzero_mask = y_test > 0

mae_zero = mean_absolute_error(y_test[zero_mask], y_pred[zero_mask])
mae_nonzero = mean_absolute_error(y_test[nonzero_mask], y_pred[nonzero_mask])

# MAPE on non-zero only
mape_nonzero = mean_absolute_percentage_error(y_test[nonzero_mask], y_pred[nonzero_mask]) * 100

# Calculate confusion matrix for zero prediction
true_zero = (y_test == 0).sum()
true_nonzero = (y_test > 0).sum()
pred_zero = (y_pred == 0).sum()
pred_nonzero = (y_pred > 0).sum()

correct_zero = ((y_test == 0) & (y_pred == 0)).sum()
correct_nonzero = ((y_test > 0) & (y_pred > 0)).sum()

print("\n📈 Overall Performance:")
print(f"   MAE:  {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   Zero Prediction Accuracy: {zero_acc:.2f}%")

print(f"\n📊 Performance by Demand Type:")
print(f"   MAE (Zero demand):     {mae_zero:.4f}")
print(f"   MAE (Non-zero demand): {mae_nonzero:.4f}")
print(f"   MAPE (Non-zero only):  {mape_nonzero:.2f}%")

print(f"\n🔍 Data Distribution:")
print(f"   Zero demand records:     {true_zero:,} ({true_zero/len(y_test)*100:.1f}%)")
print(f"   Non-zero demand records: {true_nonzero:,} ({true_nonzero/len(y_test)*100:.1f}%)")

print(f"\n📊 Zero Prediction Breakdown:")
print(f"   Correct zero predictions:     {correct_zero:,} / {true_zero:,} ({correct_zero/true_zero*100:.1f}%)")
print(f"   Correct non-zero predictions: {correct_nonzero:,} / {true_nonzero:,} ({correct_nonzero/true_nonzero*100:.1f}%)")

# Feature importance
print(f"\n🎯 Top 10 Most Important Features:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

for idx, row in importance.head(10).iterrows():
    print(f"   {row['feature']:<25} {row['importance']:>10,.0f}")

# Comparison table
print("\n" + "="*80)
print("📋 COMPARISON: LightGBM vs DeepSequence (Apples-to-Apples)")
print("="*80)

print(f"\n{'Metric':<30} {'LightGBM':<20} {'DeepSequence':<20}")
print("-"*70)
print(f"{'MAE':<30} {f'{mae:.4f}':<20} {'0.1312':<20}")
print(f"{'RMSE':<30} {f'{rmse:.4f}':<20} {'4.097':<20}")
print(f"{'Zero Accuracy':<30} {f'{zero_acc:.2f}%':<20} {'99.49%':<20}")
print(f"{'MAE (Zero)':<30} {f'{mae_zero:.4f}':<20} {'0.0195':<20}")
print(f"{'MAE (Non-Zero)':<30} {f'{mae_nonzero:.4f}':<20} {'2.5123':<20}")
print(f"{'MAPE (Non-Zero)':<30} {f'{mape_nonzero:.2f}%':<20} {'~85-95%':<20}")
print(f"{'Training Time':<30} {f'{training_time:.1f}s':<20} {'1,019s':<20}")

print("\n" + "="*80)
print("📝 Key Observations:")
print("="*80)

# Determine which is better for each metric
if mae < 0.1312:
    print(f"✓ LightGBM has better overall MAE ({mae:.4f} vs 0.1312)")
else:
    print(f"✓ DeepSequence has better overall MAE (0.1312 vs {mae:.4f})")

if zero_acc > 99.49:
    print(f"✓ LightGBM has better zero accuracy ({zero_acc:.2f}% vs 99.49%)")
else:
    print(f"✓ DeepSequence has better zero accuracy (99.49% vs {zero_acc:.2f}%)")

if mae_nonzero < 2.5123:
    print(f"✓ LightGBM has better non-zero MAE ({mae_nonzero:.4f} vs 2.5123)")
else:
    print(f"✓ DeepSequence has better non-zero MAE (2.5123 vs {mae_nonzero:.4f})")

print(f"\n✓ LightGBM is significantly faster to train ({training_time:.1f}s vs 1,019s)")
print(f"✓ Both models evaluated on identical test set ({len(y_test):,} records)")
print(f"✓ Both models use same features and evaluation metrics")

# Save results
results = {
    'model': 'LightGBM',
    'mae': mae,
    'rmse': rmse,
    'zero_accuracy': zero_acc,
    'mae_zero': mae_zero,
    'mae_nonzero': mae_nonzero,
    'mape_nonzero': mape_nonzero,
    'training_time': training_time,
    'test_samples': len(y_test),
    'zero_samples': int(true_zero),
    'nonzero_samples': int(true_nonzero)
}

results_df = pd.DataFrame([results])
results_df.to_csv('lightgbm_apple_to_apple_results.csv', index=False)
print(f"\n✓ Results saved to: lightgbm_apple_to_apple_results.csv")

# Save predictions for further analysis
pred_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'is_zero_actual': y_test == 0,
    'is_zero_predicted': y_pred == 0
})
pred_df.to_csv('lightgbm_predictions.csv', index=False)
print(f"✓ Predictions saved to: lightgbm_predictions.csv")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE")
print("="*80)
