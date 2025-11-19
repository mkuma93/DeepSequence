# Data Directory

This directory should contain your time series forecasting data.

---

## Required Data Format

### Main Dataset: `cleaned_data_week.csv`

Your main dataset should have the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `ds` | datetime | Date/timestamp | 2024-01-01 |
| `id_var` | string/int | SKU/Product identifier | SKU_12345 |
| `Quantity` | float | Target variable (demand/sales) | 15.0 |

### Optional Columns

Additional features that improve model performance:

| Column | Type | Description |
|--------|------|-------------|
| `Price` | float | Product price |
| `cluster` | int | Product cluster ID (from clustering) |
| `holiday` | int | Holiday indicator (0/1) |
| `week_no` | int | Week number |
| `year` | int | Year |
| `month` | int | Month |

---

## Directory Structure

```
data/
├── README.md                    # This file
├── cleaned_data_week.csv       # Your main dataset
├── test_lgb.csv               # Optional: LightGBM baseline predictions
└── raw/                        # Your raw data (gitignored)
    ├── sales_data.csv
    ├── product_info.csv
    └── ...
```

---

## Data Preparation

### Step 1: Collect Your Data

Gather your time series data with at least these three columns:
- Date/timestamp
- Product/SKU identifier  
- Target value (sales, demand, etc.)

### Step 2: Feature Engineering

Add useful features:
```python
import pandas as pd

# Load raw data
df = pd.read_csv('raw/sales_data.csv')

# Create time features
df['ds'] = pd.to_datetime(df['date_column'])
df['year'] = df['ds'].dt.year
df['month'] = df['ds'].dt.month
df['week_no'] = df['ds'].dt.isocalendar().week

# Create lag features
df['lag_1'] = df.groupby('SKU')['Quantity'].shift(1)
df['lag_7'] = df.groupby('SKU')['Quantity'].shift(7)

# Save processed data
df.to_csv('cleaned_data_week.csv', index=False)
```

### Step 3: Weekly Aggregation (Optional)

If your data is daily but you want weekly forecasts:
```python
df_weekly = df.groupby(['SKU', pd.Grouper(key='ds', freq='W')]).agg({
    'Quantity': 'sum',
    'Price': 'mean',
    # ... other aggregations
}).reset_index()
```

---

## Example Data

Minimal example of what `cleaned_data_week.csv` should look like:

```csv
ds,id_var,Quantity,week_no,year,month
2024-01-01,SKU_001,25.0,1,2024,1
2024-01-08,SKU_001,30.0,2,2024,1
2024-01-15,SKU_001,0.0,3,2024,1
2024-01-22,SKU_001,18.0,4,2024,1
2024-01-01,SKU_002,12.0,1,2024,1
2024-01-08,SKU_002,15.0,2,2024,1
...
```

---

## Data Requirements

### Minimum Requirements
- ✅ At least 52 weeks of historical data per SKU
- ✅ Regular time intervals (weekly recommended)
- ✅ Numeric target variable
- ✅ Valid datetime column

### Recommended
- 📊 100+ SKUs for robust evaluation
- 📊 2+ years of historical data
- 📊 Exogenous variables (price, promotions, holidays)
- 📊 SKU metadata (category, cluster)

---

## Data Quality Checks

Before training models, verify:

```python
import pandas as pd

df = pd.read_csv('cleaned_data_week.csv')

# Check required columns
required_cols = ['ds', 'id_var', 'Quantity']
assert all(col in df.columns for col in required_cols), "Missing required columns"

# Check for missing values
print(f"Missing values:\n{df[required_cols].isnull().sum()}")

# Check date format
df['ds'] = pd.to_datetime(df['ds'])
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

# Check data distribution
print(f"\nTarget statistics:")
print(df['Quantity'].describe())

# Check intermittency
zero_pct = (df['Quantity'] == 0).mean() * 100
print(f"\nZero demand: {zero_pct:.1f}%")
```

---

## Sample Data Generation

If you don't have data yet, generate sample data for testing:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate dates
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(weeks=i) for i in range(104)]  # 2 years

# Generate SKUs
skus = [f'SKU_{i:03d}' for i in range(1, 51)]  # 50 SKUs

# Create sample data
data = []
for sku in skus:
    for date in dates:
        # Simulate intermittent demand
        if np.random.random() > 0.7:  # 70% zeros
            quantity = 0
        else:
            quantity = np.random.poisson(10)
        
        data.append({
            'ds': date,
            'id_var': sku,
            'Quantity': quantity,
            'week_no': date.isocalendar()[1],
            'year': date.year,
            'month': date.month
        })

df = pd.DataFrame(data)
df.to_csv('cleaned_data_week.csv', index=False)
print(f"Generated {len(df)} records for {len(skus)} SKUs")
```

---

## Privacy & Security

⚠️ **Important**: 
- Never commit proprietary data to version control
- The `data/raw/` directory is gitignored
- Anonymize SKU identifiers if sharing publicly
- Remove sensitive pricing information if needed

---

## Need Help?

If you have questions about data format or preparation, please:
1. Check the example notebooks in `notebooks/`
2. Review `performance_evaluation.py` for data loading examples
3. Open an issue on GitHub

---

**Data Format Version**: 1.0  
**Last Updated**: November 18, 2025
