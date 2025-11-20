# DeepSequence - SKU Level Forecasting

A machine learning project for forecasting at SKU (Stock Keeping Unit) level using various techniques including LightGBM models and deep learning approaches.

## Project Overview

This project aims to forecast stock demand at the SKU level using historical data. Multiple modeling approaches have been explored and compared:

- **DeepSequence** ⭐: A custom deep learning architecture inspired by Prophet, designed specifically for SKU-level forecasting with seasonal patterns
- **LightGBM Models**: Cluster-based and distance-based forecasting approaches
- **Baseline Models**: Naive forecasting methods for comparison and benchmarking

### Key Innovation: DeepSequence

DeepSequence is an original deep learning architecture that combines:
- **Seasonal Components**: Inspired by Prophet's additive model for capturing weekly, monthly, and yearly seasonality
- **Trend Components**: Temporal pattern learning with attention mechanisms
- **Contextual Features**: Cluster-based and exogenous variables integration
- **Multi-horizon Forecasting**: Direct prediction of multiple future time steps

This custom architecture was developed to handle the unique challenges of retail SKU forecasting, including intermittent demand and multiple seasonal patterns.

## Architecture Diagram

```mermaid
graph TB
    subgraph Input["📊 Input Data"]
        TS[Time Series Data<br/>ds, StockCode, Quantity]
        EXOG[Exogenous Variables<br/>Price, Clusters, Holidays]
    end
    
    subgraph Seasonal["🌊 Seasonal Component"]
        TIME[Time Feature<br/>Extraction]
        EMBED_W[Weekly<br/>Embedding]
        EMBED_M[Monthly<br/>Embedding]
        EMBED_Y[Yearly<br/>Embedding]
        HIDDEN_S[Hidden Layers<br/>+ Dropout]
        OUT_S[Seasonal Output]
        
        TIME --> EMBED_W
        TIME --> EMBED_M
        TIME --> EMBED_Y
        EMBED_W --> HIDDEN_S
        EMBED_M --> HIDDEN_S
        EMBED_Y --> HIDDEN_S
        HIDDEN_S --> OUT_S
    end
    
    subgraph Regressor["📈 Regressor Component"]
        CAT[Categorical Features<br/>StockCode, Cluster]
        CONT[Context Variables<br/>Price, Lags]
        EMBED_R[Embedding Layer]
        LATTICE[Lattice Layer<br/>Constraints]
        HIDDEN_R[Hidden Layers<br/>+ Dropout]
        OUT_R[Regression Output]
        
        CAT --> EMBED_R
        CONT --> LATTICE
        EMBED_R --> HIDDEN_R
        LATTICE --> HIDDEN_R
        HIDDEN_R --> OUT_R
    end
    
    subgraph Combine["⚡ Combination Layer"]
        ADD[Additive/Multiplicative<br/>Mode]
        FINAL[Final Forecast<br/>ŷ]
        
        ADD --> FINAL
    end
    
    TS --> TIME
    TS --> CAT
    EXOG --> CONT
    
    OUT_S --> ADD
    OUT_R --> ADD
    
    style Input fill:#e1f5ff
    style Seasonal fill:#fff4e1
    style Regressor fill:#ffe1f5
    style Combine fill:#e1ffe1
```

**Formula**: `ŷ = f(σ_seasonal, τ_trend, τ_impact)` 

**Flexible Combination Modes**:
- **Additive**: `ŷ = σ_s + τ_r` (default, most stable)
- **Multiplicative**: `ŷ = σ_s × τ_r` (for scaling seasonality)
- **Hybrid**: `ŷ = σ_s × τ_i + τ_r` (seasonal scaled by impact + trend)
- **Learnable**: `ŷ = w₁σ_s + w₂τ_r + bias` (adaptive weights)
- **Custom**: Define your own combination function

📖 **[View combination modes guide →](COMBINATION_MODES_GUIDE.md)**  
📖 **[View detailed architecture diagrams →](docs/architecture_diagram.md)**

## Project Structure

```
forecasting/
├── src/deepsequence/            # Core DeepSequence package
│   ├── __init__.py
│   ├── model.py                 # Main model class
│   ├── seasonal_component.py   # Seasonal decomposition
│   ├── regressor_component.py  # Regression component
│   ├── combination_layer.py    # Flexible combinations ✨ NEW
│   ├── tabnet_encoder.py       # TabNet integration
│   ├── intermittent_handler.py # Zero demand handling
│   ├── unit_norm.py            # Normalization layer
│   ├── cross_layer.py          # Feature interactions
│   ├── utils.py                # Utility functions
│   ├── activations.py          # Custom activations
│   └── config.py               # Configuration
│
├── notebooks/
│   └── DeepSequence_Demo.ipynb # End-to-end demo
│
├── scripts/
│   └── fix_notebooks.py        # Notebook utility scripts
│
├── README.md
├── ARCHITECTURE.md             # Technical architecture docs
├── PERFORMANCE_COMPARISON.md   # Model benchmarks
├── TEST_REPORT.md             # Validation report
├── LICENSE                    # MIT License
└── requirements.txt           # Dependencies
```

## Requirements

```bash
# Core dependencies
pandas
numpy
scikit-learn
lightgbm
tensorflow  # or pytorch
matplotlib
seaborn
jupyter
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mkuma93/forecasting.git
cd forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Demo Notebook

```bash
jupyter notebook notebooks/DeepSequence_Demo.ipynb
```

### Using DeepSequence in Your Code

```python
import sys
sys.path.insert(0, 'src')
from deepsequence import DeepSequenceModel, SeasonalComponent, RegressorComponent

# Prepare your time series data
# ts: DataFrame with columns ['ds', 'id_cat', 'target_variable']
# exog: DataFrame with exogenous variables

# Build seasonal component
seasonal = SeasonalComponent(
    data=ts, target=['target'], id_var='id_cat',
    horizon=8, weekly=True, monthly=True, yearly=True
)
seasonal.seasonal_feature()
seasonal.seasonal_model(hidden=2, hidden_unit=32)

# Build regressor component
regressor = RegressorComponent(
    ts=ts, exog=exog, target=['target'], id_var='id_cat',
    categorical_var=['cluster'], context_variable=['price', 'lag1']
)
regressor.reg_model(id_input=seasonal.s_model.input[-1])

# Combine and train
model = DeepSequenceModel(mode='additive')
model.build(seasonal, regressor)
model.compile(loss='mape', learning_rate=0.001)
history = model.fit(train_input, train_target, epochs=50)

# Predict
predictions = model.predict(test_input)
```

## Models

### DeepSequence (Custom Architecture) ⭐
**Original contribution** - A Prophet-inspired deep learning architecture featuring:
- Seasonal decomposition modules (weekly, monthly, yearly)
- Trend, regressor, and holiday components with attention mechanisms
- Embedding layers for categorical features (SKU identifiers, clusters)
- TabNet encoder for feature selection
- Cross-layer attention for feature interactions
- Intermittent demand handler for sparse forecasting

**Architecture Highlights**:
- Modular design with separate seasonal and regression components
- Configurable hidden layers and activation functions
- L1 regularization for feature selection
- Early stopping and model checkpointing
- Support for exogenous variables (price, holidays, clusters)

### LightGBM Models
- **Cluster-based**: Groups similar SKUs and forecasts by cluster
- **Zero/Non-zero distance**: Handles intermittent demand patterns
- Lag features and distance-to-zero variables

### Baseline
- Naive shift-7 method for comparison
- Benchmark for model performance evaluation

## Results

### Performance Summary

DeepSequence with TabNet and Cross-Layer integration achieves superior performance on highly intermittent demand patterns.

| Model | Typical Use Case | Test MAE |
|-------|-----------------|----------|
| **DeepSequence** | High-volume, intermittent demand | 0.1312 |
| **LightGBM** | Standard forecasting | 0.5580 |

**📊 Detailed Comparison**: See [PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md) for comprehensive analysis.

## Data

**Note**: Sample data files are not included in this repository. 

To use this project with your own data:
1. Prepare your time series data with columns: `ds` (date), `SKU_ID` (SKU identifier), `Quantity` (target)
2. Add exogenous variables (optional): price, clusters, holidays, etc.
3. Follow the demo notebook for complete workflow
4. See `ARCHITECTURE.md` for detailed data requirements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact

Mritunjay Kumar - [GitHub](https://github.com/mkuma93)

## Citation

If you use DeepSequence or find this work helpful, please cite:

```
@misc{deepsequence,
  author = {Mritunjay Kumar},
  title = {DeepSequence: A Prophet-Inspired Deep Learning Architecture with TabNet and Cross-Layer Attention for SKU-Level Forecasting},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mkuma93/forecasting}
}
```

## Acknowledgments

- **DeepSequence**: Original architecture designed by Mritunjay Kumar, inspired by Facebook's Prophet
- Built for retail SKU-level forecasting with intermittent demand patterns
- Combines deep learning with seasonal decomposition methodology
- Enhanced with TabNet feature selection and Cross-Layer attention mechanisms
