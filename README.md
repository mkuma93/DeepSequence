# DeepSequence PWL

A general-purpose time series forecasting model with optional intermittent demand handling.

## Features

- **General-purpose architecture**: Works for both sparse/intermittent and continuous demand forecasting
- **Optional intermittent handling**: Enable/disable two-stage prediction via single parameter
- **Specialized components**: Trend, Seasonal, Holiday (PWL+Lattice), Regressor
- **Fixed additive combination**: Simple and interpretable component aggregation
- **Efficient**: 86% parameter savings when intermittent handling disabled
- **Validated performance**: Test MAE 0.0772 for 98.6% zero rate data

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/mkuma93/DeepSequence.git

# Or clone and install in development mode
git clone https://github.com/mkuma93/DeepSequence.git
cd DeepSequence
pip install -e .
```

## Project Structure

```
DeepSequence/
├── src/
│   └── deepsequence_pwl/     # Main package
│       ├── model.py           # Core model
│       ├── trend_component.py
│       ├── seasonal_component.py
│       ├── holiday_component.py
│       ├── regressor_component.py
│       └── ...
├── examples/
│   └── DeepSequence_PWL_Demo.ipynb  # Complete demo
├── tests/                     # Unit tests (coming soon)
├── setup.py                   # Package setup
├── pyproject.toml            # Modern Python packaging
└── requirements.txt          # Dependencies
```

## Quick Start

```python
from deepsequence_pwl import DeepSequencePWL

# For sparse/intermittent demand (default)
model = DeepSequencePWL(
    num_skus=100,
    n_features=10,
    enable_intermittent_handling=True  # Two-stage prediction
)

# For continuous demand forecasting
model = DeepSequencePWL(
    num_skus=100,
    n_features=10,
    enable_intermittent_handling=False  # Direct forecast, 86% fewer params
)

# Build and train
main_model, trend_model, seasonal_model, holiday_model, regressor_model = model.build_model()
main_model.compile(optimizer='adam', loss='mae')
main_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=32)
```

## Documentation

- **Package Documentation**: `src/deepsequence_pwl/README.md`
- **Demo Notebook**: `examples/DeepSequence_PWL_Demo.ipynb`
- **API Reference**: See docstrings in source code

## Development

```bash
# Clone repository
git clone https://github.com/mkuma93/DeepSequence.git
cd DeepSequence

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests (coming soon)
pytest tests/
```

All experimental work and additional features are maintained in the `develop` branch.

## License

See LICENSE file for details.
