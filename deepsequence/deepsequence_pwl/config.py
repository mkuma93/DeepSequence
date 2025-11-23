"""
Configuration file for DeepSequence.
Contains default parameters and path configurations.
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
FORECAST_DIR = os.path.join(OUTPUT_DIR, 'forecasts')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)

# Model default parameters
DEFAULT_HORIZON = 8
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 500
DEFAULT_LEARNING_RATE = 0.001

# Seasonal component defaults
SEASONAL_PARAMS = {
    'hidden_layers': 1,
    'hidden_units': 4,
    'embed_size': 50,
    'dropout': 0.1,
    'l1_reg': 0.011,
    'hidden_activation': 'mish',
    'output_activation': 'swish'
}

# Regression component defaults
REGRESSOR_PARAMS = {
    'latent_units': 4,
    'lattice_size': 4,
    'hidden_units': 4,
    'hidden_layers': 1,
    'embed_size': 50,
    'dropout': 0.1,
    'l1_reg': 0.032,
    'hidden_activation': 'mish',
    'output_activation': 'listh',
    'range': 0.8
}

# Training defaults
TRAINING_PARAMS = {
    'patience': 10,
    'monitor': 'val_loss',
    'mode': 'min',
    'save_best_only': True
}
