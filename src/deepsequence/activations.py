"""
Custom activation functions for DeepSequence.

Available activations:
- swish: x * sigmoid(x)
- mish: x * tanh(softplus(x))
- listh: x * tanh(x)
- intermittent: x / (e^(-x)) - designed for sparse/intermittent data
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def swish(x):
    """
    Swish activation function: x * sigmoid(x)
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return x * K.sigmoid(x)


def listh(x):
    """
    Lisht activation function: x * tanh(x)
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return x * K.tanh(x)


def mish(x):
    """
    Mish activation function: x * tanh(softplus(x))
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return x * K.tanh(K.softplus(x))


def intermittent(x):
    """
    Intermittent activation function: x / (e^(-x))
    
    Designed for intermittent demand forecasting where values can be zero or sparse.
    This function provides smooth gradients even for small values.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return x / (K.exp(-x))


# Dictionary of custom activations
CUSTOM_ACTIVATIONS = {
    'swish': swish,
    'mish': mish,
    'listh': listh,
    'intermittent': intermittent,
    'sigmoid': 'sigmoid',
    'relu': 'relu',
    'tanh': 'tanh'
}


def get_activation(name: str):
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function
    """
    if name in CUSTOM_ACTIVATIONS:
        return CUSTOM_ACTIVATIONS[name]
    else:
        return name  # Return string for standard Keras activations
