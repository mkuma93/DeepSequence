"""
Sequential Component Combination for DeepSequence.

Implements sequential combination of components following the pattern:
T → T+S → (T+S)+H → (T+S+H)+R

Each combination supports additive and multiplicative modes:
- Additive: output = previous + component
- Multiplicative: output = previous × (1 + component)
"""

import tensorflow as tf
from tensorflow.keras import layers


class SeasonalCombination:
    """
    Combines trend output with seasonal component.
    
    Pattern: T → T+S
    
    Additive mode: T + S
    Multiplicative mode: T × (1 + S)
    """
    
    def __init__(self, mode: str = 'additive'):
        """
        Initialize seasonal combination.
        
        Args:
            mode: 'additive' or 'multiplicative'
        """
        if mode not in ['additive', 'multiplicative']:
            raise ValueError(
                f"mode must be 'additive' or 'multiplicative', got '{mode}'"
            )
        self.mode = mode
    
    def combine(self, trend_output, seasonal_model):
        """
        Combine trend output with seasonal model.
        
        Args:
            trend_output: Output tensor from trend component
            seasonal_model: Seasonal model (Keras Model)
            
        Returns:
            Combined output tensor (T+S or T×(1+S))
        """
        seasonal_output = seasonal_model.output
        
        if self.mode == 'additive':
            combined = layers.Add(name='seasonal_combination_add')([
                trend_output, seasonal_output
            ])
        else:
            # Multiplicative: T × (1 + S)
            seasonal_plus_one = layers.Lambda(
                lambda x: x + tf.constant(1.0),
                name='seasonal_plus_one'
            )(seasonal_output)
            combined = layers.Multiply(name='seasonal_combination_multiply')([
                trend_output, seasonal_plus_one
            ])
        
        return combined


class HolidayCombination:
    """
    Combines seasonal+trend output with holiday component.
    
    Pattern: (T+S) → (T+S)+H
    
    Additive mode: (T+S) + H
    Multiplicative mode: (T+S) × (1 + H)
    """
    
    def __init__(self, mode: str = 'additive'):
        """
        Initialize holiday combination.
        
        Args:
            mode: 'additive' or 'multiplicative'
        """
        if mode not in ['additive', 'multiplicative']:
            raise ValueError(
                f"mode must be 'additive' or 'multiplicative', got '{mode}'"
            )
        self.mode = mode
    
    def combine(self, seasonal_combined_output, holiday_model):
        """
        Combine seasonal-trend output with holiday model.
        
        Args:
            seasonal_combined_output: Output tensor from seasonal
                combination (T+S)
            holiday_model: Holiday model (Keras Model)
            
        Returns:
            Combined output tensor ((T+S)+H or (T+S)×(1+H))
        """
        holiday_output = holiday_model.output
        
        if self.mode == 'additive':
            combined = layers.Add(name='holiday_combination_add')([
                seasonal_combined_output, holiday_output
            ])
        else:
            # Multiplicative: (T+S) × (1 + H)
            holiday_plus_one = layers.Lambda(
                lambda x: x + tf.constant(1.0),
                name='holiday_plus_one'
            )(holiday_output)
            combined = layers.Multiply(name='holiday_combination_multiply')([
                seasonal_combined_output, holiday_plus_one
            ])
        
        return combined


class RegressorCombination:
    """
    Combines seasonal+trend+holiday output with regressor component.
    
    Pattern: (T+S+H) → (T+S+H)+R
    
    Additive mode: (T+S+H) + R
    Multiplicative mode: (T+S+H) × (1 + R)
    """
    
    def __init__(self, mode: str = 'additive'):
        """
        Initialize regressor combination.
        
        Args:
            mode: 'additive' or 'multiplicative'
        """
        if mode not in ['additive', 'multiplicative']:
            raise ValueError(
                f"mode must be 'additive' or 'multiplicative', got '{mode}'"
            )
        self.mode = mode
    
    def combine(self, holiday_combined_output, regressor_model):
        """
        Combine holiday-seasonal-trend output with regressor model.
        
        Args:
            holiday_combined_output: Output tensor from holiday
                combination ((T+S)+H)
            regressor_model: Regressor model (Keras Model)
            
        Returns:
            Combined output tensor (final forecast: (T+S+H)+R or (T+S+H)×(1+R))
        """
        regressor_output = regressor_model.output
        
        if self.mode == 'additive':
            combined = layers.Add(name='regressor_combination_add')([
                holiday_combined_output, regressor_output
            ])
        else:
            # Multiplicative: (T+S+H) × (1 + R)
            regressor_plus_one = layers.Lambda(
                lambda x: x + tf.constant(1.0),
                name='regressor_plus_one'
            )(regressor_output)
            combined = layers.Multiply(name='regressor_combination_multiply')([
                holiday_combined_output, regressor_plus_one
            ])
        
        return combined


class SequentialCombiner:
    """
    Orchestrates full sequential combination flow.
    
    Complete pattern: T → T+S → (T+S)+H → (T+S+H)+R
    
    This class simplifies the process by managing all combination steps
    in sequence with a single mode configuration.
    """
    
    def __init__(self, mode: str = 'additive'):
        """
        Initialize sequential combiner.
        
        Args:
            mode: 'additive' or 'multiplicative' (applies to all combinations)
        """
        self.mode = mode
        self.seasonal_combiner = SeasonalCombination(mode=mode)
        self.holiday_combiner = HolidayCombination(mode=mode)
        self.regressor_combiner = RegressorCombination(mode=mode)
    
    def combine_all(self,
                    trend_output,
                    seasonal_model=None,
                    holiday_model=None,
                    regressor_model=None):
        """
        Sequentially combine all components.
        
        Args:
            trend_output: Output tensor from trend component
            seasonal_model: Seasonal model (optional)
            holiday_model: Holiday model (optional)
            regressor_model: Regressor model (optional)
            
        Returns:
            Final combined output tensor
        """
        current_output = trend_output
        
        # Step 1: T → T+S
        if seasonal_model is not None:
            current_output = self.seasonal_combiner.combine(
                current_output, seasonal_model
            )
        
        # Step 2: (T+S) → (T+S)+H
        if holiday_model is not None:
            current_output = self.holiday_combiner.combine(
                current_output, holiday_model
            )
        
        # Step 3: (T+S+H) → (T+S+H)+R
        if regressor_model is not None:
            current_output = self.regressor_combiner.combine(
                current_output, regressor_model
            )
        
        return current_output
