"""
Regressor Component for DeepSequence.
Handles trend, exogenous variables, and contextual features.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Optional

# Setup precision
def _setup_precision():
    """Configure dtype and mixed precision based on available device."""
    try:
        mps_devices = tf.config.list_physical_devices('GPU')
        if mps_devices:
            from tf_keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            return tf.float32
    except Exception:
        pass
    return tf.float32

DTYPE = _setup_precision()

# Try tf_keras first (Keras 2), fallback to tensorflow.keras
LATTICE_AVAILABLE = False
try:
    import tensorflow_lattice as tfl
    import tf_keras as keras
    from tf_keras.layers import Input, Dense, Embedding, Flatten, Dropout, Concatenate
    from tf_keras.models import Model
    LATTICE_AVAILABLE = True
except ImportError:
    from tf_keras.layers import Input, Dense, Embedding, Flatten, Dropout, Concatenate
    from tf_keras.models import Model


class RegressorComponent:
    """
    Regression component that models trend and exogenous variables.
    Uses recurrent layers and lattice layers for constraint handling.
    """
    
    def __init__(self,
                 ts: pd.DataFrame,
                 exog: pd.DataFrame,
                 target: List[str],
                 id_var: str,
                 categorical_var: List[str],
                 context_variable: List[str],
                 constraint: List = None,
                 embed_size: int = 50,
                 lat_unit: int = 4,
                 lattice_size: int = 4,
                 hidden_unit: int = 4,
                 hidden_act='relu',
                 output_act='linear',
                 hidden_layer: int = 1,
                 drop_out: float = 0.1,
                 L1: float = 0.01,
                 rnge: float = 0.8,
                 use_pwl: bool = False,
                 n_pwl_keypoints: int = 10):
        """
        Initialize regressor component.
        
        Args:
            ts: Time series DataFrame
            exog: Exogenous variables DataFrame
            target: List of target column names
            id_var: ID variable column name
            categorical_var: List of categorical variable names
            context_variable: List of context variable names (continuous)
            constraint: List of constraints for lattice layers
            embed_size: Embedding size
            lat_unit: Lattice units
            lattice_size: Lattice size
            hidden_unit: Hidden layer units
            hidden_act: Hidden layer activation
            output_act: Output activation
            hidden_layer: Number of hidden layers
            drop_out: Dropout rate
            L1: L1 regularization strength
            rnge: Range parameter for lattice
            use_pwl: [OPTIONAL] Apply PWL to context variables (default: False)
                    Set True to learn non-linear effects for continuous regressors
            n_pwl_keypoints: PWL keypoints per context variable (default: 10)
        """
        self.ts = ts
        self.exog = exog
        self.target = target
        self.id_var = id_var
        self.categorical_var = categorical_var
        self.context_variable = context_variable
        self.constraint = constraint if constraint else [None] * len(context_variable)
        self.embed_size = embed_size
        self.lat_unit = lat_unit
        self.lattice_size = lattice_size
        self.hidden_unit = hidden_unit
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.hidden_layer = hidden_layer
        self.drop_out = drop_out
        self.L1 = L1
        self.rnge = rnge
        self.use_pwl = use_pwl
        self.n_pwl_keypoints = n_pwl_keypoints
        
        self.combined_reg_model = None
        self.input_names = []
        
    def reg_model(self, id_input: Optional[tf.Tensor] = None):
        """
        Build regression component neural network model.
        
        Args:
            id_input: Optional ID input tensor from seasonal model
        """
        inputs = []
        embeddings = []
        
        # ID input: use shared from Trend or create new
        n_ids = self.ts[self.id_var].nunique()
        embed_dim = min(self.embed_size, n_ids // 2 + 1)
        
        if id_input is None:
            # Create new ID input
            id_in = Input(shape=(1,), name='id')
            self.input_names.append('id')
        else:
            # Use shared ID input from Trend component
            id_in = id_input
        
        # Always add to inputs (needed for Model graph tracing)
        inputs.append(id_in)
        
        # Create ID embedding
        id_embed = Embedding(
            n_ids + 1,
            embed_dim,
            input_length=1,
            name='regressor_id_embed'
        )(id_in)
        id_embed = Flatten()(id_embed)
        embeddings.append(id_embed)
        
        # Categorical variable embeddings
        for cat_var in self.categorical_var:
            n_unique = self.exog[cat_var].nunique()
            cat_in = Input(shape=(1,), name=cat_var)
            cat_embed = Embedding(n_unique + 1, 
                                 min(self.embed_size, n_unique),
                                 name=f'{cat_var}_embed')(cat_in)
            cat_embed = Flatten()(cat_embed)
            inputs.append(cat_in)
            self.input_names.append(cat_var)
            embeddings.append(cat_embed)
        
        # Context variables (continuous) - with optional PWL
        context_features = []
        if self.use_pwl and LATTICE_AVAILABLE and len(self.context_variable) > 0:
            print(f"Regressor: Applying PWL to {len(self.context_variable)} context variables")
            
            for ctx_var in self.context_variable:
                ctx_in = Input(shape=(1,), name=ctx_var)
                inputs.append(ctx_in)
                self.input_names.append(ctx_var)
                
                # Get value range for this variable
                ctx_min = float(self.exog[ctx_var].min())
                ctx_max = float(self.exog[ctx_var].max())
                
                # PWL calibration
                calibrator = tfl.layers.PWLCalibration(
                    input_keypoints=np.linspace(
                        ctx_min, ctx_max, num=self.n_pwl_keypoints
                    ).astype(DTYPE.as_numpy_dtype),
                    dtype=DTYPE,
                    output_min=0.0,
                    output_max=1.0,
                    monotonicity='none',
                    kernel_regularizer=tf.keras.regularizers.l1(self.L1),
                    name=f'pwl_{ctx_var}'
                )(ctx_in)
                context_features.append(calibrator)
        else:
            # No PWL: use context variables directly
            for ctx_var in self.context_variable:
                ctx_in = Input(shape=(1,), name=ctx_var)
                inputs.append(ctx_in)
                self.input_names.append(ctx_var)
                context_features.append(ctx_in)
        
        # Concatenate embeddings and context
        all_features = embeddings + context_features
        if len(all_features) > 1:
            x = Concatenate()(all_features)
        else:
            x = all_features[0]
        
        # Hidden layers
        for i in range(self.hidden_layer):
            x = Dense(
                self.hidden_unit,
                activation=self.hidden_act,
                kernel_regularizer=tf.keras.regularizers.l1(self.L1),
                name=f'regressor_hidden_{i}'
            )(x)
            x = Dropout(self.drop_out)(x)
        
        # Lattice layer (if available)
        if LATTICE_AVAILABLE and self.lattice_size > 0:
            try:
                lattice_sizes = [self.lattice_size] * self.lat_unit
                x = tfl.layers.Lattice(lattice_sizes=lattice_sizes,
                                       output_min=0.0,
                                       output_max=self.rnge,
                                       kernel_regularizer=tfl.regularizers.Laplacian(self.L1))(x)
            except Exception:
                # Fall back to dense layer if lattice fails
                x = Dense(
                    self.lat_unit,
                    activation=self.hidden_act,
                    kernel_regularizer=tf.keras.regularizers.l1(self.L1)
                )(x)
        
        # Output layer
        reg_output = Dense(1, activation=self.output_act, name='regressor_output')(x)
        
        self.combined_reg_model = Model(inputs=inputs, outputs=reg_output, 
                                        name='regressor_component')
        
        return self.combined_reg_model
    
    def get_input_data(self, ts: pd.DataFrame, exog: pd.DataFrame) -> dict:
        """
        Prepare input data for the regressor model.
        
        Args:
            ts: Time series DataFrame with id_var column
            exog: Exogenous variables DataFrame
            
        Returns:
            Dictionary mapping input names to numpy arrays
        """
        input_dict = {}
        
        # Get all input names from the model
        for input_layer in self.combined_reg_model.inputs:
            input_name = input_layer.name.split(':')[0]  # Remove ':0' suffix
            
            if input_name == f'{self.id_var}_regressor':
                input_dict[input_name] = ts[self.id_var].values
            elif input_name in self.categorical_var:
                input_dict[input_name] = exog[input_name].values
            elif input_name in self.context_variable:
                input_dict[input_name] = exog[input_name].values
        
        return input_dict
