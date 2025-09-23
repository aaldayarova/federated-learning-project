import tensorflow as tf
import numpy as np

keras = tf.keras
layers = tf.keras.layers

def create_mlp_model(input_dim, hidden_layers=[64, 32, 16], dropout_rate=0.3):
    """
    Create a Multi-Layer Perceptron (MLP) model for cycle length prediction.
    
    Args:
        input_dim (int): Number of input features
        hidden_layers (list): List of hidden layer sizes
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        tf.keras.Model: Compiled MLP model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(hidden_layers[0], activation='relu'),
        layers.Dropout(dropout_rate),
        
        # Additional hidden layers
        *[layer for i, units in enumerate(hidden_layers[1:]) 
        for layer in [layers.Dense(units, activation='relu'), layers.Dropout(dropout_rate)]],
        
        # Output layer (single neuron for regression)
        layers.Dense(1, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    
    return model

def get_model_parameters(model):
    """ Get model parameters as a list of numpy arrays."""
    return [layer.get_weights() for layer in model.layers if len(layer.get_weights()) > 0]

def set_model_parameters(model, parameters):
    """ Set model parameters from a list of numpy arrays."""
    layer_idx = 0
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            layer.set_weights(parameters[layer_idx])
            layer_idx += 1