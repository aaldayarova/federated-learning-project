# Import the specific classes you need directly
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2


def create_mlp_model(input_dim=40): # Default to 40 features
    """Creates the MLP model based on the centralized baseline."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-3)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-3)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1) # Output layer for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def get_model_parameters(model):
    """
    Get model weights as a flat list of NumPy arrays, which is what Flower expects.
    """
    return model.get_weights()


def set_model_parameters(model, parameters):
    """
    Set model weights from a flat list of NumPy arrays, which is what Flower provides.
    """
    model.set_weights(parameters)