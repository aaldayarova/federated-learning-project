import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_mlp_model, get_model_parameters, set_model_parameters
from data_preprocessing import load_and_preprocess_data, simulate_federated_data_split

class MenstrualCycleClient(fl.client.NumPyClient):
    """Flower client for menstrual cycle length prediction."""
    
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray, 
        X_test: np.ndarray, y_test: np.ndarray, input_dim: int):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Create model
        self.model = create_mlp_model(input_dim)
        
        print(f"Client {client_id} initialized with {len(X_train)} training samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters."""
        return [layer.get_weights() for layer in self.model.layers if len(layer.get_weights()) > 0]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters."""
        layer_idx = 0
        for layer in self.model.layers:
            if len(layer.get_weights()) > 0:
                layer.set_weights(parameters[layer_idx])
                layer_idx += 1
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on the client's data."""
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Get training configuration
        epochs = config.get("epochs", 5)
        batch_size = config.get("batch_size", 32)
        
        # Train model
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        # Get updated parameters
        updated_parameters = self.get_parameters({})
        
        # Return updated parameters, number of examples, and metrics
        return updated_parameters, len(self.X_train), {
            "train_loss": float(history.history["loss"][-1]),
            "train_mae": float(history.history["mean_absolute_error"][-1])
        }
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model on the client's test data."""
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Evaluate model
        loss, mae, mse = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        return float(loss), len(self.X_test), {
            "test_mae": float(mae),
            "test_mse": float(mse)
        }

def main():
    """Main function to run the federated client."""
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client-id', type=int, default=0, help='Client ID')
    parser.add_argument('--data-path', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--server-address', type=str, default='localhost:8080', help='Server address')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(
        args.data_path
    )
    
    # Simulate federated data split (in practice, each client would have their own data)
    print("Simulating federated data split...")
    client_data = simulate_federated_data_split(X_train, y_train, num_clients=3)
    
    # Get this client's data
    if args.client_id >= len(client_data):
        print(f"Error: Client ID {args.client_id} is out of range. Available clients: 0-{len(client_data)-1}")
        return
    
    X_train_client, y_train_client = client_data[args.client_id]
    
    # Create client
    client = MenstrualCycleClient(
        client_id=args.client_id,
        X_train=X_train_client,
        y_train=y_train_client,
        X_test=X_test,
        y_test=y_test,
        input_dim=X_train.shape[1]
    )
    
    # Start client
    print(f"Starting client {args.client_id}...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )

if __name__ == "__main__":
    main()