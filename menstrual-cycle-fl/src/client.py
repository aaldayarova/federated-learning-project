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
        # Use the model's built-in function that returns a flat list
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters."""
        # Use the model's built-in function that accepts a flat list
        self.model.set_weights(parameters)
    
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
        updated_parameters = self.model.get_weights()
        
        # Return updated parameters, number of examples, and metrics
        return updated_parameters, len(self.X_train), {
            "train_loss": float(history.history["loss"][-1]),
            "train_mae": float(history.history["mae"][-1])
        }
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model on the client's test data."""
        # Set parameters received from server
        self.set_parameters(parameters)
        
        # Evaluate model
        loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        rmse = np.sqrt(loss)

        
        return float(loss), len(self.X_test), {
            "test_mae": float(mae),
             "rmse": float(rmse)
        }

def main():
    """Main function to run the federated client."""
    parser = argparse.ArgumentParser(description='Federated Learning Client')

    # We still need client-id and server-address
    parser.add_argument('--client-id', type=int, required=True, help='Client ID (e.g., 0, 1, 2)')
    parser.add_argument('--server-address', type=str, default='localhost:8080', help='Server address')

    args = parser.parse_args()

    # new data loading logic 
    print(f"Client {args.client_id}: Loading pre-processed data...")

    # Construct the file path for this client's specific data
    client_data_path = f"fl_artifacts/client_{args.client_id}_data.npz"

    try:
        # Load this client's training data
        client_data = np.load(client_data_path)
        X_train_client, y_train_client = client_data["X_train"], client_data["y_train"]

        # Load the global test set for evaluation
        # All clients will evaluate on the same hold-out test set
        test_data = np.load("fl_artifacts/global_test_set.npz")
        X_test, y_test = test_data["X_test"], test_data["y_test"]

    except FileNotFoundError as e:
        print(f"\nERROR: Could not load data files from 'fl_artifacts/'. ({e})")
        print("Please ensure you have run the 'prepare_data_for_fl.py' script first.\n")
        sys.exit(1)

    # Get the input dimension from the loaded training data
    input_dim = X_train_client.shape[1]

    # Create client
    client = MenstrualCycleClient(
        client_id=args.client_id,
        X_train=X_train_client,
        y_train=y_train_client,
        X_test=X_test,  # Use the global test set
        y_test=y_test,  # Use the global test set
        input_dim=input_dim
    )

    # Start client
    print(f"Starting client {args.client_id} (Input dim: {input_dim})...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )

if __name__ == "__main__":
    main()