import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import sys
import os
from flwr.common import Metrics


# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_mlp_model

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of metrics."""
    # Calculate total number of examples
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    # Initialize result dictionary
    result = {}
    
    # Get all metric keys from the first client
    if metrics:
        metric_keys = metrics[0][1].keys()
        
        # Compute weighted average for each metric
        for key in metric_keys:
            weighted_sum = sum([num_examples * m[key] for num_examples, m in metrics])
            result[key] = weighted_sum / total_examples
    
    return result

class FedAvgStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy with evaluation metrics aggregation."""
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes] | Tuple[fl.server.client_proxy.ClientProxy, BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results from all clients."""
        
        if not results:
            return None, {}
        
        # Convert results to the format expected by weighted_average
        metrics_list = []
        losses = []
        
        for _, evaluate_res in results:
            metrics_list.append((evaluate_res.num_examples, evaluate_res.metrics))
            losses.append(evaluate_res.loss * evaluate_res.num_examples)
        
        # Calculate weighted average loss
        total_examples = sum([evaluate_res.num_examples for _, evaluate_res in results])
        avg_loss = sum(losses) / total_examples if total_examples > 0 else None
        
        # Calculate weighted average of other metrics
        avg_metrics = weighted_average(metrics_list)
        
        print(f"Round {server_round} - Evaluation Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        for key, value in avg_metrics.items():
            print(f"  Average {key}: {value:.4f}")
        
        return avg_loss, avg_metrics

def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    try:
        # Load the global test set from the artifacts directory
        test_data = np.load("fl_artifacts/global_test_set.npz")
        X_test, y_test = test_data["X_test"], test_data["y_test"]
    except FileNotFoundError:
        print("\nERROR: Global test set not found at 'fl_artifacts/global_test_set.npz'")
        print("Please run the 'prepare_data_for_fl.py' script first.\n")
        sys.exit(1)

    # The inner function that will be called by Flower each round
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        # Create a new model instance for evaluation
        model = create_mlp_model(input_dim=X_test.shape[1])
        # Set the global model weights sent by the server
        model.set_weights(parameters)
        # Evaluate the model on the global test set
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        rmse = np.sqrt(loss)

        print(f"Server-side evaluation round {server_round}")
        print(f"Loss: {loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Return the loss and a dictionary of metrics
        return loss, {"server_mae": mae, "server_rmse": rmse}

    return evaluate

def main():
    """Main function to run the federated server."""
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--rounds', type=int, default=5, help='Number of federated rounds')
    parser.add_argument('--min-clients', type=int, default=2, help='Minimum number of clients')
    parser.add_argument('--min-available-clients', type=int, default=2, help='Minimum available clients')
    parser.add_argument('--server-address', type=str, default='0.0.0.0:8080', help='Server address')
    
    args = parser.parse_args()
    
    
    # Create initial model to get parameters
    initial_model = create_mlp_model(input_dim=40)
    initial_parameters = initial_model.get_weights()
    
    def get_initial_parameters():
        return initial_parameters
    
    # Define strategy
    strategy = FedAvgStrategy(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(),  # Pass the server-side evaluation function
        on_fit_config_fn=lambda server_round: {
            "epochs": 5,
            "batch_size": 32,
        },
        on_evaluate_config_fn=lambda server_round: {},
    )
    
    print(f"Starting server on {args.server_address}")
    print("Model input dimension: 40 (based on pre-selected features)")
    print(f"Running for {args.rounds} rounds with minimum {args.min_clients} clients")
    
    # Start server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()