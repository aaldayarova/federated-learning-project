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
from data_preprocessing import load_and_preprocess_data

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

def server_evaluate(server_round: int, parameters, config) -> Optional[Tuple[float, Dict[str, float]]]:
    """Optional server-side evaluation function."""
    # This could be implemented if you have a global test set
    # For now, we rely on client-side evaluation
    return None

def main():
    """Main function to run the federated server."""
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--rounds', type=int, default=5, help='Number of federated rounds')
    parser.add_argument('--min-clients', type=int, default=2, help='Minimum number of clients')
    parser.add_argument('--min-available-clients', type=int, default=2, help='Minimum available clients')
    parser.add_argument('--server-address', type=str, default='0.0.0.0:8080', help='Server address')
    parser.add_argument('--data-path', type=str, required=True, help='Path to CSV data file for model initialization')
    
    args = parser.parse_args()
    
    # Load data to get input dimensions for model initialization
    print("Loading data to initialize model...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(
        args.data_path
    )
    
    # Create initial model to get parameters
    initial_model = create_mlp_model(X_train.shape[1])
    initial_parameters = [layer.get_weights() for layer in initial_model.layers if len(layer.get_weights()) > 0]
    
    def get_initial_parameters():
        return initial_parameters
    
    # Define strategy
    strategy = FedAvgStrategy(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_available_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda server_round: {
            "epochs": 5,
            "batch_size": 32,
        },
        on_evaluate_config_fn=lambda server_round: {},
    )
    
    print(f"Starting server on {args.server_address}")
    print(f"Model input dimension: {X_train.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"Running for {args.rounds} rounds with minimum {args.min_clients} clients")
    
    # Start server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()