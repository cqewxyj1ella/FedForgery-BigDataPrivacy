from typing import List, Tuple
from collections import OrderedDict
import numpy as np
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import flwr as fl
from src.models import CVAE_imagenet
import torch


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
net = CVAE_imagenet(d=64, k=128, num_classes=2).to(DEVICE)


class SaveModelStrategy(FedAvg):
    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"../pretrained/fl_2clients/model_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


# Define strategy
strategy = SaveModelStrategy(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)


# Define config
config = ServerConfig(num_rounds=40)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )
