"""fl_dp_sa: A Flower / PyTorch app."""

from typing import List, Tuple
from collections import OrderedDict
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server.strategy import (
    DifferentialPrivacyClientSideFixedClipping,
    FedAvg,
)
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
import flwr as fl
from src.models import CVAE_imagenet
from DP.task import MIN_CLIENTS, NORM, DEVICE, get_weights
import torch
import numpy as np


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


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
            torch.save(
                net.state_dict(), f"../../pretrained/fl_dp/model_{server_round}.pth"
            )

        return aggregated_parameters, aggregated_metrics


# Initialize model parameters
ndarrays = get_weights(CVAE_imagenet(d=64, k=128, num_classes=2))
parameters = ndarrays_to_parameters(ndarrays)

# Define strategy
strategy = SaveModelStrategy(
    fraction_fit=0.2,
    fraction_evaluate=0.0,  # Disable evaluation for demo purpose
    min_fit_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    fit_metrics_aggregation_fn=weighted_average,
    initial_parameters=parameters,
)
strategy = DifferentialPrivacyClientSideFixedClipping(
    strategy, noise_multiplier=0.2, clipping_norm=NORM, num_sampled_clients=MIN_CLIENTS
)


app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    # Construct the LegacyContext
    context = LegacyContext(
        state=context.state,
        config=ServerConfig(num_rounds=1),
        strategy=strategy,
    )

    # Create the train/evaluate workflow
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=7,
            reconstruction_threshold=4,
        )
    )

    # Execute
    workflow(driver, context)


if __name__ == "__main__":
    # Define config
    config = ServerConfig(num_rounds=1)
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024,
    )
