"""fl_dp_sa: A Flower / PyTorch app."""

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import fixedclipping_mod, secaggplus_mod
from src.models import CVAE_imagenet
from DP.task import (
    DEVICE,
    NUM_CLIENTS,
    BATCH_SIZE,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
)
import sys

# Load model and data (simple CNN, CIFAR-10)
bs = BATCH_SIZE


# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, testloader) -> None:
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = CVAE_imagenet(d=64, k=128, num_classes=2)
        self.device = DEVICE
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        results = train(
            self.model, self.trainloader, self.testloader, epochs=1, device=DEVICE
        )
        return get_weights(self.model), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    trainloaders, testloaders = load_data(bs=bs, num_clients=NUM_CLIENTS)
    trainloader = trainloaders[int(cid)]
    testloader = testloaders[int(cid)]
    return FlowerClient(trainloader, testloader).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
        fixedclipping_mod,
    ],
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        # get cid from command line arguments
        client=client_fn(cid=sys.argv[1]),
        grpc_max_message_length=1024 * 1024 * 1024,
    )
