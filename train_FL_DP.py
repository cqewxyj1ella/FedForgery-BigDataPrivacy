import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import syft as sy  # PySyft


# Define your model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc1(x.view(-1, 28 * 28))


# Federated learning setup
hook = sy.TorchHook(torch)
clients = ["alice", "bob"]
client_workers = [sy.VirtualWorker(hook, id=name) for name in clients]

# Create a federated dataset
federated_train_loader = sy.FederatedDataLoader(..., shuffle=True)

# Model and optimizer
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Integrate Opacus
privacy_engine = PrivacyEngine(
    model,
    batch_size=64,
    sample_size=len(federated_train_loader.dataset),
    epochs=10,
    max_grad_norm=1.0,
)
privacy_engine.attach(optimizer)

# Training loop
for epoch in range(10):
    for batch in federated_train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(batch["data"]), batch["target"])
        loss.backward()
        optimizer.step()

    # Privacy accounting
    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(1e-5)
    print(f"Epoch {epoch} | ε = {epsilon:.2f}, δ = 1e-5")
