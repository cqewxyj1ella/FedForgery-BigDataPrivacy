from PIL import Image
from options import args_parser
import warnings
from collections import OrderedDict
import os
from flwr.client import NumPyClient, ClientApp
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from tqdm import tqdm
from src.models import CVAE_imagenet


# Define hyperparameters
args = args_parser()
bs = args.local_bs
epochs = args.epochs
lr = args.lr
cid = args.cid
K = args.K


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader, "Training"):
            optimizer.zero_grad()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # Forward pass
            outputs, _, _, _ = net(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader, "Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs, _, _, _ = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# define a dataset class
class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, cid=0, num_clients=2):
        self.transforms = transforms
        # there are two folders under root: 0 and 1. 0 is real, 1 is fake
        # get imgs from both folders
        # divide the imgs into num_clients parts
        imgs = []
        for folder in os.listdir(root):
            imgs += [
                os.path.join(root, folder, img)
                for img in os.listdir(os.path.join(root, folder))
            ]
        self.imgs = imgs[cid::num_clients]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if "0" in img_path.split("/")[-2] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


def load_data():
    transform = Compose(
        [
            Resize((296, 296)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = DogCat(
        r"../kaggle_dataset/c23/train", transforms=transform, cid=cid, num_clients=K
    )
    trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataset = DogCat(
        r"../kaggle_dataset/c23/val", transforms=transform, cid=cid, num_clients=K
    )
    testloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    return trainloader, testloader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = CVAE_imagenet(d=64, k=128, num_classes=2).to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=epochs)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=client_fn(cid),
    )
