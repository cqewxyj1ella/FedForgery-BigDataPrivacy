"""fl_dp_sa: A Flower / PyTorch app."""

import os
from PIL import Image
from collections import OrderedDict
import torch
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 100
MIN_CLIENTS = 10
NORM = 5.0
BATCH_SIZE = 32


# define a dataset class
class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
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
        self.imgs = imgs

    def __getitem__(self, index):
        # if the index is out of range, the dataset will automatically wrap around
        index = index % len(self.imgs)
        img_path = self.imgs[index]
        label = 1 if "0" in img_path.split("/")[-2] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)


def load_data(bs, num_clients):
    transform = Compose(
        [
            Resize((296, 296)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = DogCat(
        r"/home/nfs/ella/BigDataPrivacy/kaggle_dataset/c23/train",
        transforms=transform,
    )
    test_dataset = DogCat(
        r"/home/nfs/ella/BigDataPrivacy/kaggle_dataset/c23/val",
        transforms=transform,
    )
    partition_size = len(train_dataset) // num_clients
    lengths = [partition_size] * num_clients
    train_dataset = data.Subset(train_dataset, range(sum(lengths)))
    test_dataset = data.Subset(test_dataset, range(sum(lengths)))
    train_ds = random_split(train_dataset, lengths, torch.Generator().manual_seed(42))
    test_ds = random_split(test_dataset, lengths, torch.Generator().manual_seed(42))
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    testloaders = []
    for i in range(num_clients):
        trainloaders.append(DataLoader(train_ds[i], batch_size=bs, shuffle=True))
        testloaders.append(DataLoader(test_ds[i], batch_size=bs))

    return trainloaders, testloaders


def train(net, trainloader, valloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for _ in range(epochs):
        for images, labels in tqdm(trainloader, "Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs, _, _, _ = net(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader):
    """Validate the model on the test set."""
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader, "Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs, _, _, _ = net(images.to(DEVICE))
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
