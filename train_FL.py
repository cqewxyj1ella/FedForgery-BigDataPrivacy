from PIL import Image
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models import CVAE_imagenet
from options import args_parser
from copy import deepcopy

# Define hyperparameters
args = args_parser()
K = args.K  # Number of local data centers
batch_size = args.local_bs
epochs = args.epochs
learning_rate = args.lr

# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((296, 296)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


# define a dataset class
class DogCat(data.Dataset):
    def __init__(
        self, root, transforms=None, subset="train", eval_size=0.2, center=0, centers=1
    ):
        self.transforms = transforms
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # Split the images into a training set and an evaluation set
        split_idx = int((1.0 - eval_size) * len(imgs))
        if subset == "train":
            imgs = imgs[:split_idx]
            # Distribute the images across the local data centers
            imgs = sorted(imgs)
            imgs = imgs[center::centers]
        elif subset == "eval":
            imgs = imgs[split_idx:]
        else:
            raise ValueError("subset must be 'train' or 'eval'")

        self.imgs = imgs

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if "real" in img_path.split("/")[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# Initialize the global model
global_model = CVAE_imagenet(d=64, k=128, num_classes=2)

# Initialize the local models and their data loaders
local_models = [deepcopy(global_model) for _ in range(K)]
local_datasets = [
    DogCat(r"../test", transforms=transform, center=k, centers=K, subset="train")
    for k in range(K)
]
local_loaders = [
    DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for dataset in local_datasets
]
# Define a separate evaluation set for the global model
global_eval_dataset = DogCat(r"../test", transforms=transform, subset="eval")
global_eval_loader = DataLoader(
    global_eval_dataset, batch_size=batch_size, shuffle=False
)


# Move models to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global_model.to(device)
local_models = [model.to(device) for model in local_models]

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizers = [Adam(model.parameters(), lr=learning_rate) for model in local_models]

# Training loop
for epoch in range(epochs):
    for model, loader, optimizer in zip(local_models, local_loaders, optimizers):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs, _, _, _ = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Aggregate the local models' weights
    global_weights = {
        name: torch.zeros_like(param) for name, param in global_model.named_parameters()
    }
    total_images = sum(len(dataset) for dataset in local_datasets)
    for model, dataset in zip(local_models, local_datasets):
        weight = len(dataset) / total_images
        for name, param in model.named_parameters():
            global_weights[name] += weight * param

    # Update the global model's weights
    for name, param in global_model.named_parameters():
        param.data = global_weights[name]

    # Validation
    global_model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in global_eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _, _, _ = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
            f"Validation Accuracy: {100 * correct / total:.2f}%"
        )

    # Download the global model's weights to local models
    for model in local_models:
        model.load_state_dict(global_model.state_dict())

# Save the model
torch.save(global_model.state_dict(), "../pretrained/FL_model_CEloss.pth")
