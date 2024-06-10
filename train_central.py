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


# define a dataset class
class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, subset="train", eval_size=0.2):
        self.transforms = transforms
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # Split the images into a training set and an evaluation set
        split_idx = int((1.0 - eval_size) * len(imgs))
        if subset == "train":
            self.imgs = imgs[:split_idx]
        elif subset == "eval":
            self.imgs = imgs[split_idx:]
        else:
            raise ValueError("subset must be 'train' or 'eval'")

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if "real" in img_path.split("/")[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# Define hyperparameters
args = args_parser()
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

# Split the dataset
train_dataset = DogCat(r"../test", transforms=transform, subset='train')
eval_dataset = DogCat(r"../test", transforms=transform, subset='eval')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = CVAE_imagenet(d=64, k=128, num_classes=2)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs, _, _, _ = model(images)
        # TODO: I may need to design loss according to the paper.
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _, _, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, "
            f"Validation Accuracy: {100 * correct / total:.2f}%"
        )

# Save the model
torch.save(model.state_dict(), "../pretrained/retrain_model.pth")
