import os
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image
from src.models import CVAE_imagenet
import torch.multiprocessing as mp
from torch.utils import data
from options import args_parser


# Define your dataset class
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


def main_worker(gpu, ngpus_per_node, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=gpu,
    )
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    # Compute local batch size for each GPU
    assert (
        args.global_bs % args.world_size == 0
    ), "Global batch size must be divisible by the number of GPUs"
    local_batch_size = args.global_bs // args.world_size

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((296, 296)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Split the dataset
    train_dataset = DogCat(r"../test", transforms=transform, subset="train")
    eval_dataset = DogCat(r"../test", transforms=transform, subset="eval")

    # Create DataLoader with DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=eval_sampler,
    )

    # Initialize model and wrap with DDP
    model = CVAE_imagenet(d=64, k=128, num_classes=2).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs, _, _, _ = model(images)
            # TODO: Design loss according to the paper
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _, _, _ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if gpu == 0:  # Print only from the process with rank 0
            print(
                f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}, "
                f"Validation Accuracy: {100 * correct / total:.2f}%"
            )

    # Save the model (only the process with rank 0)
    if gpu == 0:
        torch.save(model.state_dict(), "../pretrained/retrain_model_distributed.pth")

    dist.destroy_process_group()


if __name__ == "__main__":
    args = args_parser()
    args.world_size = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
