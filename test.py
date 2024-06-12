#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
from torch.utils.data import DataLoader
import glob
from options import args_parser
from src.models import CVAE_imagenet
from PIL import Image
from torchvision import transforms
import os
from torch.utils import data
import sys
# import flwr as fl

sys.path.append(".")


class Logger(object):
    def __init__(self, filename="test.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def test_inference(args, model, test_dataset):
    """Returns the test accuracy and loss."""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = "cuda:1" if args.gpu else "cpu"

    criterion = nn.CrossEntropyLoss().to(device)
    print("criterion loaded")
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)

        batch_loss = criterion(outputs[0], labels)
        loss += batch_loss.item()

        # Prediction

        _, pred_labels = torch.max(outputs[0], 1)
        pred_labels = pred_labels.view(-1)

        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


sys.stdout = Logger(stream=sys.stdout)


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, subset="train", eval_size=0.2):
        self.transforms = transforms
        # ########################################################
        # if use hybrid dataset in the original paper
        # ########################################################
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # # Split the images into a training set and an evaluation set
        # split_idx = int((1.0 - eval_size) * len(imgs))
        # if subset == "train":
        #     self.imgs = imgs[:split_idx]
        # elif subset == "test":
        #     self.imgs = imgs[split_idx:]
        # else:
        #     raise ValueError("subset must be 'train' or 'test'")
        # ########################################################
        # if use kaggle dataset, splited into train, val and test
        # ########################################################
        imgs = []
        for folder in os.listdir(root):
            imgs += [
                os.path.join(root, folder, img)
                for img in os.listdir(os.path.join(root, folder))
            ]
        self.imgs = imgs

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # ########################################################
        # if use hybrid dataset in the original paper
        # label = 1 if "real" in img_path.split("/")[-1] else 0
        # ########################################################
        # if use kaggle dataset, splited into train, val and test
        label = 1 if "0" in img_path.split("/")[-2] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    args = args_parser()
    transform = transforms.Compose(
        [
            transforms.Resize((296, 296)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((296, 296)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_dataset = DogCat(
        r"../kaggle_dataset/c23/test", transforms=transform_test, subset="test"
    )  # testset

    # BUILD MODEL
    if args.model == "FedForgery":
        if args.dataset == "forgery_dataset":
            global_model = CVAE_imagenet(d=64, k=128, num_classes=2)
    else:
        exit("Error: unrecognized model")

    # ########################################################
    # fl load
    # ########################################################
    list_of_files = [fname for fname in glob.glob("../pretrained/fl_2clients/model_*")]
    # latest_round_file = max(list_of_files, key=os.path.getctime)
    # not choose latest round by time, but by round number
    round_number = [int(fname.split("_")[-1].split(".")[0]) for fname in list_of_files]
    latest_round_file = list_of_files[round_number.index(min(round_number))]
    if args.gpu:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)
    for file in list_of_files[1:-1]:
        model_file = file
        print("Loading pre-trained model from: ", model_file)
        state_dict = torch.load(model_file)
        global_model.load_state_dict(state_dict)
        # state_dict_ndarrays = [v.cpu().numpy() for v in global_model.state_dict().values()]
        # parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)
        # parameters can be further used to customize flower strategy
        # ########################################################
        # normal load
        # ########################################################
        # path = "../pretrained/retrain_central_model.pth"
        # global_model.load_state_dict(torch.load(path))
        
        global_model.to(device)

        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print("Test on", len(test_dataset), "samples")
        print("Test Accuracy: {:.2f}%".format(100 * test_acc))
