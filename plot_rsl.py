import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader
from options import args_parser
from src.models import CVAE_imagenet
from PIL import Image
from torchvision import transforms
import os
from torch.utils import data
import sys

# Output confusion matrix


def plot_loss(train_loss_avg, test_loss_avg, num_epochs):
    loss_train = train_loss_avg
    loss_val = test_loss_avg
    print(num_epochs)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_train, "g", label="Training loss")
    plt.plot(epochs, loss_val, "b", label="validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_accuracy(train_accuracy, test_accuracy, num_epochs):
    loss_train = train_accuracy
    loss_val = test_accuracy
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss_train, "g", label="Training accuracy")
    plt.plot(epochs, loss_val, "b", label="validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def print_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    print("True positive = ", cm[0][0])
    print("False positive = ", cm[0][1])
    print("False negative = ", cm[1][0])
    print("True negative = ", cm[1][1])
    print("\n")
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set_theme(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.ylabel("Actual label", size=20)
    plt.xlabel("Predicted label", size=20)
    plt.xticks(np.arange(2), ["Fake", "Real"], size=16)
    plt.yticks(np.arange(2), ["Fake", "Real"], size=16)
    plt.ylim([2, 0])
    plt.show()
    plt.tight_layout()
    plt.savefig("confusion_matrix_{}.png".format(name))
    calculated_acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print("Calculated Accuracy", calculated_acc * 100)


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
    # get array like pred, true
    pred = []
    true = []
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)

        batch_loss = criterion(outputs[0], labels)
        loss += batch_loss.item()

        # Prediction

        _, pred_labels = torch.max(outputs[0], 1)
        pred_labels = pred_labels.view(-1)
        pred.extend(pred_labels.cpu().numpy())
        true.extend(labels.cpu().numpy())

        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss, pred, true


sys.stdout = Logger(stream=sys.stdout)


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, subset="train", eval_size=0.2):
        self.transforms = transforms
        imgs = []
        for folder in os.listdir(root):
            imgs += [
                os.path.join(root, folder, img)
                for img in os.listdir(os.path.join(root, folder))
            ]
        self.imgs = imgs

    def __getitem__(self, index):
        img_path = self.imgs[index]
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
        r"../kaggle_dataset/c23/train", transforms=transform_test, subset="test"
    )  # testset

    # BUILD MODEL
    if args.model == "FedForgery":
        if args.dataset == "forgery_dataset":
            global_model = CVAE_imagenet(d=64, k=128, num_classes=2)
    else:
        exit("Error: unrecognized model")

    if args.gpu:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)
    path = "../pretrained/xception-b5690688.pth"
    print("Loading pre-trained model from: ", path)
    state_dict = torch.load(path)
    global_model.load_state_dict(state_dict)
    global_model.to(device)

    test_acc, test_loss, pred, true = test_inference(args, global_model, test_dataset)
    print("Test on", len(test_dataset), "samples")
    print("Test Accuracy: {:.2f}%".format(100 * test_acc))


# Confusion Matrix for Efficient_GRU_Model
print(confusion_matrix(true, pred))
print_confusion_matrix(true, pred, "xception")

# plot of loss and accuracy for ResNext_LSTM_Model
# plot_loss(train_loss_avg, test_loss_avg, len(train_loss_avg))
# plot_accuracy(train_accuracy, test_accuracy, len(train_accuracy))
