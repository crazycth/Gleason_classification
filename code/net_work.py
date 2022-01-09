import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,dataset
from torch.utils.data import sampler,TensorDataset
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import timm
import cv2
import wandb
import torchvision
dtype = torch.float32
print("CUDA: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imshow(axis, inp):
    """Denormalize and show"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    axis.imshow(inp)

def get_resnet_34(classes=3):
    model = torchvision.models.resnet34(pretrained=True,progress=True)
    for para in model.parameters():
        para.requires_grad = False
    in_channel = model.fc.in_features
    model.fc = torch.nn.Linear(in_channel,classes,True)
    return model

def get_swin_transformer(classes=3):
    model = timm.models.swin_base_patch4_window7_224(pretrained=True)
    for para in model.parameters():
        para.requires_grad = False
    in_channel = model.head.in_features
    model.head = torch.nn.Linear(in_channel,classes,True)
    return model


def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc


def train_part34(model, optimizer, loader_train , loader_val , epochs=1 , print_every = 100,device = torch.device('cpu'),scheduler=None):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    wandb.init(project="medical",name="Swin_transformer")
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    count = 0
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            count = count + 1

            if count % print_every == 0:
                print('Iteration %d, loss = %.4f' % (count, loss.item()))
                acc_val = check_accuracy_part34(loader_val, model)
                acc_train = check_accuracy_part34(loader_train,model)
                wandb.log({"val_acc": acc_val, "train_acc": acc_train, "loss": loss})
                print()
        if scheduler is not None:
            scheduler.step()


if __name__ == '__main__':
    get_resnet_34(3)