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
import timm
import cv2
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
    print(model)
    in_channel = model.head.in_features
    model.head = torch.nn.Linear(in_channel,classes,True)
    return model

if __name__ == '__main__':
    get_swin_transformer()