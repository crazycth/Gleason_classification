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
import cv2
import torchvision
dtype = torch.float32
print("CUDA: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

def imshow(axis, inp):
    """Denormalize and show"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    axis.imshow(inp)

def blue(img):
    """
    :param img: 4-channel pic (R-0 G-1 B-2)
    :return: how blue it is
    """
    img = np.array(img)
    matrix1 = (B(img)*100)/np.max(np.ones_like(img),(R(img)+G(img)))
    matrix2 = 256/np.max(np.ones_like(img),(R(img)+G(img)+B(img)))
    matrix1 = matrix1 * matrix2
    return np.average(matrix1)

