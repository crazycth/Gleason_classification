import openslide
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
from pprint import pprint
from pandas import DataFrame
dtype = torch.float32
print("CUDA: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
from select_pic import Select

def init():
    """
    delete the file pic_save
    :return: None
    """
    os.system("rm -r ./pic_save")
    os.system("mkdir ./pic_save")

def pic_trans():
    """
    :return: dataframe contains pic2label
    """
    name2label=[]
    for idx,pic_name in enumerate(os.listdir("./svs_pic")):
        if pic_name.startswith('.'):
            continue
        name = pic_name[-16:-4]
        pic = openslide.OpenSlide("./svs_pic/"+pic_name)
        name2label.append((name,'label'))
        Select(pic,10,name)
    df = DataFrame(name2label,columns=['name','label'])
    df.set_index(['name'],inplace=True)
    return df

if __name__ == '__main__':
    init()
    df = pic_trans()
    pprint(df)