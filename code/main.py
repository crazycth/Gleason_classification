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
from tqdm import tqdm
dtype = torch.float32
from select_pic import Select
from DataLoader import get_loader
from net_work import *

print("CUDA: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    dic = dict()
    for idx,pic_name in tqdm(enumerate(os.listdir("./svs_pic")),total=len(os.listdir("./svs_pic"))):
        if pic_name.startswith('.'):
            continue
        name = pic_name[-16:-4]
        pic = openslide.OpenSlide("./svs_pic/"+pic_name)
        name2label.append((name,'label'))
        dic[name]= 0
        Select(pic,10,name)
    df = DataFrame(name2label,columns=['name','label'])
    df.set_index(['name'],inplace=True)
    return df,dic


if __name__ == '__main__':
    init()
    df,dic = pic_trans()
    loader_train , loader_val = get_loader(dic,batch_size=4)
    model = get_resnet_34(3)
    optimizer = torch.optim.SGD(params=(para for para in model.parameters() if para.requires_grad == False),lr=1e-4)
    train_part34(model,optimizer,loader_train,loader_val,1,10,device)
