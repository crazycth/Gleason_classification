import torch
from torch.utils.data import DataLoader,dataset
from torch.utils.data import sampler,TensorDataset
import torchvision.datasets as dset
from torchvision import transforms
import torchvision
from pprint import pprint
import os
import cv2
import numpy as np
from PIL import Image
import random
import pandas as pd

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class Date(dataset.Dataset):
    def __init__(self,label_dic,train=True,transform_train = None , transform_val = None , root = None):
        self.root = root
        super(Date,self).__init__()
        self.dic = label_dic
        self.train = train
        self.data = os.listdir(self.root)
        random.shuffle(self.data)
        self.len = len(self.data)
        self.transform_train = transform_train
        self.transform_val = transform_val

    def __getitem__(self, index):
        name = self.data[index]
        data = cv2.imread(self.root+"/"+name)
        transform = self.transform_train if self.train else self.transform_val
        data = transform(data)
        label = self.dic[name[:12]]
        return data,label

    def __len__(self):
        return self.len

transform_train = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomResizedCrop(224,scale=(0.08,1.0),ratio=(3.0/4.0,4.0/3.0)),
    # transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
)

transform_val = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((224,224)),
    # transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_loader(batch_size=16,root=""):
    """
    :return: loader_train , loader_val
    """
    data = pd.read_excel("tem_10.xlsx")
    dic = dict(zip(data['cases'].values, data['tag'].values))
    data_train = Date(dic,True,transform_train,transform_val,root=root)
    data_val = Date(dic,False,transform_train,transform_val,root=root)
    total_num = len(os.listdir(root))
    train_num = int(total_num * 0.9)
    val_num = int(total_num * 0.1)
    loader_train = DataLoader(data_train,batch_size=batch_size,sampler=ChunkSampler(train_num,0),drop_last=True)
    loader_val = DataLoader(data_val,batch_size=batch_size,sampler=ChunkSampler(val_num,train_num),drop_last=True)
    return loader_train ,loader_val


def get_random_loader(batch_size=16):
    data = pd.read_excel("tem_10.xlsx")
    dic = dict(zip(data['cases'].values, data['tag'].values))
    Full_data = Date(dic,True,transform_train,transform_val)
    train_size = int(len(Full_data) * 0.8)
    test_size = len(Full_data) - train_size
    data_train , data_test = torch.utils.data.random_split(Full_data,[train_size,test_size])
    data_train.train = True
    data_test.train = False
    loader_train = DataLoader(data_train, batch_size=batch_size)
    loader_val = DataLoader(data_test, batch_size=batch_size)
    return loader_train, loader_val

def show_pic(img):
    cv2.imshow("test", img)
    cv2.waitKey()
    cv2.destroyAllWindows()  # important part!


def to_pic(pic):
    beard = transforms.Compose([torchvision.transforms.ToPILImage(mode="RGB")])
    way = Image.fromarray(pic)
    return way


if __name__ == '__main__':
    #dic = {'TCGA-EJ-5525': 1, 'TCGA-2A-A8VX': 1, 'TCGA-EJ-5517': 0, 'TCGA-H9-A6BX': 0, 'TCGA-VN-A88R': 1, 'TCGA-CH-5752': 1, 'TCGA-YL-A8SF': 1, 'TCGA-2A-A8VL': 0, 'TCGA-G9-A9S0': 1,'TCGA-EJ-AB20': 0}
    loader_train , loader_val = get_loader(batch_size=50)
    data = (loader_train.__iter__().next()[0][0])
    print(data)
