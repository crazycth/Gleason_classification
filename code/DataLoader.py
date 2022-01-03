import torch
from torch.utils.data import DataLoader,dataset
from torch.utils.data import sampler,TensorDataset
import torchvision.datasets as dset
from torchvision import transforms
from pprint import pprint
import os
import cv2

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
    def __init__(self,label_dic,train=True,transform_train = None , transform_val = None):
        super(Date,self).__init__()
        self.dic = label_dic
        self.train = train
        self.data = os.listdir("./pic_save")
        self.len = len(self.data)
        self.transform_train = transform_train
        self.transform_val = transform_val

    def __getitem__(self, index):
        name = self.data[index]
        svs_from = name.split('_')[0]
        label = self.dic[svs_from]
        data = cv2.imread("./pic_save/"+name)
        transform = self.transform_train if self.train else self.transform_val
        data = transform(data)
        label = 0
        return data,label

    def __len__(self):
        return len(self)

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224,scale=(0.08,1.0),ratio=(3.0/4.0,4.0/3.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
)

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_loader(dic,batch_size=16):
    """
    :return: loader_train , loader_val
    """
    data_train = Date(dic,True,transform_train,transform_val)
    data_val = Date(dic,False,transform_train,transform_val)
    total_num = len(os.listdir("./pic_save"))
    train_num = int(total_num * 0.8)
    val_num = int(total_num * 0.2)
    loader_train = DataLoader(data_train,batch_size=batch_size,sampler=ChunkSampler(train_num,0))
    loader_val = DataLoader(data_val,batch_size=batch_size,sampler=ChunkSampler(val_num,train_num))
    return loader_train , loader_val

def show_pic(img):
    cv2.imshow("test", img)
    cv2.waitKey()
    cv2.destroyAllWindows()  # important part!


if __name__ == '__main__':
    dic = {'14bd30742016': 'label', 'a40035025174': 'label', 'f14f35f54086': 'label'}
    loader_train , loader_val = get_loader(dic,batch_size=2)
    data = (loader_val.__iter__().next()[0][0])
    pprint(data.shape)


