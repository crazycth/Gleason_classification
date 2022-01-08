import openslide
import torch
dtype = torch.float32
from select_pic import Select
from DataLoader import get_loader,get_random_loader
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

def pic_trans(num_limit=10):
    """
    :param num_limit:
    :return:
    """
    for idx,pic_name in tqdm(enumerate(os.listdir("./svs_pic")),total=len(os.listdir("./svs_pic"))):
        if pic_name.startswith('.'):
            continue
        name = pic_name[:12]
        pic = openslide.OpenSlide("./svs_pic/"+pic_name)
        Select(pic,num_limit,name)


def main_train(fold=10):
    model = get_swin_transformer(3)
    optimizer = torch.optim.SGD(params=(para for para in model.parameters() if para.requires_grad == True),lr=0.001,weight_decay=0.1,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)
    for t in range(1):
        loader_train , loader_val = get_loader(batch_size=64)
        train_part34(model,optimizer,loader_train,loader_val,300,100,device,scheduler)


if __name__ == '__main__':
    init()
    pic_trans(20)
