import openslide
import torch
dtype = torch.float32
from select_pic import Select,Select_Blue
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

def pic_trans():
    """
    :param num_limit:
    :return:
    """
    for idx,pic_name in tqdm(enumerate(os.listdir("./svs_pic")),total=len(os.listdir("./svs_pic"))):
        if pic_name.startswith('.'):
            continue
        name = pic_name[:12]
        pic = openslide.OpenSlide("./svs_pic/"+pic_name)
        Select(pic,name,"./pic_save_1")


def main_train(fold=10):
    model = get_swin_transformer(2)
    optimizer = torch.optim.SGD(params=(para for para in model.parameters() if para.requires_grad == True),lr=0.001,weight_decay=0.1,momentum=0.9)
    #optimizer = torch.optim.Adam(params=(para for para in model.parameters() if para.requires_grad == True),lr=0.001,weight_decay=0.1,betas=(0.5,0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.8)
    for t in range(1):  
        loader_train , loader_val = get_loader(batch_size=32)
        train_part34(model,optimizer,loader_train,loader_val,300,50,device,scheduler)
    torch.save(model,"model")


if __name__ == '__main__':
    main_train()