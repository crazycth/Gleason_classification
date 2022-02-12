import torch
import numpy as np
from queue import PriorityQueue
from select_pic import *
import pandas as pd
from tqdm import tqdm
import openslide
import cv2
from torchvision import transforms
from pprint import pprint

transform_val = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((224,224)),
    # transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(root):
    model = torch.load(root,map_location=torch.device('cpu'))
    return model


def Select_Blue(img,num=40,name="init",save_root="./pic_save"):
    """
    :param img: Openslide_img
    :param num: the num of
    :param name: the name of file
    :return: the Sequential of img
    """
    que = PriorityQueue()
    for x in tqdm(range(0, img.level_dimensions[0][0] - 224, 224)):
        for y in range(0, img.level_dimensions[0][1] - 224, 224):
            im = img.read_region((x, y), 1, (224//4, 224//4))
            im = im.convert('RGB')
            valid = check_valid(im)
            valid_blue = blue(im)
            valid_red = red(im)
            valid_green = green(im)
            if valid <= 0.7 or valid_blue >= 100 or valid_red>=110 or valid>=0.99 or valid_green>=100:
                continue
            que.put((valid_blue, random.random(),x, y))
            while que.qsize() > num:
                que.get()
    count = 0
    pic_list = []
    while not que.empty():
        count = count + 1
        valid_blue , rd , x , y = que.get()
        im = img.read_region((x,y),0,(224,224))
        im = im.convert('RGB')
        pic_list.append(im)
    return pic_list


def predict(model,limit):
    data = pd.read_excel("tem_10.xlsx")
    dic = dict(zip(data['cases'].values, data['tag'].values))
    score_dic = dict()
    for pic in tqdm(os.listdir("./pic_save")):
        if pic.startswith("."): continue
        svs = pic[:12]
        if svs not in limit: continue
        label = dic[svs]
        pic_data = cv2.imread("./pic_save/"+pic)
        pic_data = transform_val(pic_data)
        pic_data = pic_data.reshape((1,3,224,224))
        score = model(pic_data)
        _,preds = score.max(1)
        score_dic[(svs,preds.item())] = 1 if (svs,preds.item()) not in score_dic else score_dic[(svs,preds.item())] + 1
    return score_dic


def get_limit():
    limit = []
    for pic in os.listdir("./svs_pic")[:20]:
        if pic.startswith("."): continue
        limit.append(pic[:12])
    return limit


def get_dict():
    #dict = {('TCGA-XJ-A9DQ', 0): 234, ('TCGA-KK-A8I9', 0): 65, ('TCGA-2A-A8VX', 1): 397, ('TCGA-J4-A67Q', 0): 249, ('TCGA-EJ-7782', 1): 417, ('TCGA-XJ-A9DQ', 1): 303, ('TCGA-G9-A9S0', 1): 489, ('TCGA-EJ-A46B', 1): 476, ('TCGA-J4-A67Q', 1): 414, ('TCGA-2A-A8VL', 1): 398, ('TCGA-VP-A87B', 1): 305, ('TCGA-H9-A6BX', 1): 508, ('TCGA-EJ-5517', 1): 453, ('TCGA-V1-A9OF', 0): 142, ('TCGA-YL-A8SF', 1): 437, ('TCGA-V1-A9OQ', 0): 323, ('TCGA-EJ-A46G', 1): 486, ('TCGA-CH-5752', 1): 165, ('TCGA-M7-A722', 1): 281, ('TCGA-2A-A8VX', 0): 39, ('TCGA-2A-A8VL', 0): 143, ('TCGA-KK-A8I9', 1): 126, ('TCGA-HC-7821', 1): 122, ('TCGA-EJ-A46G', 0): 58, ('TCGA-VP-A87B', 0): 40, ('TCGA-EJ-A46B', 0): 50, ('TCGA-HC-7077', 0): 3, ('TCGA-V1-A9OF', 1): 17, ('TCGA-HC-7077', 1): 30, ('TCGA-H9-A6BX', 0): 30, ('TCGA-V1-A9OQ', 1): 30, ('TCGA-M7-A722', 0): 14, ('TCGA-G9-A9S0', 0): 14, ('TCGA-EJ-5517', 0): 1}
    dict = {('TCGA-XJ-A9DQ', 0): 472, ('TCGA-KK-A8I9', 0): 70, ('TCGA-2A-A8VX', 1): 221, ('TCGA-J4-A67Q', 1): 482, ('TCGA-EJ-7782', 1): 235, ('TCGA-G9-A9S0', 1): 409, ('TCGA-EJ-A46B', 1): 410, ('TCGA-2A-A8VL', 0): 227, ('TCGA-VP-A87B', 1): 199, ('TCGA-2A-A8VX', 0): 215, ('TCGA-H9-A6BX', 1): 408, ('TCGA-EJ-5517', 0): 145, ('TCGA-V1-A9OF', 0): 53, ('TCGA-YL-A8SF', 1): 251, ('TCGA-V1-A9OQ', 1): 231, ('TCGA-EJ-5517', 1): 309, ('TCGA-EJ-7782', 0): 182, ('TCGA-EJ-A46G', 0): 318, ('TCGA-H9-A6BX', 0): 130, ('TCGA-CH-5752', 1): 57, ('TCGA-YL-A8SF', 0): 186, ('TCGA-2A-A8VL', 1): 314, ('TCGA-V1-A9OQ', 0): 122, ('TCGA-M7-A722', 1): 211, ('TCGA-G9-A9S0', 0): 94, ('TCGA-EJ-A46G', 1): 226, ('TCGA-J4-A67Q', 0): 181, ('TCGA-KK-A8I9', 1): 121, ('TCGA-VP-A87B', 0): 146, ('TCGA-HC-7821', 0): 62, ('TCGA-HC-7821', 1): 60, ('TCGA-CH-5752', 0): 108, ('TCGA-V1-A9OF', 1): 106, ('TCGA-XJ-A9DQ', 1): 65, ('TCGA-EJ-A46B', 0): 116, ('TCGA-M7-A722', 0): 84, ('TCGA-HC-7077', 1): 16, ('TCGA-HC-7077', 0): 17}
    return dict

# if __name__ == '__main__':
#     svs = "TCGA-2A-A8VL-01Z-00-DX1.2C2BD6EF-EC17-4117-AE89-A22B67AFB233.svs"
#     model = load_model("model_this")
#     score_dic = predict(model,get_limit())
#     print(score_dic)


def final_res():
    res = get_dict()
    data = pd.read_excel("tem_10.xlsx")
    dic = dict(zip(data['cases'].values, data['tag'].values))
    svs_list = np.unique([x for (x, y) in res.keys()])
    for svs in svs_list:
        label = dic[svs]
        predict_true = res.get((svs, label), 0)
        predict_false = res.get((svs, 1 ^ label), 0)
        print(svs, predict_true / (predict_false + predict_true))

def get_predict():
    limit = get_limit()
    model = load_model("model_best")
    score_dic = predict(model,limit)
    print(score_dic)


if __name__ == '__main__':
    final_res()

