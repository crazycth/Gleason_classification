import numpy as np
import matplotlib.pyplot as plt
import openslide
import random
from queue import PriorityQueue
import os
from tqdm import tqdm

def check_valid(img):
    m = np.array(img)
    valid = np.sum(m<240)/np.prod(m.shape)
    return valid

def R(img):
    return img[:,:,0]

def G(img):
    return img[:,:,1]

def B(img):
    return img[:,:,2]

def blue(img):
    """
    :param img: 4-channel pic (R-0 G-1 B-2)
    :return: how blue it is
    """
    img = np.array(img).astype('int')
    matrix1 = (B(img)*100)/(1+R(img)+G(img))
    matrix2 = 256/(1+R(img)+G(img)+B(img))
    matrix1 = matrix1 * matrix2
    return np.average(matrix1)

def red(img):
    """
    :param img: 4-channel pic(R-0 G-1 B-2)
    :return: how red it is
    """
    img = np.array(img).astype('int')
    matrix1 = (R(img)*100)/(1+B(img)+G(img))
    matrix2 = 256.0/(1+R(img)+G(img)+B(img))
    matrix1 = matrix1 * matrix2
    return np.average(matrix1)

def Select_Blue(img,num=40):
    """
    :param img: Openslide_img
    :param num: the num of
    :param name: the name of file
    :return: the Sequential of img
    """
    que = PriorityQueue()
    for x in range(0, img.level_dimensions[0][0] - 224 * 4, 224 * 4):
        for y in range(0, img.level_dimensions[0][1] - 224 * 4, 224 * 4):
            im = img.read_region((x, y), 1, (224, 224))
            valid = check_valid(im)
            valid_blue = blue(im)
            if valid <= 0.7 or valid_blue >= 100:
                continue
            que.put((valid_blue, random.random(), im, x, y))
            while que.qsize() > num:
                que.get()
    pic_save = []
    while not que.empty():
        valid_blue , rd , im , x ,y = que.get()
        im = im.convert('RGB')
        pic_save.append(im)
    return pic_save



def Select_Red(img_list,num=10,name="init",save_root=",/pic_save"):
    que = PriorityQueue()
    for img in img_list:
        #Change!
        valid_red = red(img)
        valid_red = blue(img)
        que.put((valid_red,random.random(),img))
        while que.qsize()>num:
            que.get()
    count = 0
    while not que.empty():
        count = count + 1
        valid_red , rd , im = que.get()
        im.save(save_root + "/" + str(name) + "_" + str(valid_red)[:4] + "_" + str(count) + ".jpg")


def Select(img,num=10,name="init",save_root="./pic_save"):
    """
    :param img: sys picture
    :param num: the num of part u want to save
    :param save_root: save root
    :return: nothing
    """
    img_list = Select_Blue(img,num*2)
    Select_Red(img_list,num,name,save_root)

    # from queue import PriorityQueue
    # que = PriorityQueue()
    # for x in range(0, img.level_dimensions[0][0] - 224*4, 224*4):
    #     for y in range(0, img.level_dimensions[0][1] - 224*4, 224*4):
    #         im = img.read_region((x, y), 1, (224, 224))
    #         valid = check_valid(im)
    #         valid_blue = blue(im)
    #         valid_red = red(im)
    #         if valid <= 0.6 or valid_blue >= 100:
    #             continue
    #         que.put((valid_blue,random.random(),im,x,y))
    #         while que.qsize() > num:
    #             que.get()
    # count = 0
    # while not que.empty():
    #     valid,rd,im,x,y = que.get()
    #     count = count + 1
    #     im = im.convert('RGB')
    #     im.save(save_root+"/"+str(name)+"_"+str(valid)[:4]+"_"+str(count)+"_"+str(x)+"_"+str(y)+".jpg")



if __name__ == '__main__':
    root = "/Users/richard/GDCdata/TCGA-PRAD/harmonized/Biospecimen/Slide_Image/d20227e2-228f-40d9-97da-1f7bef83db44/TCGA-FC-A8O0-01A-04-TSD.88941EBE-4275-4123-8696-D579D916F810.svs"
    pic = openslide.OpenSlide(root)
    Select(pic,20*5)