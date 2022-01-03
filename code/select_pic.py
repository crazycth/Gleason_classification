import numpy as np
import matplotlib.pyplot as plt
import openslide
import random
import os

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

def Select(img,num=10,save_root="/Users/richard/PycharmProjects/FUN/pic_save"):
    from queue import PriorityQueue
    que = PriorityQueue()
    for x in range(0, img.level_dimensions[1][0] - 224, 224):
        for y in range(0, img.level_dimensions[1][1] - 224, 224):
            im = img.read_region((x, y), 1, (224, 224))
            valid = check_valid(im)
            if valid <= 0.6:
                continue
            que.put((blue(im),random.random(),im))
            while que.qsize() > num:
                que.get()
    while not que.empty():
        valid,rd,im = que.get()
        im.save("/Users/richard/PycharmProjects/FUN/pic_save/" + str(valid) + ".png")



if __name__ == '__main__':
    root = "/Users/richard/PycharmProjects/FUN/data/0fad68ea-4c83-4c3e-918a-c8f63bb3adef/TCGA-EJ-5542-01Z-00-DX1.84ef2f6d-29d7-4d36-96b4-14bd30742016.svs"
    pic = openslide.OpenSlide(root)
    Select(pic,num=20)

