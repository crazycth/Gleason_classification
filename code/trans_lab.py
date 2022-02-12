import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def BGR2RGB(img):
    B,G,R = cv2.split(img)
    img = cv2.merge([R,G,B])
    return img


def get_target(img):
    img = img.astype('float32')
    img *= 1./255
    img_lab = cv2.cvtColor(img,cv2.COLOR_RGB2Lab)

    l,a,b = cv2.split(img_lab)
    ave_l = np.average(l)
    ave_a = np.average(a)
    ave_b = np.average(b)

    std_l = np.std(l)
    std_a = np.std(a)
    std_b = np.std(b)

    return ( ave_l , ave_a , ave_b , std_l , std_a , std_b )


def trans_lab(img,target):
    t_l , t_a , t_b , t_std_l , t_std_a , t_std_b = target

    img = np.array(img).astype('float32')
    img *= 1./255
    img_lab = cv2.cvtColor(img,cv2.COLOR_RGB2Lab)

    l,a,b = cv2.split(img_lab)
    ave_l = np.average(l)
    ave_a = np.average(a)
    ave_b = np.average(b)

    std_l = np.std(l)
    std_a = np.std(a)
    std_b = np.std(b)

    l,a,b = l-ave_l,a-ave_a,b-ave_b
    l,a,b = (t_std_l/std_l)*l + t_l , (t_std_a/std_a)*a + t_a , (t_std_b/std_b)*b + t_b

    img = cv2.merge([l,a,b]).astype('float32')
    img = cv2.cvtColor(img,cv2.COLOR_Lab2RGB)
    return img


def Color_Normalization(target_img=None):
    target = BGR2RGB(target_img)
    target = get_target(target)
    for pic_path in tqdm(os.listdir("pic_save_1")):
        if pic_path.startswith("."):continue
        path = "./pic_save_1/"+pic_path
        pic = cv2.imread(path)
        pic = BGR2RGB(pic)
        pic = trans_lab(pic,target)
        img = Image.fromarray(np.uint8(pic * 255))
        img.save("./pic_trans_1/"+pic_path)


if __name__ == '__main__':
    target_pic = cv2.imread("/Users/richard/PycharmProjects/FUN/pic_save_1/TCGA-2A-A8VL_53.7_2.jpg")
    Color_Normalization(target_pic)