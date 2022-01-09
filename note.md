# Weaky-supervised-Gleason-Grading

基于深度学习的弱监督格里森分级方法

本质是运用深度学习方法对病理图像进行格里森分级

最大类间法Otsu

得到的图像是RGB图像



文章中提及基于OTSU的图像分割

四channel : RGBA
$$
B R_{i}=\frac{100 \times B_{i}}{1+R_{i}+G_{i}} \times \frac{256}{1+R_{i}+G_{i}+B_{i}}
$$


初步确定在第一层使用224*224的窗口大小进行切片



如果acc>=0.99 则舍弃

需要改进的地方：







1. imread读进去是BGR，需要转化为RGB
2. cv2.cvtCOlor前需要*1./255



```python
root = "svs_pic/TCGA-2A-A8VX-01A-01-TSA.E697DF9D-58CF-468C-AF29-A26F44757653.svs"
pic = openslide.OpenSlide(root)
from PIL import Image

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


def get_target(img):
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



img = pic.read_region((8000,4000),1,(224,224))
img = img.convert('RGB')
# img
standard_pic = cv2.imread("/Users/richard/PycharmProjects/FUN/pic_save/TCGA-EJ-A65M_66.0_129.jpg")
standard_pic = cv2.cvtColor(standard_pic,cv2.COLOR_BGR2RGB).astype('float32')
standard = get_target(standard_pic)
standard
img = trans_lab(img,standard)
# #plt.imshow(standard_pic)
# # img = np.asarray(img.convert("RGB"))
# # #plt.imshow(img)
# # # #get_target(img)
img = Image.fromarray(img)
img

```





色彩归一化

```python
root = "svs_pic/TCGA-2A-A8VX-01A-01-TSA.E697DF9D-58CF-468C-AF29-A26F44757653.svs"
pic = openslide.OpenSlide(root)
from PIL import Image

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


#实际上不是float32
def BGR2RGBfloat32(img):
    B,G,R = cv2.split(img)
    img = cv2.merge([R,G,B])
    return img



img = cv2.imread("/Users/richard/PycharmProjects/FUN/pic_save/TCGA-EJ-5525_66.0_37.jpg")
B,G,R = cv2.split(img)
img = cv2.merge([R,G,B])
img_old = Image.fromarray(np.uint8(img))
img_old.show(title='old')
# img

standard_pic = cv2.imread("/Users/richard/PycharmProjects/FUN/pic_save/TCGA-EJ-A46B_61.0_50.jpg")
standard_pic = BGR2RGBfloat32(standard_pic)
#standard_pic = cv2.cvtColor(standard_pic,cv2.COLOR_BGR2RGB).astype('float32')
standard = get_target(standard_pic)


img = trans_lab(img,standard)
# #plt.imshow(standard_pic)
# # img = np.asarray(img.convert("RGB"))
# # #plt.imshow(img)
# # # #get_target(img)
img= Image.fromarray(np.uint8(img*255))
img.show(title='new')
```

