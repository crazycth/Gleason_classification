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

😚
