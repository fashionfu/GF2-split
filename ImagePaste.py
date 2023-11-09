# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：ImagePaste.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/5/12 14:53 
'''
from PIL import Image

# 打开图片并调整大小
image1 = Image.open('image1.png')
image1 = image1.resize((1000, 1000))
image2 = Image.open('image2.png')
image2 = image2.resize((1000, 1000))
image3 = Image.open('image3.png')
image3 = image3.resize((1000, 1000))
image4 = Image.open('image4.png')
image4 = image4.resize((1000, 1000))

image5 = Image.open('image5.png')
image5 = image5.resize((1000, 1000))
image6 = Image.open('image6.png')
image6 = image6.resize((1000, 1000))
image7 = Image.open('image7.png')
image7 = image7.resize((1000, 1000))
image8 = Image.open('image8.png')
image8 = image8.resize((1000, 1000))

# 创建一个新的空白图片，大小为2x2的400x400像素图片
new_image = Image.new('RGB', (3000, 3000))

# 将四张图片按照2x2的方式拼接在一起
new_image.paste(image1, (0, 0))
new_image.paste(image2, (1000, 0))
new_image.paste(image3, (2000, 0))
new_image.paste(image4, (0, 1000))
new_image.paste(image5, (1000, 1000))
new_image.paste(image6, (2000, 1000))
new_image.paste(image7, (0, 2000))
new_image.paste(image8, (1000, 2000))
new_image.paste(image8, (2000, 2000))

# 保存拼接后的图片
new_image.save('merged_image.jpg')