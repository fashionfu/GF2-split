# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：TiffToJPG.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/5/9 15:57 
'''
from osgeo import gdal
from PIL import Image
import os

# 定义转换函数
def tiff_to_jpg(input_file, output_dir):
    """
    将输入的tiff文件转换为jpg文件，并保存到指定的目录下
    :param input_file: 输入的tiff文件路径
    :param output_dir: 转换后jpg文件保存目录
    """
    # 打开tiff文件
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)

    # 获取tif影像的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 获取波段数
    num_bands = dataset.RasterCount

    # 逐波段读取数据
    data = []
    for b in range(num_bands):
        band = dataset.GetRasterBand(b+1)
        data.append(band.ReadAsArray())

    # 关闭数据集
    dataset = None

    # 合并多波段数据
    scale = 255 / (data[0].max() - data[0].min())  # 计算比例因子
    # 当我们将每个像素的各个波段的值分别存储在RGB三元组的三个分量中后，我们需要将它们组合成一个RGB三元组，并将它添加到img_data列表中。
    # 这一步可以通过tuple函数来实现。在Python中，tuple函数可以将一个序列或者一个可迭代对象转换为一个元组。
    # 在这里，我们可以使用tuple函数将一个包含三个波段值的列表转换为一个RGB三元组。
    # 具体来说，我们首先创建一个空列表img_data，用于存储合并后的图像数据。然后，对于每个像素，我们创建一个空列表pixel，用于存储各个波段的值。
    # 接着，对于每个波段，我们将它的值进行线性变换，并将它添加到pixel列表中。最后，我们使用tuple函数将pixel列表转换为一个RGB三元组，并将它添加到imgdata列表中。
    img_data = []
    for i in range(height):
        row = []
        for j in range(width):
            pixel = []
            for b in range(num_bands):
                pixel.append(int(round((data[b][i][j] - data[b].min()) * scale)))
            row.append(tuple(pixel))
        img_data.append(row)
    # 在上面的代码中，我们对于每个像素，使用一个嵌套的循环遍历各个波段，将它们的值进行线性变换，并将它们添加到一个列表pixel中。
    # 然后，我们使用tuple函数将pixel列表转换为一个RGB三元组，并将它添加到一个列表row中。最后，我们将row列表添加到img_data列表中，完成一幅RGB图像的合并。
    # 需要注意的是，我们使用了int和round函数将像素值转换为整数，并使用tuple函数将它们转换为RGB三元组。

    # 生成输出文件名
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + ".jpg")

    # 将像素值转换为整数类型，并保存jpg文件
    img = Image.new("RGB", (width, height))
    img.putdata([tuple(pixel) for row in img_data for pixel in row])
    img.save(output_file, "JPEG")

# 遍历目录，批量转换
def batch_tiff_to_jpg(input_dir, output_dir):
    """
    批量将指定目录下的tiff文件转换为jpg文件
    :param input_dir: 输入目录
    :param output_dir: 转换后jpg文件保存目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 判断是否为tiff文件
            if file.endswith(".tif") or file.endswith(".tiff"):
                # 转换文件
                input_file = os.path.join(root, file)
                tiff_to_jpg(input_file, output_dir)

