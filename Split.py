# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：Split.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/5/8 20:44 
'''
# 定义分割函数
from osgeo import gdal, osr
import os
import numpy as np
import cv2
from numba import jit
import math

# 以下是经验证过的可裁剪高分二号影像的函数
def gdal_image_split(filename, output_dir, size=256,im_bands=3):
    in_ds = gdal.Open(filename)
    #get tiff 信息
    xsize = in_ds.RasterXSize
    ysize = in_ds.RasterYSize
    bands = in_ds.RasterCount
    geotransform = in_ds.GetGeoTransform() # 获取仿射变换信息
    projection = in_ds.GetProjectionRef() # 获取投影信息
    block_data = in_ds.ReadAsArray(0,0,xsize,ysize).astype(np.float32)

    print("波段数为：",bands)
    print("行数为：",xsize)
    print("列数为：",ysize)

    # 逐行逐列进行裁剪
    for i in range(0, xsize, size):
        for j in range(0, ysize, size):
            # 计算裁剪窗口的范围
            x_min = i
            x_max = min(i + size, xsize)
            y_min = j
            y_max = min(j + size, ysize)
            # 注意，如果要保存4波段的tiff文件，记得把range(im_bands)改成range(bands)即可
            # 读取数据
            data = []
            for b in range(bands):
                band = in_ds.GetRasterBand(b + 1)
                data.append(band.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min))

            # band_1 = in_ds.GetRasterBand(1)
            # data_1 = band_1.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
            # band_2 = in_ds.GetRasterBand(2)
            # data_2 = band_2.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
            # band_3 = in_ds.GetRasterBand(3)
            # data_3 = band_3.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)
            # band_4 = in_ds.GetRasterBand(4)
            # data_4 = band_4.ReadAsArray(x_min, y_min, x_max - x_min, y_max - y_min)

            # 生成输出文件名
            output_file = os.path.join(output_dir, f"{i}_{j}.tif")

            # 保存裁剪后的子图片
            driver = gdal.GetDriverByName('GTiff')
            # 注意，如果要保存4波段的tiff文件，记得把create中im_bands改成bands即可
            dataset = driver.Create(output_file, x_max-x_min, y_max-y_min, bands, gdal.GDT_Float32)  # 1为波段数量
            dataset.SetProjection(projection)  # 写入投影
            dataset.SetGeoTransform((geotransform[0]+x_min*geotransform[1], geotransform[1], 0, geotransform[3]+y_min*geotransform[5], 0, geotransform[5]))  # 写入仿射变换参数
            # 注意，如果要保存4波段的tiff文件，记得把range(im_bands)改成range(bands)即可
            for b in range(bands):
                dataset.GetRasterBand(b + 1).WriteArray(data[b])
            # dataset.GetRasterBand(1).WriteArray(data_1)
            # dataset.GetRasterBand(2).WriteArray(data_2)
            # dataset.GetRasterBand(3).WriteArray(data_3)
            '''
            dataset.GetRasterBand(4).WriteArray(data_4)#这个地方出错了，可能是写数组的时候，应该要从0开始
            '''
            # for i in range(im_bands):
            #     dataset.GetRasterBand(i+1).WriteArray(data)
            # dataset.GetRasterBand(1).WriteArray(data)
            dataset.FlushCache()
            dataset = None

            # 关闭数据集
        dataset = None


# 定义裁剪函数
def crop_image(input_file, output_dir, size=256):
    """
    将输入的高分二号tif影像裁剪成256*256大小的子图片，并保存到指定的目录下
    :param input_file: 输入的高分二号tif影像路径
    :param output_dir: 裁剪后子图片保存目录
    :param size: 子图片大小，默认为256
    """
    # 打开tif影像文件
    dataset = gdal.Open(input_file)
    if dataset is None:
        print("Could not open input file")
        return

    # 获取tif影像的地理参考信息
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    # print(projection,geotransform)

    # 获取tif影像的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 构建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 逐行逐列进行裁剪
    for i in range(0, width, size):
        for j in range(0, height, size):
            # 计算裁剪窗口的范围
            x_min = i
            x_max = min(i+size, width)
            y_min = j
            y_max = min(j+size, height)

            # 读取数据
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray(x_min, y_min, x_max-x_min, y_max-y_min)

            # 生成输出文件名
            output_file = os.path.join(output_dir, f"{i}_{j}.tif")

            # 保存裁剪后的子图片
            driver = gdal.GetDriverByName("GTiff")
            out_dataset = driver.Create(output_file, x_max-x_min, y_max-y_min, 1, gdal.GDT_Float32)
            out_dataset.SetGeoTransform((geotransform[0]+x_min*geotransform[1], geotransform[1], 0, geotransform[3]+y_min*geotransform[5], 0, geotransform[5]))
            out_dataset.SetProjection(projection)
            out_dataset.GetRasterBand(1).WriteArray(data)
            out_dataset.FlushCache()
            out_dataset = None

    # 关闭数据集
    dataset = None

# 调用裁剪函数
# crop_image(r"G:\傅炜舜-数据处理\water\GF2_32.6\GF2_PMS1_E108.7_N32.6_20160402_L1A0001501541-GS1_2.tif", "test")












