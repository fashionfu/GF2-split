# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：DataValue.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/5/16 11:15 
'''
import osgeo.gdal as gdal

# 打开遥感影像文件
dataset = gdal.Open(r"D:\python\PycharmDemo\demo\train\GF2\GF2_PMS1_E108.7_N32.4_20160402_L1A0001501540-NND1_49N_wurenji_xiaomixi_roi.tif", gdal.GA_Update)
if dataset is None:
    raise Exception("Could not open file: " + "filename.tif")

# 获取影像的宽度和高度
width = dataset.RasterXSize
height = dataset.RasterYSize

# 将data ignore value设置为0
for i in range(1, dataset.RasterCount + 1):
    band = dataset.GetRasterBand(i)
    band.SetNoDataValue(0)

# 关闭文件
dataset = None
