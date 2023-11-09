# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/5/8 20:45 
'''
from Split import gdal_image_split
from Split import crop_image
from TiffToJPG import batch_tiff_to_jpg
if __name__ == '__main__':
    # 先将整幅高分二号影像裁剪到gdal可以读取的大小，再调用tiff文件裁剪成256*256大小的小tiff文件
    # gdal_image_split(r"G:\傅炜舜-数据处理\water\GF2_32.6\GF2_PMS1_E108.7_N32.6_20160402_L1A0001501541-GS1_2.tif", "test")
    # 调用批量转换函数
    batch_tiff_to_jpg("test_bands=3", "jpgImages")
