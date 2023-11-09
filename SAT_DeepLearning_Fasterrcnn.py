# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：SAT_DeepLearning_Fasterrcnn.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/5/12 23:26 
'''
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from osgeo import gdal
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F_t
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 需要定义一个用于加载遥感图像数据的函数，该函数使用GDAL库读取图像数据，并将其转换为PyTorch张量：
# 首先是load_image函数，该函数用于加载遥感图像数据。
# 在这个函数中，我们使用GDAL库读取图像数据，并将其转换为PyTorch张量。
# 具体来说，我们首先打开图像文件，并获取其波段数量。然后，我们依次读取每个波段的数据，并将其转换为numpy数组。
# 接下来，我们将所有波段的数据拼接在一起，形成一个三维数组，表示整个图像。
# 最后，我们将numpy数组转换为PyTorch张量，并将其返回。

# 定义加载遥感图像数据的函数
def load_image(filename):
    # 打开图像文件
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    if dataset is None:
        raise Exception("Could not open file: " + filename)
    # 获取波段数量
    band_count = dataset.RasterCount
    bands = []
    # 依次读取每个波段的数据
    for i in range(1, band_count + 1):
        band = dataset.GetRasterBand(i)
        data = band.ReadAsArray()
        data = np.expand_dims(data, axis=0)
        bands.append(data)
    # 将所有波段的数据拼接在一起，形成一个三维数组
    image = np.concatenate(bands, axis=0)
    image = image.astype(np.float32)
    # 将numpy数组转换为PyTorch张量
    image = torch.from_numpy(image)
    return image

# 需要定义一个Faster R-CNN模型，用于对遥感图像进行目标检测。这里我们使用预训练的ResNet-50作为特征提取器
# 接下来是Model类，该类用于定义Faster R-CNN模型。
# 在这个类中，我们首先使用fasterrcnn_resnet50_fpn函数加载预训练的ResNet-50模型，并将其作为特征提取器。
# 然后，我们根据任务需要修改模型的最后一层，用于预测目标类别和边界框。
# 具体来说，我们使用FastRCNNPredictor类定义一个新的预测器，其输入特征数量为ResNet-50的输出特征数量，输出类别数量为2（背景和变化）。

# 定义Faster R-CNN模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 加载预训练的ResNet-50模型
        self.base_model = fasterrcnn_resnet50_fpn(pretrained=True)
        # 修改模型的最后一层
        num_classes = 2  # 0: background, 1: changed
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        # 对输入图像进行归一化
        x = F_t.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 将图像传递给模型，并获取模型的输出
        outputs = self.base_model([x])
        return outputs

# 需要定义训练函数和测试函数，用于训练Faster R-CNN模型并对测试数据进行目标检测：
# 接下来是train函数，该函数用于训练Faster R-CNN模型。在这个函数中，我们首先将模型设置为训练模式，然后将优化器的梯度清零。
# 接着，我们将输入数据传递给模型，得到模型的输出。由于Faster R-CNN模型的输出是一个字典，包含多个关键字，
# 因此我们需要使用outputs[0]获取字典的第一个元素。然后，我们计算损失函数，并调用backward方法计算梯度。
# 最后，我们使用优化器更新模型参数。
# 需要注意的是，在这个示例代码中，我们并没有定义具体的损失函数，而是将其设置为None。在实际应用中，您需要根据任务需要定义合适的损失函数。

# 定义训练函数
def train(model, optimizer, criterion, train_data, train_labels):
    # 将模型设置为训练模式
    model.train()
    # 将优化器的梯度清零
    optimizer.zero_grad()
    # 将输入数据传递给模型，得到模型的输出
    outputs = model(train_data)
    # 计算损失函数
    loss_dict = outputs[0]
    losses = sum(loss for loss in loss_dict.values())
    loss = losses.item()
    # 计算梯度
    losses.backward()
    # 使用优化器更新模型参数
    optimizer.step()

# 需要定义一个函数，用于将检测结果可视化并保存为图像文件：
# 接下来是save_detection_result函数，该函数用于将检测结果可视化并保存为图像文件。
# 在这个函数中，我们首先使用F_t.to_pil_image函数将PyTorch张量转换为PIL图像。
# 然后，我们使用ImageDraw库在图像上绘制检测结果，包括边界框和置信度分数。最后，我们将结果保存为图像文件。

# 定义将检测结果可视化并保存为图像文件的函数
def save_detection_result(image, detections, filename):
    colors = [[255, 0, 0], [0, 255, 0]]
    # 将PyTorch张量转换为PIL图像
    image = F_t.to_pil_image(image)
    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = detection['boxes'][0]
        score = detection['scores'][0]
        label = detection['labels'][0]
        color = colors[label]
        # 在图像上绘制检测结果
        draw.rectangle(box.tolist(), outline=tuple(color), width=2)
        draw.text((box[0], box[1]), f"{score:.2f}", fill=tuple(color))
    # 将结果保存为图像文件
    image.save(filename)

# 需要定义一个主函数，用于读取遥感图像数据、训练模型、对测试数据进行目标检测并保存结果：
# 最后是main函数，该函数是程序的主入口。在这个函数中，我们首先加载遥感图像数据，并创建一个Faster R-CNN模型。
# 然后，我们使用定义好的训练函数对模型进行训练。接着，我们对测试数据进行目标检测，并将检测结果保存为图像文件。

# 定义主函数
def main():
    # Load image data
    # 加载遥感图像数据
    image1 = load_image(r"C:\Users\10208\Desktop\wureji_roi3.dat")
    image2 = load_image(r"C:\Users\10208\Desktop\gf7_roi3.dat")
    images = [image1, image2]

    # Load model
    # 创建Faster R-CNN模型
    model = Model()
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = None  # TODO: Define criterion for training

    # Define targets for training
    targets1 = [...]  # define targets for image1
    targets2 = [...]  # define targets for image2
    targets = [targets1, targets2]

    # # Train model
    # # 训练模型
    # num_epochs = 1
    # for epoch in range(num_epochs):
    #     for i, image in enumerate(images):
    #         labels = targets[i]
    #         train(model, optimizer, criterion, image, labels)

    # Train model
    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, image in enumerate(images):
            labels = None  # TODO: Define labels for training
            train(model, optimizer, criterion, image, labels)

    # Test model
    # 测试模型
    detections = []
    for image in images:
        outputs = model(image)
        detections.append(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels'])
    # 将检测结果保存为图像文件
    save_detection_result(image1, detections[0], "result1.png")
    save_detection_result(image2, detections[1], "result2.png")

main()