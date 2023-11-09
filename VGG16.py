# -*- coding: UTF-8 -*-
'''
@Project ：PycharmDemo 
@File    ：VGG16.py
@IDE     ：PyCharm 
@Author  ：10208 
@Date    ：2023/5/8 20:46 
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 设置超参数
batch_size = 32
epochs = 10
learning_rate = 0.01

# 定义VGG16网络模型
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.data = []
        for file_name in os.listdir(data_dir):
            self.data.append(os.path.join(data_dir, file_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transform(img)
        return img

testdataset = MyDataset('test/')
testdataloader = DataLoader(testdataset, batchsize=batch_size, shuffle=False)
net = VGG16()
net.loadstatedict(torch.load('model.pth'))
result = []
with torch.nograd():
    for data in testdataloader:
        outputs = net(data)
        predicted = torch.round(outputs)
        result.extend(predicted.tolist())
print(result)


# 上述代码中，使用PyTorch框架搭建了一个VGG16网络模型，并基于MyDataset类实现了对数据集的预处理。
# 使用load_state_dict函数加载之前训练好的模型，使用torch.no_grad函数关闭了梯度的计算，以提高运行效率。
# 在处理过程中，使用extend函数将处理结果加入到result列表中，并最终打印出结果。
# 需要注意的是，由于没有提供具体的高分二号图像数据，因此代码中的处理结果仅供参考。