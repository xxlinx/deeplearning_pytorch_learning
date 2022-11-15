import torch
import torchvision
from torch import nn
import os

from torchvision.models import VGG16_Weights

os.environ['TORCH_HOME']='E:/torch-model'
# 方式1-》保存方式1，加载模型

model = torch.load('vgg16_method1.pth')
print(model)

# 方式2，加载模型 字典格式
vgg16 = torchvision.models.vgg16(pretrained= False)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
# model = torch.load('vgg16_method2.pth')
print(vgg16)

# 陷阱1
# class Lx(nn.Module):
#     def __init__(self):
#         super(Lx, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# lx = Lx()
#
# torch.save(lx, "lx_method.pth")
