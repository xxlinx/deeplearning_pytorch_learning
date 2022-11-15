import torch

import torchvision
import os

from torchvision.models import VGG16_Weights

os.environ['TORCH_HOME']='E:/torch-model'

vgg16 = torchvision.models.vgg16(VGG16_Weights.DEFAULT)
# 方式1-》模型结构和模型的参数
# torch.save(vgg16,"vgg16_method1.pth")

# 方式2， 模型的参数  官方推荐
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

