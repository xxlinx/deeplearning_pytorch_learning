import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

#1000分类
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('../cifar10', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
#在这个模型后面再加一个线性的全连接 转换为10分类
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
#把原本的模型进行修改
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)