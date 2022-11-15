import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Lx(nn.Module):
    def __init__(self):
        super(Lx, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

lx = Lx()



for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    #torch.size([64,3,32,32])

    #output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    #flatten 可以把这个torch直接展平（一行）
    print(output.shape)
    #torch.size([196608])
    output = lx(output)
    print(output.shape)
    #torch.Size([10])

