from torch import nn
import torch

class Lx(nn.Module):
    def __init__(self) :
        super().__init__()
    def forward(self, input):
        output = input+1
        return output

lx= Lx()
x = torch.tensor(1.0)
output = lx(x)
print(output)