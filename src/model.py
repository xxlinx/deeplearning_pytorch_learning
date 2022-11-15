from torch import nn


class Lx(nn.Module):
    def __init__(self):
        super(Lx, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    lx = Lx()
    input = torch.ones((64,3,32,32))
    output=lx(input)
    print(output.shape)