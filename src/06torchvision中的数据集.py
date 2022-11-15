from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
trans_set = torchvision.datasets.CIFAR10(root='../cifar10', train=True,transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root='../cifar10', train=False,transform=dataset_transform, download=False)
# print(test_set[0])
# (<PIL.Image.Image image mode=RGB size=32x32 at 0x1448D97A4C0>, 3)
# print(test_set.classes)
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# img, target = test_set[0]
# print(img)
# <PIL.Image.Image image mode=RGB size=32x32 at 0x1448D97A520>
# print(target)
# 3
# print(test_set.classes[target])
# cat

# transform=dataset_transform 加上之后
# print(test_set[0])
# 输出的是tensor

writer = SummaryWriter('p10')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()