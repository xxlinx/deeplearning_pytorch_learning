from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
#tensor数据类型
#通过transforms.ToTensor去看两个问题

#1.如何使用
#2.为什么需要tensor数据类型
img_path = "../dataset/train/ants_image/7759525_1363d24e88.jpg"
img_path_abs = 'E:\Desktop\learn_pytorch\dataset\train\ants_image\7759525_1363d24e88.jpg'
img = Image.open(img_path)
# print(img)
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x333 at 0x16AEF223880>
writer = SummaryWriter('logs')

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)
#tensor([[[0.8863, 0.8824, 0.874 ....... 0.4235, 0.4235]]])

#需要tensor的原因
#backward_hooks  grad device requires_grad
#里面包装了神经网络需要的数据

writer.add_image('Tensor_img',tensor_img)

writer.close()