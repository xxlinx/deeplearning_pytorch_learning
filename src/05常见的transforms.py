from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
img = Image.open("../imgs/pytorch.png")

# print(img)
#<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1096x610 at 0x2AC57BCE700>

writer = SummaryWriter('logs')

#ToTensor


trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTensor', img_tensor)


#Normalize归一化
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
writer.add_image('Normalize', img_norm)

#Resize
# print(img.size)
#(1096, 610)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
# print(img_resize)
#<PIL.Image.Image image mode=RGB size=512x512 at 0x27ECE26F100>

img_resize = trans_totensor(img_resize)
writer.add_image('Resize', img_resize,0)

#compose - resize - 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('Resize',img_resize_2,1)

#randomcrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('randomcrop',img_crop,i)




writer.close()