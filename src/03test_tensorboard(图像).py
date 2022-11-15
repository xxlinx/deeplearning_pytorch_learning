from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
# image_path = "../dataset/train/ants_image/0013035.jpg"
image_path = "../dataset/train/ants_image/5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

# writer.add_image('test', img_array, 1, dataformats='HWC')
writer.add_image('test', img_array, 2, dataformats='HWC')

writer.close()
