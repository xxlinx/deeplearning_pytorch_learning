#gpu训练起作用的部分为 网络模型  数据（输入输出）  损失函数
import time

import torch.cuda
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="../cifar10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
lx = Lx()
#cuda1
if torch.cuda.is_available():
    lx = lx.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
#cuda2
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
# learning_rate = 0.01  拿出来的话阚看着方便，修改也方便
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

#记录时间
start_time = time.time()

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始

    #加入这个train() 只对部分网络层起作用
    lx.train()
    for data in train_dataloader:
        imgs, targets = data
        #cuda 3
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = lx(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            #训练100次的时间
            print(end_time - start_time)
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始

    #加入eval() 只对某些层起作用
    tudui.eval()
    total_test_loss = 0
    #整体正确的个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            #cuda 3
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = lx(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()

            #分类问题的正确率 argmax（1）表示方向是横向
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(lx, "lx_{}.pth".format(i))
    print("模型已保存")

writer.close()