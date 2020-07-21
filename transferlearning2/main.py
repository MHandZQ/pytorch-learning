from __future__ import print_function, division

import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler
from torchvision import models


from train import train_model
from visualize import visualize_model


def main():
    '''
    迁移学习的第二种方法:除了最后一层，固定整个网络参数。训练也就只修改最后一层的参数
    实现方法是:requires_grad == False
    '''
    #直接使用预训练的resnet50
    model_ft = models.resnet50(pretrained=True)

    '''
    与第一种方法相比就只修改这里就可以了
    '''
    for param in model_conv.parameters():
        param.requires_grad = False

    #新构建的网络结构的requires_grad是默认为 True 的，所以参数是会修改的
    #获得最后全连接层的输入通道数num_ftrs
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    #把全连接层的输出修改成类别数:本数据集中只有两类,所以是2
    model_ft.fc = nn.Linear(num_ftrs, 2)


    print('CUDA available: {}'.format(torch.cuda.is_available()))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #多GPU运算的实现,可以看到是先进行DataParallel然后to(device)
    '''
    这里有一个坑:多GPU训练在原始网络结构中添加了一层module,在测试模型时不需要用到多GPU测试，因此在保存模型时应该把module层去掉
    '''
    if torch.cuda.device_count() > 1:
        model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #训练模型然后返回结果最好的模型
    model_ft = train_model(model_ft,criterion,optimizer_ft,exp_lr_scheduler,num_epochs=25)

    #保存该模型的权重,如果是d多GPU训练还需要加 .module 把module层去掉。这样在加载模型时不会有限制
    PATH = './model_ft.pth'
    if torch.cuda.device_count() > 1:
        torch.save(model_ft.module.state_dict(),PATH)
    else:
        torch.save(model_ft.state_dict(),PATH)

if __name__ == "__main__":
    main()
    visualize_model(num_images=6)
