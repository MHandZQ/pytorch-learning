# 迁移学习

在实际应用中，很少有网络会从头开始训练，大多采用迁移学习。

迁移学习的两种方法：

- 微调ConvNet：不是随机初始化，而是使用预训练网络的权重对网络进行初始化，然后再对整个网络进行训练
- 把ConvNet作为固定的特征提取器：就是，我直接使用预训练的网络，然后把前面的特征提取部分固定(freeze)，把分类器部分用一个新的随机初始化的部分替代，然后只训练随机初始化的部分。



# Finetuning ConvNet

```python
def main():
    '''
    迁移学习的第一种方法:修改网络最后分类器的结构,然后对整个网络进行训练
    '''
    #直接使用预训练的resnet50
    model_ft = models.resnet50(pretrained=True)
    #获得最后全连接层的输入通道数num_ftrs
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    #把全连接层的输出修改成类别数:本数据集中只有两类,所以是2
    model_ft.fc = nn.Linear(num_ftrs, 2)


    print('CUDA available: {}'.format(torch.cuda.is_available()))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #多GPU运算的实现,可以看到是先进行DataParallel然后to(device)
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

    #保存该模型的权重
    PATH = './model_ft.pth'
    torch.save(model_ft.state_dict(),PATH)

if __name__ == "__main__":
    main()
    visualize_model(num_images=6)
```

其他代码见：[transferlearning](E:\transferlearning)

多GPU实现如下：

![image-20200720200314379](C:\Users\77960\AppData\Roaming\Typora\typora-user-images\image-20200720200314379.png)



# ConvNet as fixed feature extractor

```python
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

    #保存该模型的权重
    PATH = './model_ft.pth'
    torch.save(model_ft.state_dict(),PATH)

if __name__ == "__main__":
    main()
    visualize_model(num_images=6)
```

其他代码见：[transferlearning1](E:\transferlearning1)