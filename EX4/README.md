# README

# 训练一个分类器

对于视觉领域，pytorch中已经创建了一个`torchvision`包，通过`torchvision.datasets`中的数据加载器(DataLoader)加载常用的数据集：ImageNet、CIFAR10、MNIST等。还可以使用`torch.utils.data.DataLoader`实现对图像的数据变换(DataTransformers)。



## Dataset，ImageFolder，DataLoader

数据集对象被抽象为`Dataset`类，实现自定义的数据集需要**继承Dataset**。且须实现__len__()和__getitem__()两个方法。

torchvision已经预先实现了常用的Dataset，包括前面使用过的CIFAR-10，以及ImageNet、COCO、MNIST、LSUN等数据集，可通过诸如`torchvision.datasets.CIFAR10`来调用。



`ImageFolder`，其也**继承自Dataset**。`ImageFolder`假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，**文件夹名为类名**，其构造函数如下：

```python
ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
```

它主要有四个参数：

- `root`：在root指定的路径下寻找图片
- `transform`：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
- `target_transform`：对label的转换
- `loader`：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象

label是按照文件夹名顺序排序后存成字典，即{类名:类序号(从0开始)}，一般来说最好直接将文件夹命名为从0开始的数字，这样会和ImageFolder实际的label一致。



如果只是每次读取一张图，那么上面的操作(ImageFolder)已经足够了，但是为了批量操作、打散数据、多进程处理、定制batch，那么我们还需要更高级的类：`DataLoader`定义如下

```python
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
```

参数依次为：

- `dataset`：加载的数据集(Dataset对象)
- `batch_size`：batch size
- `shuffle`：是否将数据打乱
- `sampler`： 样本抽样。定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
- `batch_sampler`（sampler，可选） - 和sampler一样，但一次返回一批索引。与batch_size，shuffle，sampler和drop_last相互排斥。
- `num_workers`：使用多进程加载的进程数，0代表不使用多进程
- `collate_fn`： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
- `pin_memory`：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些，默认为false
- `drop_last`：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃，默认为false



## 加载并归一化 CIFAR0

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(), #把数据变成tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #归一化
    ])
#使用torchvision.datasets中的CIFAR10数据集,把它下载下来,放到该文件夹下的data文件夹,并且进行transform
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#使用数据加载类DataLoader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```



## show images

```python
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2+0.5 #去归一化
    npimg = img.numpy() #转为numpy
    #把图像的后两个轴移到前面，第一个轴移到最后
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter = iter(trainloader) #使用.iter()把可迭代的数据变成迭代器
images, labels = dataiter.next() #对迭代器使用next可以不断获取batch
imshow(torchvision.utils.make_grid(images,nrow=2)) #把若干图像拼成一幅图像，nrow每一行展示的图片的数目(默认为8)。这里因为每一个batch中只有4幅图像
print(' '.join('%s' % classes[labels[j]] for j in range(4)))#.join用于将序列中的元素以指定的字符连接生成一个新的字符串。
```

`numpy.transpose()`用于反转或排列数组的轴；返回修改后的数组。对于具有两个轴的数组a，transpose（a）给出矩阵转置。

```python
x = np.ones((1, 2, 3))
>>> np.transpose(x, (1, 0, 2)).shape
(2, 1, 3)
```

`torchvision.utils.make_grid`可以将若干幅图像拼成一幅图像。



## 定义一个网络

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```



## 损失函数和最优化

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```



## 训练网络

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): #相当于把data=trainloader,i=0
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

'''
OUT
[1,  2000] loss: 2.255
[1,  4000] loss: 1.946
[1,  6000] loss: 1.726
[1,  8000] loss: 1.621
[1, 10000] loss: 1.530
[1, 12000] loss: 1.480
[2,  2000] loss: 1.420
[2,  4000] loss: 1.410
[2,  6000] loss: 1.366
[2,  8000] loss: 1.327
[2, 10000] loss: 1.337
[2, 12000] loss: 1.283
Finished Training
'''
```

`enumerate() `函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标。

```python
enumerate(sequence, start)
```

- sequence -- 一个序列、迭代器或其他支持迭代对象。
- start -- 下标起始位置。

```python
>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```



## 保存/加载网络

```python
PATH = 'E:/PytorchLearning/EX4/cifarnet.pth'
torch.save(net.state_dict(),PATH)
```

网络的保存有两种：

第一种：`torch.save(the_model.state_dict(), PATH)`

第二种：`torch.save(model,PATH)`

第一种只保存模型的参数(建议使用)，第二种保存整个模型



网络的加载：

对应第一种保存的加载方法是：

```python
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
```

对应第二种保存的加载方法是：

```python
the_model = torch.load(PATH)
```



## 网络的测试

测试一个Batch

```python
###测试网络,一个Batch
dataiter = iter(testloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images,nrow=2))
print('GroundTruth:',' '.join('%s'% classes[labels[j]] for j in range(4)))

##把之前保存的网络参数加载到新实例化的网络中
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)###outputs二维tensor，第一维是batchsize(数目为4)，第二维是类别数(数目为10)

_, predicted = torch.max(outputs, 1)### _是4幅图像中每一个类别的最大值,predicted是最大值所对应的索引,也就是预测的label值

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
'''
OUT:
GroundTruth:  cat   ship  ship plane
Predicted:    cat   car   car  ship
'''
```

测试所有Batch

```python
###测试所有Batch
correct = 0
total = 0
with torch.no_grad():  #这里的意思是不再更新网络中的参数
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0) #测试样本数, labels.size()的结果是torch.Size([4])
        correct += (predicted == labels).sum().item() #正确预测的样本数

print('Accuracy of the network on the 10000 test images: %d,%%' % (100*correct/total))
"""
OUT:
Accuracy of the network on the 10000 test images: 54 %
"""
```

还可以打印具体类别的准确率

```python
class_correct = list(0. for i in range(10))# 生成含有10个0.的列表
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data #比如labels=tensor([3, 8, 8, 0])
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze() #把判断predicted == labels的值(True,False)变成序列,c=tensor([ True, False, False, False])
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item() #True为1,False为0,从而实现统计每一类中预测正确的数目
            class_total[label] += 1 #统计每一类的数目


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    
"""
OUT:
Accuracy of plane : 55 %
Accuracy of   car : 81 %
Accuracy of  bird : 44 %
Accuracy of   cat : 45 %
Accuracy of  deer : 48 %
Accuracy of   dog : 19 %
Accuracy of  frog : 58 %
Accuracy of horse : 60 %
Accuracy of  ship : 65 %
Accuracy of truck : 61 %
"""
```



## 在GPU上进行训练

```python
###其他的可以不用修改,把模型以及输入数据加载到gpu上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
images, labels = data[0].to(device), data[1].to(device)
```



## 在多GPU上训练

在多个GPU上执行向前，向后传播是很自然的。但是，Pytorch默认仅使用一个GPU。您可以使用`DataParallel`以下方法使模型并行运行，从而轻松地在多个GPU上运行操作 ：

```
model = nn.DataParallel(model)
```

具体细节：

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#自己制作一个虚拟数据集
class RandomDataset(Dataset):  #继承Dataset类
    
    def __init__(self, size, length):  #自己定义的构造函数
        self.len = length
        self.data = torch.randn(length,size)
        
    def __getitem__(self, index):   #继承了Dataset类就一定要实现getitem函数和len函数
        return self.data[index]
    
    def __len__(self):
        return self.len
    
rand_loader = DataLoader(dataset = RandomDataset(input_size,output_size),batch_size = batch_size, shuffle=True)

#再定义一个简单模型
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
    
#创建模型以及并行计算
net = Net(input_size, output_size)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(model)

model.to(device)

#训练
for data in rand_loader:
    input = data.to(device)
    output = net(input)     
```

![image-20200720200314379](C:\Users\77960\AppData\Roaming\Typora\typora-user-images\image-20200720200314379.png)