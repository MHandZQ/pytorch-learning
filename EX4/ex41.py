import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


###数据加载
transform = transforms.Compose([
    transforms.ToTensor(), #把数据变成tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #归一化
    ])
#使用torchvision.datasets中的CIFAR10数据集,把它下载下来,放到该文件夹下的data文件夹,并且进行transform
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#使用数据加载类DataLoader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


###显示图片
def imshow(img):
    img = img/2+0.5 #去归一化
    npimg = img.numpy() #转为numpy
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#imshow(torchvision.utils.make_grid(images,nrow=2))
#print(' '.join('%s' % classes[labels[j]] for j in range(4)))

###定义网络
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

'''
net = Net()

###定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

###训练网络
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i,data in enumerate(trainloader,start=0):
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

###保存网络
PATH = 'E:/PytorchLearning/EX4/cifarnet.pth'
torch.save(net.state_dict(),PATH)
'''
PATH = 'E:/PytorchLearning/EX4/cifarnet.pth'

###测试网络,一个Batch
#dataiter = iter(testloader)
#images, labels = dataiter.next()

#imshow(torchvision.utils.make_grid(images,nrow=2))
#print('GroundTruth:',' '.join('%s'% classes[labels[j]] for j in range(4)))

##把之前保存的网络参数加载到新实例化的网络中
net = Net()
net.load_state_dict(torch.load(PATH))

#outputs = net(images)###outputs二维tensor，第一维是batchsize(数目为4)，第二维是类别数(数目为10)

#_, predicted = torch.max(outputs, 1)### _是4幅图像中每一个类别的最大值,predicted是最大值所对应的索引,也就是预测的label值

#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

'''
###测试整个网络
correct = 0
total = 0
with torch.no_grad():  #这里的意思是不再更新网络中的参数
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0) #测试样本数, labels.size()的结果是torch.Size([4])
        correct += (predicted == labels).sum().item() #正确预测的样本数

print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))
'''

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