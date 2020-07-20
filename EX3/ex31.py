import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module): #Net类继承nn.Module类
    def __init__(self): #定义自己的构造函数，因为子类不会继承父类的构造方法，要么自己显式定义，要么编译器自动生成
        #super关键字跟java中的作用一样，super关键字表示父类（超类）。子类引用父类的字段时，可以用super.
        #此外，任何class的构造方法，第一行语句必须是调用父类的构造方法。
        super(Net,self).__init__()
        #输入图像是1个通道,输出为6通道，3x3的卷积核
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*6*6,120)#输入图像经过两层卷积后得到的图像尺寸是6*6
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.relu(self.conv1(x)) #输入图像先经过卷积，然后经过relu激活函数
        x = F.max_pool2d(x,(2,2)) #使用(2,2)窗口进行最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,(2,2))
        x = x.view(-1,self.num_flat_featires(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_featires(self,x):
        size = x.size()[1:] #因为切片从1开始,所有维度除了batch维度
        #假如x是四个维度[B,W,H,C],依次是batch size,图像宽度,高度,通道数,那么size为[W,H,C]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features #num_features = W*H*C

net = Net()#实例化Net类
#print(net)

params = list(net.parameters())#获取网络中的学习的参数,转换成列表
#print(len(params))
#print(params[0].size())# conv1中的权重

input = torch.randn(1,1,32,32)
out = net(input)
target = torch.randn(10) #随机产生十个数
target = target.view(1,-1) #使target和output有一样的形状
criterion = nn.MSELoss() #使用均方差损失函数

loss = criterion(out,target)
print(loss)
#print(loss.grad_fn) #MSELoss <MseLossBackward object at 0x000001EC323BA470>
#print(loss.grad_fn.next_functions[0][0]) #Linear <AddmmBackward object at 0x000001EC323BA390>
#print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLu <AccumulateGrad object at 0x000001EC323BA390>
#print(out)

#net.zero_grad() #将梯度缓冲区中的所有参数置0
#print(f"conv1.bias.grad before backward:{net.conv1.bias.grad}") #查看反向传播前conv1的偏差梯度

#loss.backward() #因为loss为标量,直接调用backward()即可
#print(f"conv1.bias.grad after backward:{net.conv1.bias.grad}")

optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad() #梯度缓冲区置0
output = net(input)
loss  = criterion(out,target)
loss.backward()
optimizer.step()