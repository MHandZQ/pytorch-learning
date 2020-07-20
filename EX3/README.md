# README

神经网络可以通过`torch.nn`包来构建。

`nn`依赖于`autograd`定义模型并对其进行求梯度。一个`nn.Module`包含`layers`和一种返回`output`的方法`forward(input)`。

神经网络的典型训练过程如下：

- 定义具有一些可学习参数（或权重）的神经网络
- 遍历输入数据集
- 通过网络处理输入
- 计算损失（距离正确输出有多远）
- 将梯度传播回网络参数
- 通常使用简单的更新规则来更新网络的权重： `weight = weight - learning_rate * gradient`



# 定义一个网络

![AlexNet](E:\PytorchLearning\EX3\AlexNet.png)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = F.relu(conv2(x))
        x = F.max_pool2d(x)
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
print(net)

"""
OUT
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=576, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
"""
```

只需要定义`forward`函数，当你使用`autograd`时`backward` 函数（计算梯度）就会被自动定义。可以在`forward`函数中使用任何Tensor操作。

网络中学习的参数可以通过调用`net.parameters()`获取

```python
params = list(net.parameters())#获取网络中的学习的参数,转换成列表
print(len(params))
print(params[0].size())# conv1中的权重

'''
OUT
10
torch.Size([6, 1, 3, 3])
'''
```



现在假设随机产生跟一个32x32的输入，然后输入网络看会得到什么输出

```python
input = torch.randn(1,1,32,32)
out = net(input)
print(out)

'''
OUT
tensor([[-0.0282,  0.1006,  0.0217,  0.1202,  0.0275, -0.0728, -0.0243,  0.0983,
          0.0192,  0.1569]], grad_fn=<AddmmBackward>)
'''
```



# 损失函数

```python
input = torch.randn(1,1,32,32)
out = net(input)
target = torch.randn(10) #随机产生十个数
target = target.view(1,-1) #使target和output有一样的形状
criterion = nn.MSELoss() #使用均方差损失函数

loss = criterion(out,target)
print(loss)

'''
OUT
tensor(0.5529, grad_fn=<MseLossBackward>)
'''
```

整个计算图如下：

```python
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

`loss`的`grad_fn`属性肯定是`MseLossBackward`,上一个函数的`grad_fn`肯定是`Linear`,再上一个函数的`grad_fn`是`Relu`:

```python
print(loss.grad_fn) #MSELoss <MseLossBackward object at 0x000001EC323BA470>
print(loss.grad_fn.next_functions[0][0]) #Linear <AddmmBackward object at 0x000001EC323BA390>
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLu <AccumulateGrad object at 0x000001EC323BA390>
```



# 反向传播

要反向传播错误，我们要做的就是`loss.backward()`。不过，您需要清除现有的梯度，否则梯度将累积到现有的梯度中。

现在，我们将调用`loss.backward()`，并查看向后前后conv1的偏差梯度。

```python
net.zero_grad() #将梯度缓冲区中的所有参数置0
print(f"conv1.bias.grad before backward:{net.conv1.bias.grad}") #查看反向传播前conv1的偏差梯度

loss.backward() #因为loss为标量,直接调用backward()即可
print(f"conv1.bias.grad after backward:{net.conv1.bias.grad}")

'''
OUT
conv1.bias.grad before backward:None
conv1.bias.grad after backward:tensor([-0.0044,  0.0082, -0.0047,  0.0085, -0.0048, -0.0033])
'''
```



# 更新权重

实践中使用的最简单的更新规则是随机梯度下降（SGD）：

> ```
> `weight = weight - learning_rate * gradient`
> ```

想要使用各种不同的更新规则，例如SGD，Nesterov-SGD，Adam，RMSProp等。为实现此目的，我们构建了一个小程序包：`torch.optim`实现所有这些方法。使用它非常简单：

```python
optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad() #梯度缓冲区置0
output = net(input)
loss  = criterion(out,target)
loss.backward()
optimizer.step()
```