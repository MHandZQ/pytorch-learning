from __future__ import print_function
import torch
import numpy as np

"""
pytorch中的tensor是一个元组(tuple)
Tensor的创建：
内置函数：eye、zero、ones、rand
四种方法：
torch.Tensor(data)
torch.tensor(data)
torch.as_tensor(data)
torch.from_numpy(data)
"""

# 直接创建tensor
##方法一：
x = torch.Tensor([[1,2,3], 
                 [4,5,6],
                 [7,8,9]]) #torch.Tensor是python类，生成单精度浮点型张量
#print(x) 

##方法二：
x = torch.tensor([[1,2,3],
                 [4,5,6],
                 [7,8,9]])
#print(x) #torch.tensor是python函数，生成整型的张量

##方法三：
x = [[1,2,3],[4,5,6],[7,8,9]] #x是一个元组
x = torch.as_tensor(x) #把数据转换成tensor
#print(x)

##方法四：
x = np.array([[1,2,3],[4,5,6],[7,8,9]]) #x为numpy数组
#print(x)
x = torch.from_numpy(x) #把numpy数据变成tensor
#print(x)

# 通过内置函数创建tensor
x = torch.empty(5,3) #创建一个5行3列的tensor,未初始化
#print(x)

x = torch.zeros(5,3,dtype=torch.long) #创建一个5行3列的tensor,全部为0，类型为long型
#print(x)

x = torch.rand(5,3,4) #创建一个5个3行4列的tensor,随机初始化
#print(x)
#print(x[0]) #打印第一个3行4列所有元素
#print(x[0][0]) #打印第一个3行4列的第一行所有元素
#print(x[1::]) #打印第二个3行4列的元素到最后

x = x.new_ones(5,3,dtype=torch.double)
#print(x)

x = torch.rand_like(x,dtype=torch.float) #生成与x一样大小的tensor,然后随机初始化
#print(x)

#print(x.size()) #打印x的尺寸

"""
操作符：+ - * /
"""
#加法实现的几种方法
y = torch.rand(5,3)
#print(x+y)
#print(y.add_(x)) #注意时 _add()
#print(torch.add(x,y))
result = torch.empty(5,3)
torch.add(x,y,out=result)
#print(result)

"""
切片(slice)
是我自己理解错了，二维张量就应该有两个切片
第一个切片指示从多少行到多少行
第二个切片指示多少列到多少列
如果一个缺省就全部打印
"""
x = torch.rand(4,4)
#print(x)
#print(x[0:])
#print(x[0:,1:])

"""
Tensor Resize
"""
y = x.view(16)
#print(y,y.size())
z = x.view(-1,8) #-1的所对应的size是参考另一个维度的，比如这里另一个维度是8，那么-1对应的size是2
#print(z,z.size())

'''
One element tensor
'''
x  = torch.rand(1)
print(x)
print(x.item()) #只有一个元素的tensor,使用 .item() 获取这个变量作为Python数值

'''
Numpy Bridge
'''
###把Torch tensor转成Numpy array
a  = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

###把Numpy array转成Torch tensor
a = np.ones(5)
b = torch.as_tensor(a)
c = torch.from_numpy(a)# 两种方法都可以
np.add(a,1,out=a)
print(a)
print(b)# 两个都改变

'''
把tensor搬到确定的设备(device:cpu or gpu)
'''
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = rand(4,4)
    y = torch.ones_like(x,device=device) #直接在gpu上创建一个tensor
    x = x.to(device) #使用 .to() 函数把数据搬到gpu
    z = x+y
    print(z)
    print(z.to("cpu",torch.double)) #同样还可以使用 .to() 把数据搬回cpu