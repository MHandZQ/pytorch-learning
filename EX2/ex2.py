import torch
import math

'''
Tensor
'''

x = torch.ones(2,2,requires_grad=True)
#print(x)

y = x+2
#print(y)

#print(x.grad_fn) #None，用户自己创建的tensor的.grad_fn为None
#print(y.grad_fn) #每个张量都有一个.grad_fn属性，该属性引用创建了张量的函数。<AddBackward0 object at 0x0000020746CE4C50>

z = y*y*3
out = z.mean()
print(z,out) #z这个张量的.grad_fn属性增加了乘Mul,out增加了求均值Mean

a = torch.rand(2,2)
a = ((a*3)/(a-1))
#print(a.requires_grad)# 在一开始没有给定requires_grad的话,默认是False
a.requires_grad = True
b = (a*a).sum()
#print(b.grad_fn)# <SumBackward0 object at 0x000002B83E7E1F98>

'''
Gradients
'''
out.backward()#因为out是只一个标量, out.backward() 等效 out.backward(torch.tensor(1.))

print(x.grad) #打印导数 d(out)/dx;          OUT:tensor([[4.5000, 4.5000],[4.5000, 4.5000]])

x = torch.rand(3,requires_grad=True)

y = x*2
while y.data.norm()<1000:
    y = y*2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)# [0.1, 1.0, 0.0001]为外部梯度
y.backward(v) #传递到`.backward()`中的张量的维数必须与正在计算梯度的张量的维数相同。

print(x.grad)# 雅可比向量积

print(x.requires_grad) #True
print((x ** 2).requires_grad) #True

with torch.no_grad():
    print((x ** 2).requires_grad) #False