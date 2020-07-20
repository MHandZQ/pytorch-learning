# README

## Tensor的创建

pytorch中的tensor是一个元组(tuple)

Tensor的创建：

内置函数：eye、zero、ones、rand

四种方法：

torch.Tensor(data)	\#torch.Tensor是python类，生成单精度浮点型张量

torch.tensor(data)	\#torch.tensor是python函数，生成整型的张量

torch.as_tensor(data)	\#把数据（不限时numpy）转换成tensor

torch.from_numpy(data	\#把numpy数据变成tensor



## Tensor操作符

操作符：+ - * /

```python
#加法实现的几种方法
y = torch.rand(5,3)
#print(x+y)
#print(y.add_(x)) #注意是 _add()
#print(torch.add(x,y))
result = torch.empty(5,3)
torch.add(x,y,out=result)
#print(result)
```



## 切片(slice)

是我自己理解错了，二维张量就应该有两个切片

第一个切片指示从多少行到多少行

第二个切片指示多少列到多少列

如果一个缺省就全部打印

```python
x = torch.rand(4,4)
print(x)
print(x[0:]) #这里缺省了列的切片，默认全打印，行的切片时从第一行到最后一行，所以这里实现对整个二维张量x的打印
print(x[0:,1:]) #这里行列切片都有，行：从第一行到最后一行；列：从第二列到最后一列
```



## Tensor Resize/Reshape

```python
x = torch.rand(4,4)

y = x.view(16)
z = x.view(-1,8) #-1的所对应的size是参考另一个维度的，比如这里另一个维度是8，那么-1对应的size是2
```



## One element tensor

```python
x  = torch.rand(1)
print(x)
print(x.item()) #只有一个元素的tensor,使用 .item() 获取这个变量作为Python数值
```



## Torch tensor2Numpy array

```python
a = torch.ones(5) #torch tensor
b = a,numpy() #numpy array
a.add_(1) #a改变 b也改变
```



## Numpy array2Torch tensor

```python
a = np.ones(5)
b = torch.as_tensor(a)
c = torch.from_numpy(a)# 两种方法都可以
np.add(a,1,out=a)
print(a)
print(b)# 两个都改变
```



## CUDA Tensor

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = rand(4,4)
    y = torch.ones_like(x,device=device) #直接在gpu上创建一个tensor
    x = x.to(device) #使用 .to() 函数把数据搬到gpu
    z = x+y
    print(z)
    print(z.to("cpu",torch.double)) #同样还可以使用 .to() 把数据搬回cpu
```