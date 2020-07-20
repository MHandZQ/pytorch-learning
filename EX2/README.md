# README

# AUTOGRAD:自动差分

`autograd`软件包**为张量上的所有操作提供了自动求导机制**。这是一个按运行定义的框架，这意味着backprop是由代码的运行方式定义的，并且每次迭代都可以不同。



# Tensor

`torch.Tensor`是程序包的中心类。如果将其属性设置 `.requires_grad`为`True`，它将开始跟踪对其的所有操作。完成计算后，可以调用`.backward()`并自动计算所有梯度。该张量的梯度将累加到`.grad`属性中。

要停止张量跟踪历史记录，可以调用`.detach()`将其从计算历史记录中分离出来，并防止跟踪将来的计算。

为了防止跟踪历史记录（和使用内存），**还可以将代码包装在`with torch.no_grad():`中。**

**这在评估模型时特别有用，因为模型可能具有可训练的参数，且`requires_grad=True`，但我们不需要在此过程中对他求梯度。**

还有一个对autograd实现非常重要的类： `Function`。



`Tensor`和`Function`相互连接并建立一个非循环图，该图对完整的计算历史进行编码。 每个张量都有一个`.grad_fn`属性，该属性引用创建了张量的函数（**用户创建的张量除外，它们的`grad_fn`为`None`**）。



为方便说明，对于这种我们自己定义的变量，我们称之为**叶子节点(leaf nodes)**，而基于叶子节点得到的中间或最终变量则可称之为**结果节点**。例如，下面例子中`x`、`b`就是叶子节点，`y`、`z`、`out`、`b`就是结果节点。



另外一个**Tensor**中通常会记录如下图中所示的**属性**：

- `data`: 即存储的数据信息
- `requires_grad`: 设置为`True`则表示该Tensor需要求导
- `grad`: 该Tensor的梯度值，每次在计算backward时都需要将前一时刻的梯度归零，否则梯度值会一直累加，这个会在后面讲到。
- `grad_fn`: **叶子节点通常为None，只有结果节点的grad_fn才有效，用于指示梯度函数是哪种类型**。例如下面示例代码中`x.grad_fn=None, y.grad_fn=<AddBackward0 object at 0x0000020746CE4C50>, z.grad_fn=<MulBackward0 at 0x2135df11be0>`
- `is_leaf`: 用来指示该Tensor是否是叶子节点。

```python
x = torch.ones(2,2,requires_grad=True)
print(x)

y = x+2
print(y)

print(x.grad_fn) #None，用户自己创建的tensor的.grad_fn为None
print(y.grad_fn) #每个张量都有一个.grad_fn属性，该属性引用创建了张量的函数。<AddBackward0 object at 0x0000020746CE4C50>

z = y*y*3
out = z.mean()
print(z,out) #z这个张量的.grad_fn属性增加了乘Mul,out增加了求均值Mean

a = torch.rand(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)# 在一开始没有给定requires_grad的话,默认是False
a.requires_grad = True
b = (a*a).sum()
print(b.grad_fn)# <SumBackward0 object at 0x000002B83E7E1F98>
```



# Gradients

如果要计算导数，可以在`Tensor`上调用`.backward（）`。如果`Tensor`为标量（即，它包含一个元素数据），则无需为`backward()`指定参数，但是，如果它具有更多元素，则需要指定一个`gradient` 参数，该参数为匹配形状的张量。

```python
out.backward()#因为out是只一个标量, out.backward() 等效 out.backward(torch.tensor(1.))

print(x.grad) #打印导数 d(out)/dx;          OUT:tensor([[4.5000, 4.5000],[4.5000, 4.5000]])
```

正如上面所说，因为`out`是一个标量，可以直接调用`backward`而不需要指定参数，会根据链式法则直接求出叶子节点的梯度，整个过程如下：

**Let’s call the `out` *Tensor* $“o”$. We have that $o=1/4∑_iz_i, z_i=3(x_i+2)^2$ and$ z_i∣_{x_i=1}=27$. Therefore, $∂o/∂x_i=∂o/∂z_i*∂z_i/∂x_i=1/4*6(x_i+2)=3/2(x_i+2)$, hence$ ∂o/∂x_i∣_{x_i=1}=9/2=4.5$**



但是**如果遇到`out`是一个向量或者是一个矩阵的情况，这个时候又该怎么计算梯度呢**？假如还用上面的方法：

```python
import torch

x = torch.ones(2, requires_grad=True)
z = x+2
z.backward()
### RuntimeError: grad can be implicitly created only for scalar outputs
### 会报错，意思是只有对标量输出才会计算梯度，不能求一个矩阵对另一个矩阵的导数
```



## grad_tensor

所以对于这种情况我们需要定义`grad_tensor`来计算矩阵的梯度。在介绍为什么使用之前我们先看一下源代码中backward的接口是如何定义的：

```python
torch.autograd.backward(
		tensors, 
		grad_tensors=None, 
		retain_graph=None, 
		create_graph=False, 
		grad_variables=None)
```

- `tensor`: 用于计算梯度的tensor。也就是说这两种方式是等价的：**`torch.autograd.backward(z) == z.backward()`**
- `grad_tensors`: **在计算矩阵的梯度时会用到。他其实也是一个tensor，shape一般需要和前面的`tensor`保持一致。**
- `retain_graph`: **通常在调用一次backward后，pytorch会自动把计算图销毁，所以要想对某个变量重复调用backward，则需要将该参数设置为`True`**
- `create_graph`: 当设置为`True`的时候可以用来计算更高阶的梯度
- `grad_variables`: 这个官方说法是grad_variables' is deprecated. Use 'grad_tensors' instead.也就是说这个参数后面版本中应该会丢弃，直接使用`grad_tensors`就好了。

继续上面的例子：

$X = \left[\begin{array}{cc} x_0 & x_1 \\ \end{array}\right] \,\,\,\,\,\,\,\,\,\ Z=X+2=\left[\begin{array}{cc} x_0+2 & x_1+2 \\ \end{array}\right] \Rightarrow \frac{\partial{Z}}{\partial{X}}=?$

我们想既然只能求标量输出，那我们把他变成标量就可以了,这里可以对$Z$求和，然后用和再对$X$求导：

$\begin{align} &Z_{sum}=\sum{z_i}=x_0+x_1+4 \notag \\ &\text{then} \,\,\,\,\,  \frac{\partial{Z_{sum}}}{\partial{x_0}}=\frac{\partial{Z_{sum}}}{\partial{x_1}}=1 \notag \end{align}$

我们用代码实现一下：

```python
import torch

x = torch.ones(2, requires_grad=True)
z = x+2
z.sum().backward()
print(x.grad)  #out = tensor([1.,1.])
```

我们再仔细想想，对z求和不就是等价于z**点乘**一个一样维度的全为1的矩阵吗？即$sum(Z)=dot(Z,I)$而这个$I$也就是我们需要传入的`grad_tensors`参数。(点乘只是相对于一维向量而言的，对于矩阵或更高为的张量，可以看做是对每一个维度做点乘)

代码如下：

```python
import torch

x = torch.ones(2,requires_grad=True)
z = x+2
z.backward(torch.ones_like(z))
print(x.grad) #OUT = tensor([1.,1.])
```



在数学上，如果有一个向量函数$\vec{y}=f(\vec{x})$，那么$\vec{y}$关于$\vec{x}$的梯度是雅可比矩阵：

$\begin{split}J=\left(\begin{array}{ccc} \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\ \vdots & \ddots & \vdots\\ \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}} \end{array}\right)\end{split}$

总的来说，**`torch.autograd`就是用来计算雅可比向量积的**引擎。比如说，给定一个向量$v=\left(\begin{array}{cccc} v_{1} & v_{2} & \cdots & v_{m}\end{array}\right)^{T}$，计算$v^{T}\cdot J$即可。

如果$v$是一个标量函数$l=g\left(\vec{y}\right)$的梯度，即，$v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}$，然后根据链式准则，向量-雅各比积将是$l$关于$\vec{x}$的梯度：

$\begin{split}J^{T}\cdot v=\left(\begin{array}{ccc} \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\ \vdots & \ddots & \vdots\\ \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}} \end{array}\right)\left(\begin{array}{c} \frac{\partial l}{\partial y_{1}}\\ \vdots\\ \frac{\partial l}{\partial y_{m}} \end{array}\right)=\left(\begin{array}{c} \frac{\partial l}{\partial x_{1}}\\ \vdots\\ \frac{\partial l}{\partial x_{n}} \end{array}\right)\end{split}$

（注意：行向量的 ![[公式]](https://www.zhihu.com/equation?tex=+v%5E%7BT%7D%5Ccdot+J) 也可以被视作列向量的 ![[公式]](https://www.zhihu.com/equation?tex=J%5E%7BT%7D%5Ccdot+v) )

向量雅可比积的这一特性使得**将外部梯度输入**具有非标量输出的模型变得非常方便。如上面out.backward() 等效 out.backward(torch.tensor(1.))中的1.0就是外部梯度，**`torch.tensor(1.)`是用来终止链式法则梯度乘法的外部梯度**。

**传递到`.backward()`中的张量的维数必须与正在计算梯度的张量的维数相同。**



再举一个例子：

```python
import torch

x = torch.tensor([[2.,1.]], requires_grad=True) #这里创建tensor的时候不能再加其他的操作了(.to(device) .view())，否则就不再是叶子节点,参考的文章里这里是错误的。
y = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)

z = torch.mm(x, y)
print(f"z:{z}")

z.backward(torch.Tensor([[1., 0]]), retain_graph=True)
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")

'''
OUT:
>>> z:tensor([[5., 8.]], grad_fn=<MmBackward>)
x.grad: tensor([[1., 3.]])
y.grad: tensor([[2., 0.],
        [1., 0.]])
'''
```

结果解析：

![grad_tensor](E:\PytorchLearning\EX2\grad_tensor.png)

这里我有一个问题就是：为什么就把`grad_tensor`设成`[[1,0]]`,完全可以设成其他的啊。事实证明确实是可以的。**`grad_tensors`的作用其实可以简单地理解成在求梯度时的权重，因为可能不同值的梯度对结果影响程度不同，所以pytorch弄了个这种接口，而没有固定为全是1。**



## torch.autograd.grad

```python
torch.autograd.grad(
		outputs, 
		inputs, 
		grad_outputs=None, 
		retain_graph=None, 
		create_graph=False, 
		only_inputs=True, 
		allow_unused=False)
```

看了前面的内容后在看这个函数就很好理解了，各参数作用如下：

- `outputs`: 结果节点，即被求导数
- `inputs`: 叶子节点
- `grad_outputs`: **类似于`backward`方法中的`grad_tensors`**
- `retain_graph`: 同上
- `create_graph`: 同上
- `only_inputs`: 默认为`True`, 如果为`True`, 则只会返回指定`input`的梯度值。 若为`False`，则会计算所有叶子节点的梯度，并且将计算得到的梯度累加到各自的`.grad`属性上去。
- `allow_unused`: 默认为`False`, 即必须要指定`input`,如果没有指定的话则报错。



## with torch.no_grad

也可以通过将代码块包装在 `with torch.no_grad():` 中，来阻止autograd跟踪设置了 `.requires_grad=True` 的张量的历史记录。（这常用在验证集上，不需要跟新它的参数）

```python
print(x.requires_grad) #True
print((x ** 2).requires_grad) #True

with torch.no_grad():
    print((x ** 2).requires_grad) #False
```



# Use Pytorch freeze network in some layers

use pytorch freeze network in some layers, only the rest of the training

https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088

大概原理就是把参数的`.requires_grad`设为`False`