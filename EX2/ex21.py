import torch

x = torch.ones(2, requires_grad=True)
z = x+2

#z.sum().backward() ##把z变成标量
#print(x.grad)


z.backward(torch.ones_like(z)) ##利用grad_tensor
print(x.grad)