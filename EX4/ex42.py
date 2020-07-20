import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:" if torch.cuda.is_available() else "cpu")

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