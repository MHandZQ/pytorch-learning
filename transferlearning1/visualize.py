from __future__ import print_function, division

import torch
import torch.nn as nn 
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt 
from torchvision import datasets, transforms,models
import os
import numpy as np


#可视化数据集
def imshow(img, title=None):
    img = img.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    #np.clip(x,a_min,a_max)也就是说clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated

def visualize_model(num_images=6):
    plt.ion() #交互模式
    data_transforms = {
        'train':transforms.Compose([
        	transforms.RandomResizedCrop(224),
        	transforms.RandomHorizontalFlip(),
        	transforms.ToTensor(),
        	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    	]),
    	'val':transforms.Compose([
        	transforms.Resize(256),
        	transforms.CenterCrop(224),
        	transforms.ToTensor(),
        	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    	]),
    }

    data_dir = './hymenoptera_data'
	#构建数据集：trainset valset, 返回值是图片路径以及类别的索引值
	#在image_datasets中有两个set:trainset、valset. 每一个set中有class_to_idx{'ants': 0, 'bees': 1},有类名classes['ants', 'bees']
	#有imgs[('E:/PytorchLearning/T...013035.jpg', 0), ('E:/PytorchLearning/T...c608f9.jpg', 0), ('E:/PytorchLearning/T...d8afde.jpg', 0), ('E:/PytorchLearning/T...9d3250.jpg', 0), ('E:/PytorchLearning/T...26745d.jpg', 0), ('E:/PytorchLearning/T...56588f.jpg', 0), ('E:/PytorchLearning/T...ada201.jpg', 0), ('E:/PytorchLearning/T...92cdab.jpg', 0), ('E:/PytorchLearning/T...e80de1.jpg', 0), ('E:/PytorchLearning/T...0adea2.jpg', 0), ('E:/PytorchLearning/T...8c5eea.jpg', 0), ('E:/PytorchLearning/T...e2fb6d.jpg', 0), ('E:/PytorchLearning/T...aacea6.jpg', 0), ('E:/PytorchLearning/T...84f5a4.jpg', 0), ...]
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x] )  for x in ['train', 'val']}
	#数据加载器:加载数据集然后进行处理:batch、shuffle、sample等
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val']}
	#数据集大小以及类名
    dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #创建一个跟训练时一样的模型,把训练好的模型参数加载进来
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    model.load_state_dict(torch.load('./model_ft.pth',map_location='cpu'))

    #测试模型
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad(): #实现不需要求梯度
        for data in dataloaders['val']:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            if images_so_far > num_images:
                break
            for j in range(inputs.size()[0]):
                images_so_far += 1
                if images_so_far > num_images:
                    break
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}/label: {}'.format(class_names[preds[j]],class_names[labels[j]]))
                imshow(inputs.cpu().data[j])
    #关闭交互模式
    plt.ioff()
    #显示结果图,不再关闭
    plt.show()
    
