from __future__ import print_function, division
import torch
import time
import copy
from torchvision import datasets, transforms
import os

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) #保存结果最好的模型权重,因为是预训练的网络,所以一开始就有权重,万一这就是最好的呢,所以这里也保存了
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())#这里保存的结果最好的模型的权重

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
