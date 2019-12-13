import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from time import time
from models.vgg import VGG
from models.resnet import ResNet18
from models.mlp import SimpleMLP

def mixnmatch(init, final, alpha, model, num_channels):
    if num_channels == 'VGG16mod_1chan':
        mixed = model('VGG16mod', 1)
    elif num_channels == 'VGG16mod_3chan':
        mixed = model('VGG16mod', 3)
    else:    
        mixed = model(num_channels)
    #do the mixing and matching
    for w in mixed.state_dict():
        alphaed = (1-alpha)*init.state_dict()[w].data + alpha*final.state_dict()[w].data
        mixed.state_dict()[w].data.copy_(alphaed)
    return mixed

def unModule(PATH, model):
    state_dict = torch.load(PATH)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def _accuracy(output, y):
    output = output.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    difference = np.count_nonzero(np.argmax(output, axis=1) - y)
    return (1 - (difference / np.size(y))) * 100

def makePlot(datasetName, modelName, dataset, model, num_channels):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    for i in range(5):
        PATH_init = f'trained_models/{datasetName}_init_{modelName}_{i}.pkl'
        PATH_final = f'trained_models/{datasetName}_final_{modelName}_{i}.pkl'

        if num_channels == 'VGG16mod_1chan':
            init = unModule(PATH_init, model('VGG16mod', 1))
            final = unModule(PATH_final, model('VGG16mod', 1))
        elif num_channels == 'VGG16mod_3chan':
            init = unModule(PATH_init, model('VGG16mod', 3))
            final = unModule(PATH_final, model('VGG16mod', 3))
        else:
            init = unModule(PATH_init, model(num_channels))
            final = unModule(PATH_final, model(num_channels))

        init.eval()
        final.eval()

        init.to(device)
        final.to(device)

        alphas = np.linspace(0, 1, 50)
        losses = []

        with torch.no_grad():
            for k, alpha in enumerate(alphas):
                matched = mixnmatch(init, final, alpha, model, num_channels)
                matched.to(device)
                matched.eval()
                
                sum_loss = 0
                accuracy = 0
                for j, (x, y) in enumerate(dataset):
                    x, y = x.to(device), y.to(device)
                    output = matched(x)
                    loss = criterion(output, y)

                    sum_loss += loss.item()
                    accuracy += _accuracy(output, y)
                
                losses += [sum_loss / (j+1)]
                accuracy /= (j+1)
                print("loss: ", losses[k], " accuracy: ", accuracy, " alpha: ", alpha)

        plt.plot(alphas, losses)

    plt.ylabel('Loss')
    plt.xlabel('Alpha')
    plt.title(f'{datasetName}, {modelName}')
    plt.savefig(f"plots/{datasetName}_{modelName}.png")
    plt.clf()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_test = torchvision.datasets.CIFAR10("./data", train=False, transform=transform_test)
cifar_loader = DataLoader(cifar_test, batch_size=100, shuffle=False)


transform_test = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

mnist_test = torchvision.datasets.MNIST('./data', train=False, transform=transform_test)
mnist_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

makePlot('mnist', 'SimpleMLP', mnist_loader, SimpleMLP, 1)
makePlot('mnist', 'Resnet18', mnist_loader, ResNet18, 1)
makePlot('mnist', 'VGG16mod', mnist_loader, VGG, 'VGG16mod_1chan')

makePlot('cifar', 'Resnet18', cifar_loader, ResNet18, 3)
makePlot('cifar', 'SimpleMLP', cifar_loader, SimpleMLP, 3)
makePlot('cifar', 'VGG16mod', cifar_loader, VGG, 'VGG16mod_3chan')