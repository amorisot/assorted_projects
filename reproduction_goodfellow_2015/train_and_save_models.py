import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from trainer import train
from models.vgg import VGG
from models.resnet import ResNet18
from models.mlp import SimpleMLP

#don't forget to check if the same is true for RL. High dependence between samples (unlike here where all images random)

transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
								transforms.RandomHorizontalFlip(),
								transforms.ToTensor(),
								transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
								)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

CIFAR_train = torchvision.datasets.CIFAR10("./data", train=True, transform=transform)
loader_train = DataLoader(CIFAR_train, batch_size=128, shuffle=True)

CIFAR_test = torchvision.datasets.CIFAR10("./data", train=False, transform=transform_test)
loader_test = DataLoader(CIFAR_test, batch_size=100, shuffle=True)

for i in range(5):
	description="VGG16, cifar"
	net = VGG('VGG16mod')
	# Device configuration
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device == 'cuda':
	    net = torch.nn.DataParallel(net)
	    cudnn.benchmark = True

	net = net.to(device)
	initial = torch.save(net.state_dict(), f"./trained_models/cifar_init_VGG16mod_{i}.pkl")
	new_net = train(net, loader_train, test_dataloader = loader_test, num_epochs=100, description=description)
	final = torch.save(new_net.module.state_dict(), f"./trained_models/cifar_final_VGG16mod_{i}.pkl")

for i in range(5):
	description="ResNet18, cifar"
	net = ResNet18()
	# Device configuration
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device == 'cuda':
	    net = torch.nn.DataParallel(net)
	    cudnn.benchmark = True

	net = net.to(device)
	initial = torch.save(net.state_dict(), f"./trained_models/cifar_init_Resnet18_{i}.pkl")
	new_net = train(net, loader_train, test_dataloader = loader_test, num_epochs=100, description=description)
	final = torch.save(new_net.module.state_dict(), f"./trained_models/cifar_final_Resnet18_{i}.pkl")


for i in range(5):
        description="MLP, cifar"
        net = SimpleMLP(channels = 3)
        # Device configuration
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        net = net.to(device)
        initial = torch.save(net.state_dict(), f"./trained_models/cifar_init_SimpleMLP_{i}.pkl")
        new_net = train(net, loader_train, test_dataloader = loader_test, num_epochs=100, description=description)
        final = torch.save(new_net.module.state_dict(), f"./trained_models/cifar_final_SimpleMLP_{i}.pkl")

# ###### START OF MNIST 

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
loader_test = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

for i in range(5):
	description = "VGG16, mnist"
	net = VGG('VGG16mod', in_channels = 1)
	# Device configuration
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device == 'cuda':
	    net = torch.nn.DataParallel(net)
	    cudnn.benchmark = True

	net = net.to(device)
	initial = torch.save(net.state_dict(), f"./trained_models/mnist_init_VGG16mod_{i}.pkl")
	new_net = train(net, loader_train, test_dataloader = loader_test, num_epochs=100, description=description)
	final = torch.save(new_net.module.state_dict(), f"./trained_models/mnist_final_VGG16mod_{i}.pkl")

for i in range(5):
	description = "ResNet18, mnist"
	net = ResNet18(num_channels = 1)
	# Device configuration
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	if device == 'cuda':
	    net = torch.nn.DataParallel(net)
	    cudnn.benchmark = True

	net = net.to(device)
	initial = torch.save(net.state_dict(), f"./trained_models/mnist_init_Resnet18_{i}.pkl")
	new_net = train(net, loader_train, test_dataloader = loader_test, num_epochs=100, description=description)
	final = torch.save(new_net.module.state_dict(), f"./trained_models/mnist_final_Resnet18_{i}.pkl")

for i in range(5):
        description = "MLP, mnist"
        net = SimpleMLP(channels = 1)
        # Device configuration
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        net = net.to(device)
        initial = torch.save(net.state_dict(), f"./trained_models/mnist_init_SimpleMLP_{i}.pkl")
        new_net = train(net, loader_train, test_dataloader = loader_test, num_epochs=100, description=description)
        final = torch.save(new_net.module.state_dict(), f"./trained_models/mnist_final_SimpleMLP_{i}.pkl")