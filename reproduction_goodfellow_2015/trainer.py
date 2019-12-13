import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from time import time

def _accuracy(output, y):
    output = output.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    difference = np.count_nonzero(np.argmax(output, axis=1) - y)
    return (1 - (difference / np.size(y))) * 100

def test(raw_net, test_dataloader, learning_rate, epoch, num_epochs, description = ""):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        raw_net = torch.nn.DataParallel(raw_net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    begin = time()
    raw_net.eval()
    test_loss_counter = 0
    accuracy_test = 0
    with torch.no_grad():
        for j, (x_test, y_test) in enumerate(test_dataloader):
            x_test, y_test = x_test.to(device), y_test.to(device)
            output_test = raw_net(x_test)
            loss_test = criterion(output_test, y_test)

            accuracy_test += _accuracy(output_test, y_test)
            test_loss_counter += loss_test.item()

        accuracy_test = accuracy_test / (j+1)
        test_loss_counter = test_loss_counter / (j+1)

    print(  f'{description}, '
            f'epoch {epoch}/{num_epochs}, '
            f'test loss = {test_loss_counter:.4f}, '
            f'test accuracy = {accuracy_test:.4f}%, '
            f'lr = {learning_rate}. '
            f'Time {time()-begin:.4f}s.')



def train(raw_net, train_dataloader, test_dataloader = None, num_epochs = 100, learning_rate = 1, description = ""):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        raw_net = torch.nn.DataParallel(raw_net)
        cudnn.benchmark = True

    print(f"Running on {device}.")
    #pitch to GPU and define SGD optimiser
    raw_net = raw_net.to(device)
    raw_net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(raw_net.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=5e-4)

    #start iterating
    for epoch in range(num_epochs):
        #time how long a single epoch takes
        begin = time()
        if epoch % 33 == 0:
            learning_rate /= 10
            optimizer = torch.optim.SGD(raw_net.parameters(),
                                        lr=learning_rate,
                                        momentum=0.9,
                                        weight_decay=5e-4)
        
        for i, (x, y) in enumerate(train_dataloader):
            #send to GPU
            x, y = x.to(device), y.to(device)
            #normal net stuff
            raw_net.train()
            optimizer.zero_grad()
            output = raw_net(x)
            loss = criterion(output, y)   
            loss.backward()
            optimizer.step()

        if test_dataloader:
            test(raw_net, test_dataloader, learning_rate, epoch, num_epochs, description)

    return raw_net 


