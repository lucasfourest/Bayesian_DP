import argparse
import itertools
import numpy as np
import pandas as pd
import os
import random
import time
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.optimizer import required
from torch.autograd import Variable
from torch.autograd import Function

from accountant import *
from utils import *



class Net(nn.Module):
    
    def __init__(self,input_dim,n_class=10,fsize=5):
        super(Net, self).__init__()
        input_channels, h, w = input_dim
        crop = 2*int((fsize-1)/2)
        self.h_out, self.w_out= int((((h - crop)/2) - crop)/2), int((((w - crop)/2) - crop)/2)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, fsize),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, fsize),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2))
        
            
        self.linear = nn.Sequential(nn.Linear(64 * self.w_out * self.h_out, 384),
            nn.SELU(inplace=True),
            nn.Linear(384, 192),
            nn.SELU(inplace=True),
            nn.Linear(192, n_class),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x=self.features(x)
        x=x.view(-1, 64 * self.w_out * self.h_out)
        return self.linear(x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', choices=['mnist','cifar10'])
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--ne', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval', type=int, default=5, help='evaluate every ... epochs')
    parser.add_argument('--C', type=float, default=1.0, help='clipping norm in private SGD')
    parser.add_argument('--sigma', type=float, default=0.1, help='noise variance')
    parser.add_argument('--n_samples', type=int, default=8, help='number of neighboring dataset gradient \
                        samples for cost estimator')
    
    args=parser.parse_args()


    # pre-settings
    use_cuda=torch.cuda.is_available()
    if use_cuda:
        DEVICE = torch.device('cuda')
        print("GPU:", torch.cuda.get_device_name(0)) 
    else:
        DEVICE = torch.device('cpu')
        print("No GPU, using CPU")

    seed=123
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


    # Data loading
        
    try:
        os.makedirs('/data')
    except OSError:
        pass
    
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
    labels = trainset.targets
    n_class = max(labels)+1
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True,drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=True)

    img,label = next(iter(trainloader))
    print(f'test input: {img.shape, label.shape}')
    input_dim = img.shape[1:]
    h,w,input_channels=input_dim[-2], input_dim[-1],input_dim[-3]
    print(f'image (h,w) = {h,w} ; n classes = {n_class} ; n of input channels = {input_channels}')
    print(f'q = {args.bs/len(trainset)}')


    # Create network

    net = Net(input_dim,n_class)
    test_output = net(img)
    print(f'test output: {test_output.shape}')

    # Training
    # powers=None
    powers=np.array([2, 4, 8, 16, 32])
    train(args,trainloader, testloader, net, powers=powers)

    pass