import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
import re
import sys, os
from InResNet import *
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

saved_model_path = sys.argv[1]
eps_ = int(sys.argv[4]) / 255 # The specified attack radius

net = torch.load(saved_model_path)
net.cuda()

def write_file_and_close(filename, *arg, flag = "a"):
    with open(filename, flag) as output_file:
        output_file.write(str(datetime.datetime.now()))
        output_file.write(":\n")
        output_file.write(*arg)
        output_file.write("\n")
        print(*arg)

net.eval()
criterion = nn.CrossEntropyLoss()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
testset = torchvision.datasets.CIFAR10( # Change this to CIFAR100 if conducting FGSM attack on CIFAR-100
    root="./data", download=True, train=False, transform=transform_test
)
now_dataloader = torch.utils.data.DataLoader(
    testset, batch_size=400, shuffle=False, num_workers=2
)

correct_sum = 0
total_ctr = 0

for inputs, labels in now_dataloader:
    inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()

    """
        The somewhat odd scaling on eps and torch.clamp is due to the normalization operation over dataset (Line #36).
        We provide the codes as an alternative way to calculate equivalent robustness scores with those who have normalized the dataset when training; a common approach is not to use Line #36 (and thus no need to scale eps and torch.clamp).
    """
    eps = torch.ones_like(inputs) * eps_
    eps = torch.stack((eps[:,0,:,:] / 0.2023, eps[:,1,:,:] / 0.1994, eps[:,2,:,:] / 0.2010), dim=1).cuda()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    net.zero_grad()
    inputs_grad = torch.autograd.grad(loss, inputs)[0]

    inputs.data = inputs.data + (eps * inputs_grad.data.sign())
    inputs = torch.stack((torch.clamp(inputs[:,0,:,:], min=(0-0.4914)/0.2023, max=(1-0.4914)/0.2023),
                          torch.clamp(inputs[:,1,:,:], min=(0-0.4822)/0.1994, max=(1-0.4822)/0.1994),
                          torch.clamp(inputs[:,2,:,:], min=(0-0.4465)/0.2010, max=(1-0.4465)/0.2010)), dim=1)
    #inputs = inputs.clamp(min=0, max=1)

    outputs2 = net(inputs)

    _, predicted = torch.max(outputs2.data, 1)
    total_ctr += labels.size()[0]
    correct_sum += (predicted == labels.data).sum()

write_file_and_close(sys.argv[3], str(correct_sum.item() / total_ctr))
