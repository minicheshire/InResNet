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

'''

Comments on CIFAR-100 training are the same with those in the FGSM attack file.

'''

saved_model_path = sys.argv[1]
eps = int(sys.argv[4]) / 255

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

M = 20
alpha = 2 / 255

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

for inputs, labels in now_dataloader:
    inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()


    cp_inputs = inputs.clone().detach()

    for i in range(M):
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        net.zero_grad()
        inputs_grad = torch.autograd.grad(loss, inputs)[0]
        inputs.data = inputs.data + (alpha * inputs_grad.data.sign())
        inputs = where(inputs > cp_inputs + eps, cp_inputs + eps, inputs)
        inputs = where(inputs < cp_inputs - eps, cp_inputs - eps, inputs)        
        inputs = inputs.clamp(min=0, max=1)

    outputs2 = net(inputs)

    _, predicted = torch.max(outputs2.data, 1)
    total_ctr += labels.size()[0]
    correct_sum += (predicted == labels.data).sum()

write_file_and_close(sys.argv[3], str(correct_sum.item() / total_ctr))
