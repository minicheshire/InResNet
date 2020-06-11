import torch
import os
from InResNet import *
import numpy as np
import sys

'''
Acknowledgement:

Plenty of lines reused from the codes of Lu et al., Beyond Finite Layer Neural Network: Bridging Deep Architects and Numerical Differential Equations, ICML 2018. Many thanks!
'''

torch.manual_seed(int(sys.argv[1]))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(int(sys.argv[1]))

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]

net = InResNet110() # Specify num_classes=100 for CIFAR-100 training
model_name = sys.argv[3]

from torch.autograd import Variable
inp=Variable(torch.FloatTensor(128,3,32,32).uniform_(0,1))

net.cuda()

import datetime

def write_file_and_close(filename, *arg, flag = "a"):
    with open(filename, flag) as output_file:
        output_file.write(str(datetime.datetime.now()))
        output_file.write(":\n")
        output_file.write(*arg)
        output_file.write("\n")
        print(*arg)

def check_control(filename):
    with open(filename, "r") as filename:
        try:
            u = int(filename.read().strip())
            return bool(u)
        except:
            write_file_and_close("Error occured checking control!!!")
            return False

def generate_filename(modelname,code = None):
    # if code = None, generate tim as the code
    if code == None:
        return sys.argv[2]+'result/'+modelname + '-' + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".mdlpkl"
    return sys.argv[2]+'result/'+modelname + '-test' + str(code) + ".mdlpkl"

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import re
import numpy
import os

import sys, time



def train_epoch(net,optimizer,trainloader,testloader,it,control_dict,global_output_filename = "out.txt"):
    """
    SGD Trainer For Cross Entropy Loss
    :param net:
    :param sgd_para: sgd parameters(should be a dictionary)
    :param trainloader:
    :param testloader:
    :param it: iteration time
    :param lr_adjust: learning rate adjust
    :param global_output_filename:
    :param global_control_filename:
    :return:None
    """
    # to do L2loss or other losses

    def lr_control(control_dict, it):
        for i in control_dict:
            if it <= i:
                return control_dict[i]

    criterion = nn.CrossEntropyLoss()

    global_cuda_available = torch.cuda.is_available()
    if global_cuda_available:
        net = net.cuda()



    def train(data, info):
        net.train()#turn to train mode
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        if global_cuda_available:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        info[0] = loss.item()
        info[1] = labels.size()[0]

    def test(info):
        net.eval()# trun to eval mode
        correct_sum = 0
        total_loss_sum = 0.
        total_ctr = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                if global_cuda_available:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_ctr += labels.size()[0]
                correct_sum += (predicted == labels.data).sum()
                loss = criterion(outputs, labels)
                total_loss_sum += loss.item()
        info[0] = correct_sum.item()
        info[1] = total_ctr
        info[2] = total_loss_sum

    running_loss_sum = 0.
    total_loss_sum = 0.
    ctr_sum = 0
    total_ctr = 0
    for g in optimizer.param_groups:
        g["lr"] = lr_control(control_dict,it)
    for i, data in enumerate(trainloader):
        info = [0., 0]
        train(data, info)
        running_loss_sum += info[0]
        total_loss_sum += info[0]
        ctr_sum += 1
        total_ctr += info[1]
        if (i + 1) % 20 == 0:
            write_file_and_close(global_output_filename,
                                 "epoch: {:d}, "
                                 "train set index: {:d}, "
                                 "average loss: {:.10f}"
                                 .format(it, i, running_loss_sum / ctr_sum)
                                 )
            running_loss_sum = 0.0
            ctr_sum = 0
        # it = it + 1
    write_file_and_close(global_output_filename,
                         "Epoch {:d} finished, average loss: {:.10f}"
                         .format(it, total_loss_sum / total_ctr)
                         )
    if True:
        write_file_and_close(global_output_filename, "Starting testing")
        info = [0., 0., 0.]
        test(info)
#        print(type(info[0]), type(info[1]), type(info[2]))
#        print(info[0], info[1], info[2])
        write_file_and_close(global_output_filename, net.print_ks())
        write_file_and_close(global_output_filename,
                             "Correct: {:d}, total: {:d}, "
                             "accuracy: {:.10f}, average loss: {:.10f}"
                             .format(info[0], info[1], info[0] / info[1], info[2] / info[1])
                             )
        return info[0], net.print_ks()


class NN_SGDTrainer(object):
    def __init__(self,net,sgd_para,trainloader,testloader,lr_adjust,global_output_filename = "out.txt"):
        self.net = net
        self.sgd_para = sgd_para
        self.optimizer = optim.SGD(net.parameters(), **sgd_para)
        self.trainloader = trainloader
        self.testloader = testloader
        self.output = global_output_filename
        self.lr_adjust = lr_adjust
        self.iter_time = 0
        self.max = 0

    def renew_trainer(self):
        self.optimizer = optim.SGD(self.net.parameters(), **self.sgd_para)

    def train(self,model_name="test"):
        self.iter_time += 1
        acc, parastr = train_epoch(self.net,self.optimizer,self.trainloader,self.testloader,self.iter_time,self.lr_adjust,self.output)
        if acc > self.max:
            model_filename = generate_filename(model_name, 1)
            torch.save(self.net, model_filename)
            self.max = acc
            self.maxs = str(parastr)
    
    def get_max_acc(self):
        return self.max, self.maxs
    
    def net_test(self):
        criterion = nn.CrossEntropyLoss()
        def test(info):
            self.net.eval()  # trun to eval mode
            correct_sum = 0
            total_loss_sum = 0.
            total_ctr = 0
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_ctr += labels.size()[0]
                correct_sum += (predicted == labels.data).sum()
                loss = criterion(outputs, labels)
                total_loss_sum += loss.data[0]
            info[0] = correct_sum
            info[1] = total_ctr
            info[2] = total_loss_sum

        write_file_and_close(self.output, "Starting testing")
        info = [0., 0., 0.]
        test(info)
        print(type(info[0]), type(info[1]), type(info[2]))
        print(info[0], info[1], info[2])
        write_file_and_close(self.output,
                             "Correct: {:d}, total: {:d}, "
                             "accuracy: {:.10f}, average loss: {:.10f}"
                             .format(info[0], info[1], info[0] / info[1], info[2] / info[1])
                             )


    def plot_loss(self,label,filename):
        extract_loss = re.compile(r"Epoch (\d+) finished, average loss: (\d.\d+)")
        arr_loss = []
        with open(self.output) as f:
            ctr = 0
            for line in f:
                match = extract_loss.match(line)
                if match:
                    arr_loss.append(match.group(2))
        nparr_loss = numpy.array(arr_loss)
        plt.plot(nparr_loss, label=label)
        plt.legend()
        plt.savefig(filename)

    def plot_train(self,lable,filename):
        extract_accuracy = re.compile(r"Correct: (\d+), total: (\d+), accuracy: (\d.\d+), average loss: (\d.\d+)")
        arr_acc = []
        with open(self.output) as f:
            ctr = 0
            for line in f:
                match = extract_accuracy.match(line)
                if match:
                    arr_acc.append(match.group(3))
        nparr_acc = numpy.array(arr_acc)
        plt.plot(nparr_acc, label=lable)
        plt.legend()
        plt.savefig(filename)

    def get_net(self):
        return self.net

    def write(self,out_txt):
        write_file_and_close(self.output,out_txt)

import torch.optim as optim
import os
import argparse

batch_size = 128


import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numbers
import functools
import numpy as np

"""
Cifar10 dataset Loading API
With data
@2prime 2017
"""


def expand_reflect(image, border=0):
    """
    Add border to the image(Symmetric padding)

    :param image: The image to expand.
    :param border: Border width, in pixels.
    :return: An image.
    """
    img = np.asarray(image)
    img = np.pad(img,pad_width=border,mode="reflect")
    return Image.fromarray(np.uint8(img[:,:,2:5]))

class Reflect_Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" reflect"""

    def __init__(self, padding):
        assert isinstance(padding, numbers.Number)
        self.padding = padding

    def __call__(self, img):
        return expand_reflect(img, border=self.padding)



def get_cifar10(batch_size):
    '''
    :param batch_size:
    :return: trainloader,testloader
    cifar10 dataset loader

    trainloader,testloader = get_cifar10(batch_size)
    '''
    transform_train = transforms.Compose([
        Reflect_Pad(2),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", download=True, train=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", download=True, train=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader,testloader

def get_cifar100(batch_size):
    '''
    :param batch_size:
    :return: trainloader,testloader
    cifar10 dataset loader

    trainloader,testloader = get_cifar10(batch_size)
    '''
    transform_train = transforms.Compose([
        Reflect_Pad(2),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root="./data", download=True, train=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data", download=True, train=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader,testloader


trainloader,testloader = get_cifar10(batch_size) # Change this line to get_cifar100 for CIFAR-100 training

sgd_para = {"lr":1e-3, "weight_decay":0.0001, "momentum":0.9}

Trainer = NN_SGDTrainer(net, sgd_para, trainloader, testloader, {80:1e-1,120:1e-2,160:1e-3}, sys.argv[2] + model_name + '.txt') # Change the learning rates to {150:1e-1,225:1e-2,300:1e-3} for CIFAR-100 training 

for i in range(160): # Change the total number of epochs to 300 for CIFAR-100 training
    Trainer.train()

Trainer.plot_loss(model_name,sys.argv[2]+model_name+'.png')

result_name = sys.argv[4]
acc, str0 = Trainer.get_max_acc()
write_file_and_close(result_name, sys.argv[1])
write_file_and_close(result_name, "\t"+str(acc/10000))
write_file_and_close(result_name, "\t"+str0)
write_file_and_close(result_name, "")
