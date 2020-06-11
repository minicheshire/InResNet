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
net = torch.load(saved_model_path)
net.cuda()

TEST_PREFIX = "/PATH_TO_SAVED_DATASET/CIFAR-10-C/" # The directory that saves the CIFAR-10-C / CIFAR-100-C (unzipped) dataset (${TEST_PREFIX}/labels.npy should exist)
label_y = torch.from_numpy(np.load(TEST_PREFIX + "labels.npy")).long()
data_name_total = ["impulse_noise.npy", "speckle_noise.npy", "gaussian_noise.npy", "shot_noise.npy"]

def write_file_and_close(filename, *arg, flag = "a"):
    with open(filename, flag) as output_file:
        output_file.write(str(datetime.datetime.now()))
        output_file.write(":\n")
        output_file.write(*arg)
        output_file.write("\n")
        print(*arg)

net.eval()
criterion = nn.CrossEntropyLoss()
ans = []

for now_data_name in data_name_total:
    now_np_data = np.load(TEST_PREFIX + now_data_name) / 255
    now_mean, now_std = np.mean(now_np_data, (0,1,2)), np.std(now_np_data, (0,1,2))
    now_np_data = (now_np_data - now_mean) / now_std
    now_dataset = data.TensorDataset(torch.from_numpy(now_np_data).permute(0, 3, 1, 2).float(), label_y)
    now_dataloader = data.DataLoader(now_dataset, batch_size=500, shuffle=False, num_workers=2)

    correct_sum = 0
    total_loss_sum = 0.
    total_ctr = 0
    for inputs, labels in now_dataloader:
        inputs, labels = Variable(inputs), Variable(labels)
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_ctr += labels.size()[0]
        correct_sum += (predicted == labels.data).sum()

    ans.append(correct_sum.item() / total_ctr)

write_file_and_close(sys.argv[3], str(ans))
