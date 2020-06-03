import os
import sys
import time
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from models.LSTM import lstm,gru
from utils.trainUtils import train_isolated
from utils.testUtils import test_isolated
from datasets import CSL_Isolated_Openpose_fixed,CSL_Isolated_Openpose
from utils.ioUtils import *
from torch.utils.tensorboard import SummaryWriter

# Hyper params
learning_rate = 1e-5
batch_size = 1
epochs = 1000
num_class = 500
num_joints = 116
length = 32
dropout = 0.1
# Options
store_name = 'test_LSTM_isolated'
summary_name = 'runs/' + store_name
checkpoint = '/home/liweijie/projects/SLR/checkpoint/20200513_LSTM_isolated_best.pth.tar'
device_list = '1'
log_interval = 100

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use writer to record
writer = SummaryWriter(os.path.join(summary_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

best_prec1 = 0.0
start_epoch = 0

# Train with Transformer
if __name__ == '__main__':
    # Load data
    trainset = CSL_Isolated_Openpose('trainval',length=length)
    devset = CSL_Isolated_Openpose('test',length=length)
    print("Dataset samples: {}".format(len(trainset)+len(devset)))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(devset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # Create model
    model = lstm(input_size=num_joints*2,hidden_size=512,hidden_dim=512,
        num_layers=3,dropout_rate=dropout,num_classes=num_class,
        bidirectional=True).to(device)
    if checkpoint is not None:
        start_epoch, best_prec1 = resume_model(model,checkpoint)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Start Evaluation
    print("Evaluation Started".center(60, '#'))
    for epoch in range(start_epoch, start_epoch+1):
        # Test the model
        prec1 = test_isolated(model, criterion, testloader, device, epoch, log_interval, writer)
        print('Epoch best acc: %.3f'%prec1)
    print("Evaluation Finished".center(60, '#'))


