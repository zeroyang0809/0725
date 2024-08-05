import time, requests, random 
import threading, sys # for using a Thread to read keyboard INPUT

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


input_shape = (7, 1, 1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x