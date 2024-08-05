#improt相关的套件
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

batch_size = 16
# read the data
Vehicle_data =pd.read_csv("data/anomalydetection_train.csv")
data = Vehicle_data.loc[:, Vehicle_data.columns != "Class"].values
label = Vehicle_data.Class.values

#split the train set and test set
train_data, test_data, train_label, test_label = train_test_split( 
                                                  data,
                                                  label,
                                                  test_size = 0.2,
                                                  random_state = 42)

# convert to tensor
train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label).type(torch.LongTensor) 

# convert to tensor
test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor) 

# convert the dataloader
training_data = TensorDataset(train_data, train_label)
train_loader = DataLoader(training_data, batch_size = batch_size, shuffle = False)