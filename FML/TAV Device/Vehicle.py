import time, DAN, requests, random 
import threading, sys 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from MAML.meta import Meta
from MAML.learner import Learner  
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

batch_size =5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Vehicle_data_train = pd.read_csv("dataset/anomalydetection_train.csv")

Vehicle_data_test = pd.read_csv("dataset/anomalydetection_test.csv")

# Split the data and labels
train_data = Vehicle_data_train.loc[:, Vehicle_data_train.columns != "Class"].values
train_label = Vehicle_data_train.Class.values

test_data = Vehicle_data_test.loc[:, Vehicle_data_test.columns != "Class"].values
test_label = Vehicle_data_test.Class.values

# Convert to tensors
train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label).type(torch.LongTensor)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)

print(len(train_data))
print(len(train_label))

print(len(test_data))
print(len(test_label))


# convert the dataloader
training_data = TensorDataset(train_data, train_label)
train_loader = DataLoader(training_data, batch_size = batch_size, shuffle = False)


config = [
        ('conv2d', [16, 7, 1, 1, 1, 0]),
        ('relu', [True]),
        ('max_pool2d', [1, 1, 0]),
        ('conv2d', [8, 16, 1, 1, 1, 0]),
        ('relu', [True]),
        ('max_pool2d', [1, 1, 0]),
        ('flatten', []),
        ('linear', [128, 8]),
        ('relu', [True]),
        ('linear', [1, 128])
    ]

local_model = Learner(config).to(device)
optimizer = optim.Adam(local_model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

loss_list = []
iteration_list = []
accuracy_list = []

num_epochs = 10

for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            local_model.train()
            train = images.reshape(images.shape[0],7,1,1)


            optimizer.zero_grad()

            outputs = local_model(train)
            labels = labels.view(-1, 1).to(torch.float32) 
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        local_model.eval()
        with torch.no_grad():
            test = test_data.view(test_data.size(0), 7, 1, 1).to(device)
            outputs = local_model(test)


            for idx, x in enumerate(outputs):
                outputs[idx] = torch.tensor([1]) if x > 0.5 else torch.tensor([0])
            outputs = outputs.view(-1)

            correct = torch.eq(outputs, test_label).sum().item()  # convert to numpy
            accs = correct / test.shape[0]

            # store loss and iteration
            loss_list.append(loss.item())
            iteration_list.append(epoch)
            accuracy_list.append(accs)
            
            print('Iteration: {}  Loss: {}  Accuracy: {}'.format(epoch, loss.item(), accs))


with torch.no_grad():
        test = test_data.reshape(test_data.shape[0],7, 1, 1)
        pred = local_model(test)

        print(f"[Before] predict result: {pred}")

        for idx, x in enumerate(pred):
            pred[idx] = torch.tensor([1]) if x > 0.5 else torch.tensor([0])
        pred = pred.view(-1)

        #print(f"[After] predict result: {pred}")

        correct = torch.eq(pred, test_label).sum().item()  # convert to numpy
        accs = correct / test_data.shape[0]
        print(correct)
        print(test_data.shape[0])
        print(f'accs: {accs}')
