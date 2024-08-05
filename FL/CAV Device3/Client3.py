import time, DAN, requests, random 
import threading, sys 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from MAML.meta import Meta
from MAML.learner import Learner  

#set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ServerURL = 'http://140.114.77.93:9999'
Reg_addr = "Client3_" + str( random.randint(1,1000 )) #if None, Reg_addr = MAC address
DAN.profile['dm_name']='client3'   # you can change this but should also add the DM in server
DAN.profile['df_list']=['loss', 'model', 'parameter']   # Check IoTtalk to see what IDF/ODF the DM has
DAN.profile['d_name']= "."+ str( random.randint(1,1000 ) ) +"_"+ DAN.profile['dm_name'] # None
DAN.device_registration_with_retry(ServerURL, Reg_addr)
print("dm_name is ", DAN.profile['dm_name']) 



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


rounds = 10  # Number of training rounds
waiting_limit = 60

batch_size = 5

try:
    # Read the vehicle data
    
    Vehicle_data =pd.read_csv("Local Dataset/anomalydetection_train.csv")
    data = Vehicle_data.loc[:, Vehicle_data.columns != "Class"].values
    label = Vehicle_data.Class.values

    train_data, test_data, train_label, test_label = train_test_split( 
                                                    data,
                                                    label,
                                                    train_size = 0.99,
                                                    random_state = 42)
    
    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor) 

    # convert the dataloader
    training_data = TensorDataset(train_data, train_label)
    train_loader = DataLoader(training_data, batch_size = batch_size, shuffle = False)


    local_model = Learner(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=0.01)




    DAN.device_registration_with_retry(ServerURL, Reg_addr) 
    input('CAV device3 Ready?  (y/n)\n')

    for r in range(rounds):
      end = 0
      recv_param = None
      
      # try to receive data from server (global model)
      while recv_param == None:
            print('CAV device3 is waiting for parameter')
            time.sleep(0.5)
            recv_param = DAN.pull('model')
            end += 1
            if end > waiting_limit: 
                break
      if end > waiting_limit: 
            break
      print('CAV device3 got model!!!')

      # take out & turn receive data (type: list) into tensor
      recv_param = [torch.tensor(p) for p in recv_param[0]]

      # set the local model weight to receive data
      for idx, (name, param) in enumerate(local_model.named_parameters()):
         param.data = recv_param[idx]

      for epoch in range(5):
        for i, (images, labels) in enumerate(train_loader):
            train = images.reshape(images.shape[0],7,1,1)
            optimizer.zero_grad()

            outputs = local_model(train)

            labels = labels.view(-1, 1).to(torch.float32) 
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
      
      # pack local model weight and send to server
      local_weights = [param.data.tolist() for _, param in local_model.named_parameters()]
      
      DAN.push ('parameter', local_weights)
      
    print(f'Training Success!! Total {rounds} epoch')
except Exception as e:
        print(e)
        if str(e).find('mac_addr not found:') != -1:
            print('Reg_addr is not found. Try to re-register...')
            DAN.device_registration_with_retry(ServerURL, Reg_addr)
        else:
            print('Connection failed due to unknow reasons.')
            time.sleep(1)    
finally:
    print('Client1 Complete!')
    time.sleep(0.25)
    try: 
        DAN.deregister()    # 試著解除註冊
    except Exception as e:
        print("===")
    print("Bye ! --------------", flush=True)
    sys.exit(0)