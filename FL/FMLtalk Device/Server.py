import time, DAN, requests, random 
import threading, sys 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from MAML.learner import Learner  
from sklearn.model_selection import train_test_split

#set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ServerURL = 'http://140.114.77.93:9999'
Reg_addr = "server_" + str( random.randint(1,1000)) #if None, Reg_addr = MAC address
DAN.profile['dm_name']='server'   # you can change this but should also add the DM in server
DAN.profile['df_list']=['Model','Meta', 'Parameter','Loss']   # Check IoTtalk to see what IDF/ODF the DM has
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

num_epochs = 10  # Number of FL rounds

try:
   global_model = Learner(config).to(device)
   criterion = nn.BCEWithLogitsLoss()
   optimizer = optim.SGD(global_model.parameters(), lr=0.01)
   batch_size =5 


   Vehicle_data =pd.read_csv("anomalydetection_test.csv")
   data = Vehicle_data.loc[:, Vehicle_data.columns != "Class"].values
   label = Vehicle_data.Class.values

   train_data, test_data, train_label, test_label = train_test_split( 
                                                    data,
                                                    label,
                                                    test_size = 0.99,
                                                    random_state = 42)
    
   test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
   test_label = torch.from_numpy(test_label).type(torch.LongTensor) 


   DAN.device_registration_with_retry(ServerURL, Reg_addr) 
   input('FMLtalk Device Ready?  (y/n)\n')
   
   for epoch in range(num_epochs):
      # pack global model weight
      global_weights = [param.data.tolist() for _, param in global_model.named_parameters()]

      # push to IoTtalk platform and wait 3s
      DAN.push ('Model', global_weights)
      time.sleep(3)

      round = 0

      # try to receive data from clients
      recv_param = None
      while recv_param == None:
         print('Server is waiting for parameter')
         time.sleep(0.5)
         recv_param = DAN.pull('Parameter')
      print('Server got parameter!!!')
      
      # unpack receive data ( [[[client1 data],[client2 data],...]] )
      # and average weight from different clients
      average_weights = []
      for client in recv_param[0]:
         if len(average_weights) == 0:
            for p in client:
               average_weights.append(np.array(p)/len(recv_param[0]))
         else:
            for id, p in enumerate(client):
               average_weights[id] += np.array(p)/len(recv_param[0])
      
      # set the local model weight to receive data
      for idx, (name, param) in enumerate(global_model.named_parameters()):
         param.data = torch.from_numpy(average_weights[idx]).type(torch.FloatTensor)

      print(f'epoch {epoch+1} complete!')

   print(f'Training Success!! Total {num_epochs} epochs!')

   # Save model -> load model -> evaluate
   torch.save(global_model, './Model/model.pt')
   
   with torch.no_grad():
            test = test_data.reshape(test_data.shape[0],7, 1, 1)
            outputs = global_model(test)

            for idx, x in enumerate(outputs):
                outputs[idx] = torch.tensor([1]) if x > 0.5 else torch.tensor([0])

            outputs = outputs.view(-1)

            correct = torch.eq(outputs, test_label).sum().item()  # convert to numpy
            print(correct)
            print(test.shape[0])
            accs = correct / test.shape[0]
            print('Accuracy: {}'.format(accs))


except Exception as e:
   print(e)
   if str(e).find('mac_addr not found:') != -1:
      print('Reg_addr is not found. Try to re-register...')
      DAN.device_registration_with_retry(ServerURL, Reg_addr)
   else:
      print('Connection failed due to unknow reasons.')
      time.sleep(1)    
finally:
   print('Server Complete!')
   time.sleep(0.25)
   try: 
      DAN.deregister()    # 試著解除註冊
   except Exception as e:
      print("===")
   print("Bye ! --------------", flush=True)
   sys.exit(0)