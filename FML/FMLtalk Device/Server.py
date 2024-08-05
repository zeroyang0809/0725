# DAI2.py #coding=utf-8 -- new version of Dummy Device DAI.py, modified by tsaiwn@cs.nctu.edu.tw
import time, DAN, requests, random 
import threading, sys # for using a Thread to read keyboard INPUT

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from MAML.learner import Learner  
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

#set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

   # convert the dataloader
testing_data = TensorDataset(test_data, test_label)
test_loader = DataLoader(testing_data, batch_size = batch_size, shuffle = False)



def main():
    try:
    
        input('Server Ready?  (y/n)\n')

        round = 0

        for epoch in range(num_epochs):
            # pack global meta model weight
            global_weights = [param.data.tolist() for _, param in global_model.named_parameters()]
            optimizer = optim.SGD(global_model.parameters(), lr= 1e-3)

            # push to IoTtalk platform and wait 3s
            DAN.push ('Model', global_weights)
            time.sleep(3)

            # try to receive data from clients
            recv_param = None
            while recv_param == None:
                print('Server is waiting for Training Loss')
                time.sleep(0.5)
                recv_param = DAN.pull('Loss')
            print('Server got Training Loss!!!')

            if round == 0:  # if it is first round, print the number of client
                print(f"Federated Meta Training is total {len(recv_param[0])} clients")
                round += 1

            print(recv_param) #[[0.46, 0.41, 0.49]]
            tensor_recv_param = torch.tensor([item for sublist in recv_param for item in sublist], dtype=torch.float32, requires_grad=True)       
            training_loss = torch.mean(tensor_recv_param)

            print(training_loss)

            optimizer.zero_grad() # clear  gradient
            training_loss.backward()

            print(f'epoch {epoch+1} complete!\n')

        print(f'Server Federated Meta Training Success!! Total {num_epochs} epochs!')

        global_meta_weights = [param.data.tolist() for _, param in global_model.named_parameters()] 
        

        for i in range(10):
            print(10-i)
            time.sleep(0.5)


        DAN.push ('Meta', global_meta_weights)        

        torch.save(global_model, './Model/model.pt')
        model = torch.load('./Model/model.pt')
        model.eval()

        '''
        with torch.no_grad():
            print(311)
            test = test_data.reshape(test_data.shape[0],7, 1, 1)

            print(test.shape[0])


            outputs = global_model(test)

            for idx, x in enumerate(outputs):
                outputs[idx] = torch.tensor([1]) if x > 0.5 else torch.tensor([0])

            outputs = outputs.view(-1)

            correct = torch.eq(outputs, test_label).sum().item()  # convert to numpy

            print(correct)
            print(test.shape[0])
            accs = correct / test.shape[0]
            print('Accuracy: {}'.format(accs))

        '''
        
        time.sleep(300)


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
        time.sleep(3)
        try: 
            DAN.deregister()    # 試著解除註冊
        except Exception as e:
            print("===")
        print("Bye ! --------------", flush=True)
        sys.exit(0)

if __name__ == '__main__':
   
   ServerURL = 'http://140.114.77.93:9999'
   Reg_addr = "server_" + str( random.randint(1,1000)) #if None, Reg_addr = MAC address
   DAN.profile['dm_name']='server'   # you can change this but should also add the DM in server
   DAN.profile['df_list']=['Model','Meta', 'Parameter','Loss', 'model']   # Check IoTtalk to see what IDF/ODF the DM has
   DAN.profile['d_name']= "."+ str( random.randint(1,1000 ) ) +"_"+ DAN.profile['dm_name'] # None
   DAN.device_registration_with_retry(ServerURL, Reg_addr) 
   print("dm_name is ", DAN.profile['dm_name'])

   num_epochs = 10  # Number of FL rounds
   
   main()

'''   
        # Save model -> load model -> evaluate
        torch.save(global_model, './Model/model.pt')
        model = torch.load('./Model/model.pt')
        model.eval()
        with torch.no_grad():
            pred = model(test_data)
            print(f"[Before] predict result: {pred}")

            for idx, x in enumerate(pred):
                pred[idx] = torch.tensor([1]) if x > 0.5 else torch.tensor([0])

            print(f"[After] predict result: {pred}")

            correct = torch.eq(pred, labels).sum().item()  # convert to numpy
            accs = correct / test_data.shape[0]
            print(f'accs: {accs}')

            loss = criterion(pred, labels)
            print(f"Loss: {loss.item():.4f}")
    '''   

