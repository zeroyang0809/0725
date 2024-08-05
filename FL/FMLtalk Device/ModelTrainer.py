import time, DAN, requests, random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from MAML.learner import Learner  
from sklearn.model_selection import train_test_split

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



ServerURL = 'http://140.114.77.93:9999'
Reg_addr = "server_" + str( random.randint(1,100 )) #if None, Reg_addr = MAC address
DAN.profile['dm_name']='server'   # you can change this but should also add the DM in server
DAN.profile['df_list']=['Meta','Model', 'Loss', 'Parameter']   # Check IoTtalk to see what IDF/ODF the DM has
DAN.profile['d_name']= "."+ str( random.randint(1,100 ) ) +"_"+ DAN.profile['dm_name'] # None
DAN.device_registration_with_retry(ServerURL, Reg_addr) 
print("dm_name is ", DAN.profile['dm_name'])

num_epochs = 5  # Number of FL rounds

def modelTrainingWithPytorch(modelName):

    if os.path.exists(f'./Model/{modelName}.pt'):
        print(f'[modelTrainingWithKeras] {modelName}.pt exist')
        global_model = torch.load(f'./Model/{modelName}.pt')
    else:
        print(f'[modelTrainingWithKeras] {modelName}.pt not exist...........')
        global_model = Learner(config).to(device)

    try:
        DAN.device_registration_with_retry(ServerURL, Reg_addr) 

        input('Server Ready?  (y/n)\n')
        

        for epoch in range(num_epochs):
            # pack global meta model weight
            global_weights = [param.data.tolist() for _, param in global_model.named_parameters()]
            optimizer = optim.SGD(global_model.parameters(), lr=0.01)

            # push to IoTtalk platform and wait 3s
            DAN.push ('Model', global_weights)
            time.sleep(3)

            # try to receive data from clients
            recv_param = None
            while recv_param == None:
                print('Server is waiting for parameters')
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

    except Exception as e:
        print(e)
        if str(e).find('mac_addr not found:') != -1:
            print('Reg_addr is not found. Try to re-register...')
            DAN.device_registration_with_retry(ServerURL, Reg_addr)
        else:
            print('Connection failed due to unknow reasons.')
            time.sleep(1)
        try: 
            print("DAN deregister")
            DAN.deregister()    # 試著解除註冊
        except Exception as e:
            print("===") 
        exit(0)
    except KeyboardInterrupt as info:
        print(info)
        try: 
            print("DAN deregister")
            DAN.deregister()    # 試著解除註冊
        except Exception as e:
            print(e) 
        exit(0)

    torch.save(global_model, f'./Model/{modelName}.pt')
    print('Model saved.')


def modelScoringWithPytorch( test_data, test_label,modelName):

    if os.path.exists(f'./Model/{modelName}.pt'):
        testModel = torch.load(f'./Model/{modelName}.pt').to(torch.float32)
        print(f"[modelScoringWithKeras] find model {modelName}.pt")

    
    test = test_data.reshape(test_data.shape[0],7, 1, 1)
    criterion = nn.BCEWithLogitsLoss()
    
    testModel.eval()
    with torch.no_grad():
        test = test_data.reshape(test_data.shape[0],7, 1, 1)
        pred = testModel(test)

        print(pred[:5])

        pred = pred.view(-1)

        print(pred[:5])

        test_label = test_label.float()

        loss = criterion(pred, test_label)
        print(f"Loss: {loss.item():.4f}")

    return loss.item()