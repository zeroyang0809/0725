# DAI2.py #coding=utf-8 -- new version of Dummy Device DAI.py, modified by tsaiwn@cs.nctu.edu.tw
import time, DAN, requests, random 
import threading, sys # for using a Thread to read keyboard INPUT

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import dataset
from MAML.meta import Meta
import argparse


#set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10

def main():
   
    try:
        maml = Meta(args, config).to(device)

        train_dataset = dataset.dataset(mode = "train", n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, num_batch=40)
        train_dataloader = DataLoader(train_dataset, batch_size=args.task_num, shuffle=True, pin_memory=True)

        input('Client3 Ready?  (y/n)\n')

        for epoch in range(num_epochs):
            end = 0
            waiting_limit = 60
            recv_param = None

            # try to receive data from server (global meta-model)
            while recv_param == None:
                print('Client3 is waiting for parameter')
                time.sleep(0.5)
                recv_param = DAN.pull('model')
                end += 1
                if end > waiting_limit: 
                    break
            if end > waiting_limit: 
                break

            print('Client3 got model!!!')

            # take out & turn receive data (type: list) into tensor
            recv_param = [torch.tensor(p).to(device) for p in recv_param[0]]

            print(f"[Initial] param check: {maml.meta_parameters()[1].data}")

            # set the local meta model weight to receive data
            for idx, param in enumerate(maml.meta_parameters()):
                param.data = recv_param[idx]

            print(f"[Receive] param check: {maml.meta_parameters()[1].data}")

            for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_dataloader):
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                accs, training_loss = maml(x_spt, y_spt, x_qry, y_qry)
            
                if step % 10 == 0:
                    print('step:', step, '\ttraining acc:', accs)

            print(training_loss)

            training_loss = training_loss.tolist()

            DAN.push('loss', training_loss)

            print('client send training loss to Server')

        print(f'Training Success!! Total {num_epochs} epoch')

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
        time.sleep(1)
        try: 
            DAN.deregister()    # 試著解除註冊
        except Exception as e:
            print("===")
        print("Bye ! --------------", flush=True)
        sys.exit(0)


if __name__ == '__main__':

   ServerURL = 'http://140.114.77.93:9999'
   Reg_addr = "client3_" + str( random.randint(1,1000 )) #if None, Reg_addr = MAC address
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

   argparser = argparse.ArgumentParser()
   
   argparser.add_argument('--n_way', type=int, help='n way', default=1)
   argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
   argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
   argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
   argparser.add_argument('--imgc', type=int, help='imgc', default=3)
   argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=2)
   argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
   argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
   argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=2)
   argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=2)

   args = argparser.parse_args()


   main()