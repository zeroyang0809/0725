from flask import Flask, request, abort
import time
import ModelTrainer as mt
import socket, time, torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from MAML.learner import Learner  
from sklearn.model_selection import train_test_split


ServerURL = 'http://140.114.77.93:9999' #with no secure connection
Reg_addr = 'VC_Test'

#ip and port3478029340
bind_ip = "127.0.0.1"
bind_port = 4041

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


end = False

def modelTraining(modelName):
   
    mt.modelTrainingWithPytorch(modelName)

def modelScoring(test_data, test_label,modelName):

    score = mt.modelScoringWithPytorch( test_data, test_label,modelName)

    return score   

def newModel(test_data, test_label, modelName):


    modelTraining(modelName)

    score = modelScoring(test_data, test_label,modelName)

    return score


if __name__ == '__main__':

    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind((bind_ip,bind_port))
    server.listen(5)
    print("Listening on %s:%d " % (bind_ip,bind_port))

    threads = []
    thread_count = 0

    while not end:
        client,addr = server.accept()

        #data receive
        data = client.recv(1024).decode()

        print("Socket Receive")
        
        print(data)

        print('Socket receive: ', time.time())

        if data == 'new':

            print("Anomaly detection model training start")
            client.send(b"anomaly detection model training start")

            modelName = client.recv(1024).decode()

            print("ModelName: " + modelName)
            Tst = time.time()

            print('Training start: ', time.time())

            score = newModel(test_data, test_label,modelName)

            print('Model: ' + modelName + ' training completed. Score: ' + str(score))
            Tsd = time.time()
            print('Training done: ', time.time())

            client.send(bytes(str(score), encoding = "utf-8"))

            client.close()

        # no use
        elif data == 'retrain':

            print("V3alidity Check retrain start")
            client.send(b"Validity Check retrain start")

            modelName = client.recv(1024).decode()

            print("ModelName: " + modelName)

            score = newModel(modelName)

            client.send(score)

            client.close()