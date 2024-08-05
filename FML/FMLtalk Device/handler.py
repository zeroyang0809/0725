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
import DataCollector
import DetectionModule
from flask import Flask, request, abort
import DAN, csmapi, random, time, threading, math, csv
import socket
import threading

FAILS_LIMIT = 300

Error_limit = 300
errors = 0
end = False

class GPS:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def equal(self, other_gps):
        return self.x == other_gps.x and self.y == other_gps.y and self.z == other_gps.z

class Vehicle_state:
    def __init__(self, throttle = None, brake = None, gps = None, speed = None):
        self.throttle = throttle
        self.brake = brake
        self.gps = gps
        self.speed = speed

def randomAddr():
    global Reg_addr
    Reg_addr = random.getrandbits(64)

def loadDataFilenames():
    global DB_LOCATION
    global traindata_x, traindata_y, testdata_x, testdata_y

    filenames = []
    with open(DB_LOCATION + 'filenames.txt') as f:
        for line in f.readlines():
            filenames.append(line)

    traindata = filenames[0]
    testdata = filenames[2]

def setErrorLimit(n):
    global Error_limit
    if isinstance(n,int):
        Error_limit = n

def registerDevice():
    global ServerURL, Reg_addr

    DAN.profile['dm_name'] = 'ValidityCheck'
    DAN.profile['df_list'] = ['Throttle-O','Brake-O','V_Gps-O','V_Speed-O','Validity-I']

    try:
        DAN.device_registration_with_retry(ServerURL, Reg_addr)

    except:
        print('Failed')


def throttle_speed_error(throttle, old_speed, new_speed):
    if throttle == None or old_speed == None or new_speed == None:
        return False
    else:
        return throttle == 0 and new_speed > old_speed

def brake_speed_error(brake, old_speed, new_speed):
    if brake == None or old_speed == None or new_speed == None:
        return False
    else:
        return brake > 0 and new_speed > old_speed

def speed_gps_error(speed, old_gps, new_gps):
    if speed == None or old_gps == None or new_gps == None:
        return False
    if speed > 0 and old_gps.equal(new_gps):
        print('Gps not moved')
        return True
    elif speed == 0 and not old_gps.equal(new_gps):
        print('Gps moved')
        return True
    else:
        return False
    
def ruleBasedDetection(ts_check = True, bs_check = True, sg_check = True):
    fails = 0
    global errors, end
    vehicle = Vehicle_state()
    old_speed = None
    old_gps = None

    while fails < FAILS_LIMIT and not end:

        # pull data
        ODF_data_speed = DAN.pull('V_Speed-O')
        if ODF_data_speed == None:
            fails += 1
        else:
            vehicle.speed = int(ODF_data_speed[0])
            fails = 0

        if vehicle.speed == None:
            print('Unconnected')
            time.sleep(1)
            continue

        ODF_data_throttle = DAN.pull('Throttle-O')
        if ODF_data_throttle == None:
            fails += 1
        else:
            vehicle.throttle = float(ODF_data_throttle[0])
            fails = 0
        
        ODF_data_brake = DAN.pull('Brake-O')
        if ODF_data_brake == None:
            fails += 1
        else:
            vehicle.brake = float(ODF_data_brake[0])
            fails = 0

        ODF_data_gps = DAN.pull('V_Gps-O')
        if ODF_data_gps == None:
            fails += 1
        else:
            gps_list = str(ODF_data_gps[0]).split(';')
            vehicle.gps = GPS(gps_list[0], gps_list[1], gps_list[2])
            fails = 0

        err = False
        # check
        if ts_check and throttle_speed_error(vehicle.throttle, old_speed, vehicle.speed):
            print('TS_Error')
            err =True
            errors += 1
        
        if bs_check and brake_speed_error(vehicle.brake, old_speed, vehicle.speed):
            print('BS_Error')
            err = True
            errors += 1

        if sg_check and ODF_data_speed != None and speed_gps_error(vehicle.speed, old_gps, vehicle.gps):
            print('SG_Error')
            err = True
            errors += 1

        print('Speed: ', vehicle.speed)
        print('Errors: ', errors)

        if old_speed != None:
            acc = vehicle.speed - old_speed
        else:
            acc = vehicle.speed
        
        if old_gps != None:
            gps_change = math.sqrt((float(vehicle.gps.x) - float(old_gps.x))**2 + (float(vehicle.gps.y) - float(old_gps.y))**2 + (float(vehicle.gps.z) - float(old_gps.z))**2)
        else:
            gps_change = 0

        old_speed = vehicle.speed
        old_gps = vehicle.gps

        time.sleep(1)
    print('Validity Check End')

def kerasDetection(modelName):

    fails = 0
    global errors, end
    vehicle = Vehicle_state()
    old_speed = None
    old_gps = None

    alert_change = False


    while fails < FAILS_LIMIT and not end:

        DataRecTime = time.time()

        ODF_data_speed = DAN.pull('V_Speed-O')
        if ODF_data_speed == None:
            fails += 1
        else:
            vehicle.speed = int(ODF_data_speed[0])
            fails = 0

        if vehicle.speed == None:
            print('Unconnected')
            time.sleep(1)
            continue

        ODF_data_throttle = DAN.pull('Throttle-O')
        if ODF_data_throttle == None:
            fails += 1
        else:
            vehicle.throttle = float(ODF_data_throttle[0])
            fails = 0
        
        ODF_data_brake = DAN.pull('Brake-O')
        if ODF_data_brake == None:
            fails += 1
        else:
            vehicle.brake = float(ODF_data_brake[0])
            fails = 0

        ODF_data_gps = DAN.pull('V_Gps-O')
        if ODF_data_gps == None:
            fails += 1
        else:
            '''
            # 學長原本的ODF_data_gps為String
            # gps_list = str(ODF_data_gps[0]).split(';')
            # vehicle.gps = GPS(gps_list[0], gps_list[1], gps_list[2])
            # fails = 0
            '''
            vehicle.gps = GPS(ODF_data_gps[0], ODF_data_gps[1], ODF_data_gps[2])
            fails = 0

        if old_speed != None:
            acc = vehicle.speed - old_speed
        else:
            acc = vehicle.speed
        
        if old_gps != None:
            gps_change = math.sqrt((float(vehicle.gps.x) - float(old_gps.x))**2 + (float(vehicle.gps.y) - float(old_gps.y))**2 + (float(vehicle.gps.z) - float(old_gps.z))**2)
        else:
            gps_change = 0

        x_list = [vehicle.speed, acc, vehicle.throttle, vehicle.brake, gps_change]

        x_data = DataCollector.xListToPandas(x_list)

        result = DetectionModule.detectionWithKeras(x_data, None, modelName, False, False)

        ResultSendTime = time.time()

        print('Detection Result:', result)

        #for alert test
        if result < 0.8: error = False
        else: error = True
        pushValidity(error)

        if error != None and error != alert_change:
            alert_change = error
            with open("DataRec.csv", "a+", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([DataRecTime, vehicle.speed])
            with open("ResultSend.csv", "a+", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ResultSendTime, error])

        time.sleep(0.1)

    print('Validity Check End')

def pushValidity(error):
    if error:
        DAN.push('Validity-I', True)
        print ("Validity Alert!")
    else:
        DAN.push('Validity-I', False)

def getErrorCount():
    global errors

    return errors

def newModel(traindata, testdata, modelName):
    #randomAddr()
    for train in traindata:
        modelTraining(train, modelName)
    registerDevice()
    #ruleBasedDetection()
    #kerasDetection(modelName)
    detect_threads = []
    detect_threads.append(threading.Thread(target = kerasDetection, args = (modelName,)))
    detect_threads[0].start()

    for test in testdata:
        score = modelScoring(test, modelName)
    return score


def retrain(traindata, testdata, modelName):
    #randomAddr()
    for train in traindata:
        modelTraining(train, modelName)
    #registerDevice()
    #ruleBasedDetection()
    #kerasDetection(modelName)
    for test in testdata:
        score = modelScoring(test, modelName)
    return score


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


            Trainingstart = time.time()
            print('Training start: ', Trainingstart)

            score = newModel(test_data, test_label,modelName)
            print('Model: ' + modelName + ' training completed. Score: ' + str(score))
            
            Trainingend = time.time()
            print('Training done: ', Trainingend)


            '''
            with open("TrainingTime.csv", "a+", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([Trainingend - Trainingstart])
            '''

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