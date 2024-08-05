import socket
import json
import time
import csv
import threading
import os
from urllib import request

from requests.sessions import Session
import ModelDatabase

def connectDatabase():

    ModelDatabase.connect('modelDB')
    ModelDatabase.create()

def uploadScore(mname, muploader, mdevice, mscore):
    
    print('Model: ' + mname + ' Score:' + str(mscore))

    mid = ModelDatabase.check_repeat(mname, muploader, mdevice)
    if  mid == 0:
        ModelDatabase.upload(mname, muploader, mdevice, mscore)
    else:
        ModelDatabase.scoreUpdate(mid, mname, muploader, mdevice, mscore)


def readDeviceList(): # return device information

    DEVICE_LIST = 'DeviceList.csv'

    deviceDict = {}

    with open(DEVICE_LIST) as deviceListFile:

        deviceList = list(csv.reader(deviceListFile))

        print("Reading Device List")

        for d in deviceList:
            
            deviceDict[d[0]] = [d[1], d[2]]

    return deviceDict

def devicesRequest(deviceDict, devicesReq, mname, muploader, op = 'new'):

    devicesNames = devicesReq.split(';')

    threads = []
    count_d = 0

    if op == 'new':
        for d in devicesNames:
            print('Register done: ', time.time())
            threads.append(threading.Thread(target = trainNewModel, args = (deviceDict[d][0], deviceDict[d][1], mname, muploader, d)))
            threads[count_d].start()
            count_d += 1

    elif op == 'retrain':
        for d in devicesNames:
            threads.append(threading.Thread(target = retrainModel, args = (deviceDict[d][0], deviceDict[d][1], mname, muploader, d)))
            threads[count_d].start()
            count_d += 1


    for ts in threads:
        ts.join()

def trainNewModel(host, port, mname, muploader, dname):

    print('Connect '+ host + ' port:', port)

    d_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    d_s.connect((host, int(port)))

    d_s.send(bytes('new', encoding = "utf-8"))


    data = d_s.recv(1024).decode()

    if data != None:
        print("Device send : %s " % (data))

        d_s.send(bytes(mname, encoding = "utf-8"))

        Storagestart = time.time()

        score = float(d_s.recv(1024).decode())
        
        print('Score receive: ', time.time())

        uploadScore(mname, muploader, dname, score)

        Storageend = time.time()

        #record the storage model time
        '''
        with open("StorageTime.csv", "a+", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([Storageend - Storagestart])
        '''


        d_s.close()
    else:
        print("Request Failed")

def registerModel(mname, muploader, devicesReq):  

    deviceDict = readDeviceList()
    devicesRequest(deviceDict, devicesReq, mname, muploader)


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'ModelFileDB')
#IOTDEVICE_FOLDER = os.path.join(os.getcwd(), 'ModelPlatform')
ALLOWED_EXTENSIONS = set(['pt'])

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['IOTDEVICE_FOLDER'] = IOTDEVICE_FOLDER

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        
        submissionstart = time.time()
        print('submission start: ', submissionstart)


        mname = request.values.get('mname')
        muploader = request.values.get('muploader')
        mdevice = request.values.get('mdevice')
        
        f = request.files['file']       
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        #f.save(os.path.join(app.config['IOTDEVICE_FOLDER'], secure_filename(f.filename)))

        print(muploader, ' upload a model \"', mname, '\" to the device ', mdevice)
        
        submissionend = time.time()
        print('submission done: ', submissionend)

        #record the task submission time
        '''
        with open("submissionTime.csv", "a+", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([submissionend - submissionstart])
        '''


        t = threading.Thread(target = registerModel, args = (mname, muploader, mdevice))
        t.start()

        t.join()
        Te = time.time()
        print('End: ', time.time())

        return render_template('success.html')

@app.route('/leaderboard_choose')
def leaderboard_choose():
    return render_template('leaderboard_choose.html')

@app.route('/leaderboard', methods = ['GET', 'POST'])
def leaderboard():
    if request.method == 'POST':
        Tes = time.time()

        print('Load leaderboard: ', time.time())

        mdevice = request.values.get('mdevice')

        if mdevice is not None:
            leaderboard_list = ModelDatabase.getLeaderboard(mdevice)

        Ted = time.time()

        print('Load complete: ', time.time())

        return render_template('leaderboard.html',deviceName = mdevice, leaderboard = leaderboard_list)


if __name__ == '__main__':
    
    connectDatabase()

    app.debug=True
    app.run()