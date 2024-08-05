from flask import Flask, request, abort
import DAN, csmapi, random, time, threading
import socket
import threading
import DataCollector
import DetectionModule
import pandas

ServerURL = 'http://140.113.199.246:9999/' #with no secure connection
Reg_addr = None

#ip and port
bind_ip = "127.0.0.1"
bind_port = 4040

FAILS_LIMIT = 180

Speed_limit = 30
Alert_happened = False

def randomAddr():
    global Reg_addr
    Reg_addr = random.getrandbits(64)

def setSpeedLimit(n):
    global Speed_limit
    if isinstance(n,int):
        Speed_limit = n

def registerDevice():
    DAN.profile['dm_name'] = 'SpeedMonitor'
    DAN.profile['df_list'] = ['V_Speed-O','Speed_Alert-I']

    try:
        DAN.device_registration_with_retry(ServerURL, Reg_addr)

    except:
        print('Failed')

def ruleBasedDetection():
    fails = 0

    DataCollector.setFilename('SpeedMonitor')
    DataCollector.setDataTitle(['Speed', 'error'])

    while fails < FAILS_LIMIT:
        err = False

        ODF_data = DAN.pull('V_Speed-O')
        if ODF_data != None:
            print ("speed: ",str(ODF_data[0]))
            if int(ODF_data[0]) > Speed_limit:
                alert(True)
                err = True
                #break
            fails = 0
        else:
            print ("Pull failed.")
            fails += 1

        DataCollector.dataCollect([ODF_data[0], err])
        time.sleep(1)

    print('Device shutdown')
    return Alert_happened

def kerasDetection():
    fails = 0

    DataCollector.setFilename('SpeedMonitor')
    DataCollector.setDataTitle(['Speed', 'error'])

    while fails < FAILS_LIMIT:
        err = False

        ODF_data = DAN.pull('V_Speed-O')
        if ODF_data != None:
            print ("speed: ",str(ODF_data[0]))

            x_data = [ODF_data[0]]
            result = DetectionModule.detectionWithKeras(x_data, None, 'ValidityCheck', False, False)
            if result:
                alert(True)
                err = True
                #break
            fails = 0
        else:
            print ("Pull failed.")
            fails += 1

        DataCollector.dataCollect([ODF_data[0], err])
        time.sleep(1)

    print('Device shutdown')
    return Alert_happened


def alert(a):
    DAN.push('Speed_Alert-I', bool(a))
    global Alert_happened
    if a == True:
        Alert_happened = True
        print ("Speed Alert!")

def getResult():
    return Alert_happened

def runAll():
    #randomAddr()
    registerDevice()
    ruleBasedDetection()
    #kerasDetection()
    listen()

if __name__ == '__main__':

    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind((bind_ip,bind_port))
    server.listen(5)
    print("Listening on %s:%d " % (bind_ip,bind_port))

    threads = []
    thread_count = 0

    while True:

        #data receive
        client,addr = server.accept()
        data = client.recv(1024).decode()
        print("Socket Receive")

        if data == 'new':

            threads.append(threading.Thread(target = runAll))
            threads[thread_count].start()
            thread_count += 1

            print("Validity Check Start")
            client.send(b"Validity Check Start")

        elif data == 'end':
            break

