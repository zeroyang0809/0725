from flask import Flask, request, abort
import DAN, csmapi, random, time, threading
import socket
import threading
import csv

ServerURL = 'http://140.113.199.246:9999/' #with no secure connection
Reg_addr = 'RC_Test'

#ip and port
bind_ip = "127.0.0.1"
bind_port = 4030

FAILS_LIMIT = 30

Error_limit = 10
errors = 0
error = {}

end = False

def randomAddr():
    global Reg_addr
    Reg_addr = random.getrandbits(64)

def setErrorLimit(n):
    global Error_limit
    if isinstance(n,int):
        Error_limit = n

def registerDevice():
    DAN.profile['dm_name'] = 'ResultCollect'
    DAN.profile['df_list'] = ['Speed_Alert-O','Validity-O','ErrorList-I']

    try:
        DAN.device_registration_with_retry(ServerURL, Reg_addr)

    except:
        print('Failed')

def errorMonitor(dn, df):
    global error, errors, end
    fails = 0

    error[dn] = 0
    while not end:

        alert = DAN.pull(df)

        if alert != None and alert[0] == True:
            error[dn] += 1
            errors += 1
            print(dn,': ',error[dn])
            fails = 0

        elif alert == None:
            fails += 1
            if fails > FAILS_LIMIT:
                print("Device Disconnect")
                break

        time.sleep(1)

def listen(device = []):
    threads = []
    count_d = 0

    if 'SpeedMonitor' in device:
        print("Speed Monitor Collecting")

        threads.append(threading.Thread(target = errorMonitor, args = ('SpeedMonitor','Speed_Alert-O',)))
        threads[count_d].start()
        count_d += 1

    if 'ValidityCheck' in device:
        print("Validity Check Collecting")

        threads.append(threading.Thread(target = errorMonitor, args = ('ValidityCheck','Validity-O',)))
        threads[count_d].start()
        count_d += 1

    for ts in threads:
        ts.join()

    pushErrorList()

def pushErrorList():
    global error

    erl = ''
    for dn in device:
        erl += dn + ':' + str(error[dn]) + ';'
    
    print("Push ErrorList")
    DAN.push('ErrorList-I', str(erl))

def compareAllScore(modelList):

    results = {}
    numOfModel = len(modelList)
    scores = {}
    numOfRows = 0

    for mname in modelList:

        with open(mname + '.csv', newline='') as csvfile:

            r = csv.DictReader(csvfile)

            results[mname] = r
            numOfRows = len(r)

    
    ans = [0] * numOfRows

    for mname in modelList:
        for n in range(0,numOfRows):
            if results[mname][n] == True:
                ans[n] += 1

    for a in ans:
        a = a/numOfModel

    for mname in modelList:
        scores[mname] = 0
        getScore(ans, results[mname])

    return ans, scores

def getScore(ans, result):
    
    score = 0

    for n in range(0,len(result)):
        if result[n] == True:
            score += ans[n]
        else:
            score += 1 - ans[n]

    return score

def getErrorCount():
    return errors

def runAll(device = []):
    #randomAddr()
    registerDevice()
    listen(device)

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

            client.send(b"Ready For Device List")

            data2 = client.recv(1024).decode()
            print("Device List Receive")

            device = data2.split(';')

            threads.append(threading.Thread(target = runAll, args = (device,)))
            threads[thread_count].start()
            thread_count += 1

            print("Result Collect Start")
            client.send(b"Result Collect Start")

        elif data == 'end':
            print('Collect End')
            end = True
            break
