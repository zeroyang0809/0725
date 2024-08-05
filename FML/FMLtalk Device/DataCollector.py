import csv
import pandas as pd

filename = ''
dataList_x = []
dataList_y = []

dataTitle_x = []
dataTitle_y = ''

def setFilename(FN):
    global filename
    filename = FN

def setDataTitle(t):

    global dataTitle_x, dataTitle_y

    dataTitle_x = t[0:-1]
    dataTitle_y = t[-1]

def dataCollect(l):
    global dataList_x, dataList_y
    dataList_x.append(l[0:-1])

    if l[-1] == True:
        dataList_y.append('1')
    else:
        dataList_y.append('0')

def xListToPandas(xl):

    global dataTitle_x

    xpd = pd.DataFrame([xl], columns = dataTitle_x)

    return xpd

def printData():
    global dataList_x, dataList_y
    
    for i in range(0,len(dataList_x)):

        print(dataList_x[i] + ' ' + dataList_y[i])

def writeFile():
    global filename, dataList_x, dataList_y
    global dataTitle_x, dataTitle_y

    #dataList_x.insert(0, dataTitle_x)
    #dataList_y.insert(0, dataTitle_y)

    with open(filename + '_x.csv', 'w', newline='') as csvfile_x:
        writer_x = csv.writer(csvfile_x)

        writer_x.writerow(dataTitle_x)

        for row in dataList_x:
            writer_x.writerow(row)

    with open(filename + '_y.csv', 'w', newline='') as csvfile_y:
        writer_y = csv.writer(csvfile_y)

        writer_y.writerow([dataTitle_y])

        for row in dataList_y:
            writer_y.writerow(row)