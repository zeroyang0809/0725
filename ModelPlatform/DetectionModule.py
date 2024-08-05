import pandas as pd
import numpy as np
import csv

model = None

def importData(fileName):

    data_x = pd.read_csv(fileName + '_x.csv')
    data_y = pd.read_csv(fileName + '_y.csv')

    return data_x,data_y

def importModel(modelName):

    from tensorflow import keras
    from keras import models
    from keras.models import Sequential
    from keras.layers import Dense

    global model

    model = models.load_model(modelName + '.h5')

def detectionWithKeras(data_x, data_y, modelName, output = False, printOnScreen = False):

    from tensorflow import keras
    from keras import models
    from keras.models import Sequential
    from keras.layers import Dense

    global model

    data_x = np.asarray(data_x).astype(np.float32)
    
    
    if output:
        result = model.predict(data_x, batch_size=2)

        output = pd.DataFrame({'result' : result.T[0]})
        output.to_csv(modelName + '_result.csv', index = False)

        print('Predict result saved.')

    #elif not output and data_y == None:
    result = model.predict(data_x, batch_size=2)

    if printOnScreen:
        print('result: ', result.T[0])
    return result.T[0]

    '''
    elif not output and data_y != None:
        data_y = np.asarray(data_y).astype(np.float32)

        score = model.evaluate(data_x, data_y)

        print('Loss:', score[0])
        print('Accuracy:', score[1])
    '''



if __name__ == '__main__':

    import sys

    data_x, data_y = importData(sys.argv[1])

    print(data_x)
    print(data_y)

    detectionWithKeras(data_x, data_y, sys.argv[1] + '_model', True)

    print("Done")