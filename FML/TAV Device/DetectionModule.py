import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cnn

def importModel(modelName):

    global model

    model = torch.load(f'../Server/Model/model.pt')

def detectionWithKeras(data_x, output = False, printOnScreen = False):

    model = torch.load(f'../Server/Model/model.pt')

    data_x = torch.tensor(np.asarray(data_x), dtype=torch.float64).view(-1, *cnn.input_shape)
    # print(data_x)

    # if output:
    #     model.eval()
    #     with torch.no_grad():
    #         pred = model(data_x)

    #     output = pd.DataFrame({'result' : result.T[0]})
    #     output.to_csv(modelName + '_result.csv', index = False)

    #     print('Predict result saved.')

    #elif not output and data_y == None:
    model.eval()
    with torch.no_grad():
        pred = model(data_x)
        # print('result: ', pred)

    if printOnScreen:
        print('result: ', pred)
    return pred.item()
