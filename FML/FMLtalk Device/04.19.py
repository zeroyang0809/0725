import  torch
from torch import optim
from torch import nn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from MAML.learner import Learner
import  argparse
from MAML import meta
import process


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score

batch_size = 16

# prepare the finetune-dataset
Vehicle_data =pd.read_csv("anomalydetection_train.csv")
data = Vehicle_data.loc[:, Vehicle_data.columns != "Class"].values
label = Vehicle_data.Class.values

#split the train set and test set
train_data, test_data, train_label, test_label = train_test_split( 
                                                  data,
                                                  label,
                                                  test_size = 0.2,
                                                  random_state = 42)

# convert to tensor
train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label).type(torch.FloatTensor) 

# convert to tensor
test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.FloatTensor) 

# convert the dataloader
training_data = TensorDataset(train_data, train_label)
train_loader = DataLoader(training_data, batch_size = batch_size, shuffle = False)

loss_list = []
iteration_list = []
accuracy_list = []


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    # setup trainModel architecture
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

    #setup testModel architecture
    config2 = [
        ('conv2d', [16, 7, 1, 1, 1, 0]),
        ('relu', [True]),
        ('max_pool2d', [1, 1, 0]),
        ('conv2d', [8, 16, 1, 1, 1, 0]),
        ('relu', [True]),
        ('max_pool2d', [1, 1, 0]),
        ('flatten', []),
        ('linear', [128, 8]),
        ('relu', [True]),
        ('linear', [1, 128]),
        ('sigmoid', [])
    ]
  
    #set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    trainModel = Meta(args, config).to(device)
    
    #print the tensor parameter
    tmp = filter(lambda x: x.requires_grad,  trainModel.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(trainModel)
    print('Total trainable tensors:', num)

    # generate dataset for Model Federated meta training
    train_dataset = process.dataset(mode = "train", k_shot=args.k_spt, k_query=args.k_qry, num_batch=10)

    # Start Federated Meta Training tasks
    print("=========== Federated Meta Training Start ==========")
    for epoch in range(args.epoch):
        db = DataLoader(train_dataset, args.task_num, shuffle=True, pin_memory=True)
        
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs =  trainModel(x_spt, y_spt, x_qry, y_qry)

            if step % 10 == 0:
                print('step:', step, '\ttraining acc:', accs)

    print("=========== Federated Meta training finish ==========")  


    #Save model 
    torch.save(trainModel.getState(), './Model.pt') #well-inilization parameter

    #define the testModel
    testModel = Learner(config2).to(device)

    # load the pretrained model into the  testModel
    testModel.load_state_dict(trainModel.getState()) 
    optimizer = optim.Adam(testModel.parameters(), lr=args.meta_lr)
    criterion = nn.BCEWithLogitsLoss()

    testModel.train()

    num_epochs = 50

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            trainModel.train()
            train = images.reshape(images.shape[0],7,1,1)

            optimizer.zero_grad()

            outputs = testModel(train) 
            labels = labels.view(-1, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    

   
        testModel.eval()
        test = test_data.reshape(test_data.shape[0],7, 1, 1)
        outputs = testModel(test)

        
        for idx, x in enumerate(outputs):
            outputs[idx] = torch.tensor([1]) if x > 0.5 else torch.tensor([0])
        outputs = outputs.view(-1)



        correct = torch.eq(outputs, test_label).sum().item()  # convert to numpy
        accs = correct / test.shape[0]

        # store loss and iteration
        loss_list.append(loss.item())
        iteration_list.append(epoch)
        accuracy_list.append(accs)
            
        print('Iteration: {}  Loss: {}  Accuracy: {}'.format(epoch, loss.item(), accs))

        

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help=' train epoch number', default=4)
    argparser.add_argument('--epoch_te', type=int, help='test epoch number', default=4)

    #argparser.add_argument('--n_way', type=int, help='n way', default=1) # 
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=2)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)


    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=2)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=2)

    args = argparser.parse_args()

    main()


    '''
    for epoch in range(args.epoch):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(train_dataset, args.task_num, shuffle=True, pin_memory=True)
        
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs =  trainModel(x_spt, y_spt, x_qry, y_qry)

            if step % 10 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 10 == 0:  # evaluation
                db_test = DataLoader(test_dataset,shuffle=True, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs =  trainModel.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)



                # Save model -> load model -> evaluate
torch.save(local_model, './Model/model.pt')
model = torch.load('./Model/model.pt')
model.eval()
with torch.no_grad():
    pred = model(test_data)
    print(f"[Before] predict result: {pred}")

    for idx, x in enumerate(pred):
        pred[idx] = torch.tensor([1]) if x > 0.5 else torch.tensor([0])
    pred = pred.view(-1)

    print(f"[After] predict result: {pred}")

    correct = torch.eq(pred, labels).sum().item()  # convert to numpy
    accs = correct / test_data.shape[0]
    print(f'accs: {accs}')

    loss = criterion(pred, labels)
    print(f"Loss: {loss.item():.4f}")   
    '''