import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader
import dataset

input_shape = (7, 1, 1)
input_shape2 =(3, 1, 1)
torch.autograd.set_detect_anomaly(True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 1), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1), stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 1)

        self.flag = 0

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def load_weight(model):
    tmp = deepcopy(model.state_dict())
    weights = [value for _, value in tmp.items()]
    return weights

def maml_inner(model_, x_spt, y_spt, x_qry):
    model = CNN()
    model.load_state_dict(model_.state_dict())
    num_inner_loop = 3
    inner_lr = 0.01
    updated_state_dict = deepcopy(model.state_dict())
    criterion = nn.MSELoss()

    print(f"Before: maml inner model fc2 bisa {model.fc2.bias}")

    for n in range(num_inner_loop):
        if n > 0:
            model.load_state_dict(updated_state_dict)
        weights_for_autograd = load_weight(model)
        pred_y_spt = model(x_spt)
        loss = criterion(pred_y_spt, y_spt)
        grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        for w_idx, (name, _) in enumerate(model.named_parameters()):
            updated_state_dict[name] = weights_for_autograd[w_idx] - inner_lr*grad[w_idx]
        
    model.load_state_dict(updated_state_dict)
    pred_y_qry = model(x_qry)

    print(f"After: maml inner model fc2 bisa {model.fc2.bias}")

    return pred_y_qry


def main():
    num_batch = 10  # total batch number
    batch_size = 2  # one batch contain how many support/query sets
    num_epochs = 5  # Number of training epochs
    
    train_dataset = dataset.dataset(num_batch=num_batch)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = CNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        for idx, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_dataloader):
            loss = 0.
            optimizer.zero_grad()  # Clear gradients
            for x in range(x_spt.shape[0]):
                pred_y_qry = maml_inner(model, x_spt[x], y_spt[x], x_qry[x])
                # pred_y_qry = model(x_qry[x])
                loss += criterion(pred_y_qry, y_qry[x])
                model.flag += 1
            loss /= x_spt.shape[0]

            for name, p in model.named_parameters():
                if name == "fc2.bias":
                    print(f"[grad]   name: {name}, before: {p.data}")

            loss.backward()  # Backpropagation

            for name, p in model.named_parameters():
                if name == "fc2.bias":
                    print(f"[grad]   name: {name}, after: {p.data}")

            optimizer.step()  # Update weights
            print(f"\n\nEpoch [{epoch+1}/{num_epochs}], batch id: {idx}, Loss: {loss.item():.4f}\n\n")

    # Save model -> load model -> evaluate
    torch.save(model, './Model/model.pt')
    model = torch.load('./Model/model.pt')

    test_dataset = dataset.dataset(mode = "test", num_batch=10)
    test_dataloader = DataLoader(test_dataset , batch_size=1, shuffle=True)

    model.eval()
    # with torch.no_grad():
    correct = 0
    total = 0
    for idx, (x_spt, y_spt, x_qry, y_qry) in enumerate(test_dataloader):
        pred_y_qry = maml_inner(model, x_spt[0], y_spt[0], x_qry[0])
        loss = criterion(pred_y_qry, y_qry[0])
        # correct += torch.eq(pred_y_qry.argmax(dim=1), y_qry[0]).sum().item()
        # total += len(y_qry[0])
        # print(pred_y_qry)
        # print(y_qry[0])
        # print(f"correct: {correct}, total: {total}")
        print(f"test id: {idx}, Loss: {loss.item():.4f}")
    # print(f"correct rate: {correct}/{total}")

if __name__ == "__main__":
    main()