import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

input_shape = (7, 1, 1)

class dataset(Dataset):
    def __init__(self, mode = "train", num_batch=30, n_way=1, k_shot=5, k_query=10):
        super().__init__()
        self.num_batch = num_batch  # total batch number
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query

        if mode == "train":
            self.input_data = torch.tensor(pd.read_csv('dataset/anomalydetection_train_x.csv').values, dtype=torch.float32).view(-1, *input_shape)
            self.labels = torch.tensor(pd.read_csv('dataset/anomalydetection_train_y.csv').values, dtype=torch.float32)
        else:
            self.input_data = torch.tensor(pd.read_csv('dataset/anomalydetection_test_x.csv').values, dtype=torch.float32).view(-1, *input_shape)
            self.labels = torch.tensor(pd.read_csv('dataset/anomalydetection_test_y.csv').values, dtype=torch.float32)

        self.create_batch()
        

    def create_batch(self):
        self.support_batch_x = []
        self.support_batch_y = []
        self.query_batch_x = []
        self.query_batch_y = []
        
        np.random.seed(1)
        for _ in range(self.num_batch):
            idx = [i for i in range(self.input_data.shape[0])]
            select_idx = np.random.choice(idx, self.k_shot + self.k_query, False)
            support_idx = select_idx[:self.k_shot]
            query_idx = select_idx[self.k_shot:]
            self.support_batch_x.append(self.input_data[support_idx])
            self.support_batch_y.append(self.labels[support_idx])
            self.query_batch_x.append(self.input_data[query_idx])
            self.query_batch_y.append(self.labels[query_idx])

    def __getitem__(self, idx):
        return self.support_batch_x[idx], self.support_batch_y[idx], \
                self.query_batch_x[idx], self.query_batch_y[idx]
    
    def __len__(self):
        return self.num_batch
    
    def __call__(self):
        print(f'1 support batch x shape: {self.support_batch_x[0].shape}')
        print(f'1 support batch y shape: {self.support_batch_y[0].shape}')
        print(self.support_batch_y[0])
    
if __name__ == '__main__':
    d = dataset(num_batch=10)
    d()
    d_l = DataLoader(d, batch_size=2, shuffle=True)
    for batch, (x_spt, y_spt, x_qry, y_qry) in enumerate(d_l):
        print(f"batch {batch}:")
        print(x_spt.shape)
        print(y_spt.shape)