import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torcchvision import datasets, transforms

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler





class NeuralNetwork(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_function=nn.Linear):
        super().__init__()
        layers = nn.ModuleList()
        prev = in_features

        for i, layer_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(prev, layer_size))
            if i < len(hidden_sizes) -1:
                self.layers.append(nn.ReLU())
            prev = layer_size
        self.layers = nn.Sequential(*layers)
        self.out_function = out_function
        
    def forward(self, x):
        output = self.out_function(self.layers(x))
        return output
    

def make_dataloader(X, y, batch_size=50, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

def gaussian_nll(mu, sigma, y):
    assert mu.shape == sigma.shape == y.shape
    return (torch.log(sigma) + (y - mu) ** 2 / (2 * (sigma ** 2))).mean()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        #backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f'loss, {loss:>7f}   [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    
    


def problem_1c(df):

    #the features for prediction and the features to be predicted
    X_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'min_of_day', 'day_of_week', 'day_of_year']
    Y_features = ['trip_duration']
    
    X_np, y_np = df[X_features].to_numpy(), df[Y_features].to_numpy()
    X, y = torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=69)

    #baseline
    lin_reg = LinearRegression().fit(X_train, y_train)
    lin_predicts = reg.predict(X_test)

    
