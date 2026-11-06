import copy
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler





class NeuralNetwork(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_function=nn.Identity()):
        super().__init__()
        layers = nn.ModuleList()
        prev = in_features

        for i, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, layer_size))
            if i < len(hidden_sizes) -1:
                layers.append(nn.ReLU())
            prev = layer_size
        self.layers = nn.Sequential(*layers)
        self.out_function = out_function
        
    def forward(self, x):
        output = self.out_function(self.layers(x))
        return output
    

class LinearBaseline(nn.Module):
    def __init__(self, sk_model, sigma):
        super().__init__()
        in_features = sk_model.coef_.shape[1]
        self.linear = nn.Linear(in_features, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(sk_model.coef_, dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor(sk_model.intercept_, dtype=torch.float32))
        self.sigma = float(sigma)

    def forward(self, x):
        mu = self.linear(x)
        return mu, torch.full_like(mu, self.sigma)


def make_loader(X, y, batch_size=50, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

def gaussian_nll(X, y):
    mu, sigma = X[0], X[1]
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

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f'loss, {loss:>7f}   [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    total_loss = 0.0
    
    with torch.no_grad():
        for X, y in dataloader:
            total_loss += loss_fn(model(X), y).item() * X.size(0)
    avg_loss = total_loss / size
    return avg_loss

def problem1d_predict(pred, y, threshold=45.0):
    mu, sigma = pred[0], pred[1]
    p = Normal(mu, sigma).cdf(torch.tensor(threshold))
    label = (y < threshold).float()
    return p, label

def cross_entropy(pred, y):
    p, label = problem1d_predict(pred, y)
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(label * p.log() + (1 - label) * (1 - p).log()).mean()
    
def accuracy(pred, y):
    p, label = problem1d_predict(pred, y)
    return ((p > 0.5).float() == label).float().mean()

def problem1(df):

    #the features for prediction and the features to be predicted
    X_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'min_of_day', 'day_of_week', 'day_of_year']
    Y_features = ['trip_duration']
    
    X_np, y_np = df[X_features].to_numpy(), df[Y_features].to_numpy()
    X, y = torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_np, y_np, test_size=0.15, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15 / 0.85, random_state=69)
    
    
    #baseline
    lin_reg = LinearRegression().fit(X_train, y_train)
    lin_predicts = lin_reg.predict(X_test)
    resid_std = np.std(y_train - lin_reg.predict(X_train))
    lin_model = LinearBaseline(lin_reg, resid_std)


    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)
    
    def problem1_linkfunction(twodim_input):
        mu = twodim_input[:, 0:1]
        sigma = nn.functional.softplus(twodim_input[:, 1:2])
        return mu, sigma
    
    nn_model = NeuralNetwork(in_features=7, hidden_sizes=[16, 16, 16, 2], out_function=problem1_linkfunction) 
    num_epochs = 100
    best, best_state = np.inf, None
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01)
    for i in range(num_epochs):
        train_loop(dataloader=train_loader, model=nn_model, loss_fn=gaussian_nll, optimizer=optimizer)
        val = test_loop(dataloader=val_loader, model=nn_model, loss_fn=gaussian_nll)
        print(val)
        if val < best:
            best = val
            best_state = copy.deepcopy(nn_model.state_dict())
    
    nn_model.load_state_dict(best_state)
    
    cross_ent = test_loop(test_loader, nn_model, cross_entropy)
    cross_ent_lin = test_loop(test_loader, lin_model, cross_entropy)
    acc  = test_loop(test_loader, nn_model, accuracy)
    acc_lin  = test_loop(test_loader, lin_model, accuracy)
    
    print(f"MLP cross entropy loss : {cross_ent}")
    print(f"linear model cross entropy loss : {cross_ent_lin}")
    print(f"MLP accuracy : {acc}")
    print(f"linear model accuracy :{acc_lin}") 

def problem2():
    df = pd.read_csv("retail.csv")
    
    
