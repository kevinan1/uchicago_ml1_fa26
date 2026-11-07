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
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataloader import HW2Dataset
from plotting import plot_surface_2d



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
    

class LinearTorch(nn.Module):
    def __init__(self, sk_model):
        super().__init__()
        in_features = sk_model.coef_.shape[1]
        self.linear = nn.Linear(in_features, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(sk_model.coef_, dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor(sk_model.intercept_, dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)


class problem2NN(nn.Module):
    def __init__(self, n_stock, n_cust, emb_dim=16, hidden=64, n_extra=0):
        super().__init__()
        self.stock_emb = nn.Embedding(n_stock, emb_dim)
        self.cust_emb = nn.Embedding(n_cust, emb_dim)
        self.mlp = nn.Sequential(nn.Linear(2 * emb_dim + n_extra, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        s = self.stock_emb(x[:, 0].long())
        c = self.cust_emb(x[:, 1].long())
        return self.mlp(torch.cat([s, c, x[:, 2:]], dim=1))

def make_loader(X, y, batch_size=50, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

def gaussian_nll(X, y):
    mu, sigma = X[0], X[1]
    assert mu.shape == sigma.shape == y.shape
    return (torch.log(sigma) + (y - mu) ** 2 / (2 * (sigma ** 2))).mean()

def train_loop(dataloader, model, loss_fn, optimizer, verbose=True):
    size = len(dataloader.dataset)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        #backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose and batch % 1000 == 0:
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

def fit(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=30, bad_limit=3, verbose=False):
    best, best_state, bad = np.inf, None, 0
    for epoch in range(num_epochs):
        train_loop(train_loader, model, loss_fn, optimizer, verbose=verbose)
        val = test_loop(val_loader, model, loss_fn)
        if val < best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= bad_limit:
                break
    model.load_state_dict(best_state)
    return model

def run_embedding_model(A_train, A_val, A_test, y_train, y_val, y_test, n_stock, n_cust, n_extra=0):
    train_l = make_loader(A_train, y_train, batch_size=1024)
    val_l = make_loader(A_val, y_val, batch_size=1024, shuffle=False)
    test_l = make_loader(A_test, y_test, batch_size=1024, shuffle=False)

    torch.manual_seed(69)
    model = problem2NN(n_stock, n_cust, n_extra=n_extra)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit(model, train_l, val_l, nn.functional.mse_loss, optimizer)
    return test_loop(test_l, model, nn.functional.mse_loss)

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
    lin_model = LinearTorch(lin_reg)


    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)
    
    def problem1_linkfunction(twodim_input):
        mu = twodim_input[:, 0:1]
        sigma = torch.exp(twodim_input[:, 1:2])
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
    cross_ent_lin = test_loop(test_loader, lin_model, lambda mu, y: cross_entropy((mu, torch.full_like(mu, resid_std)), y))
    acc  = test_loop(test_loader, nn_model, accuracy)
    acc_lin  = test_loop(test_loader, lin_model, lambda mu, y: accuracy((mu, torch.full_like(mu, resid_std)), y))
    
    print(f"MLP cross entropy loss : {cross_ent}")
    print(f"linear model cross entropy loss : {cross_ent_lin}")
    print(f"MLP accuracy : {acc}")
    print(f"linear model accuracy :{acc_lin}") 

def problem2(df):
    X = df[['StockCode', 'CustomerID']].astype(str)
    y = np.log1p(df['Quantity']).to_numpy().reshape(-1, 1)


    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15 / 0.85, random_state=69)
    
    enc = TargetEncoder(target_type='continuous', random_state=69)
    new_X_train = enc.fit_transform(X_train, y_train.ravel())
    new_X_val = enc.transform(X_val)
    new_X_test = enc.transform(X_test)

    lin_reg = LinearRegression().fit(new_X_train, y_train)
    lin_model = LinearTorch(lin_reg)

    test_data_loader = make_loader(new_X_test, y_test)
    loss = test_loop(test_data_loader, lin_model, nn.functional.mse_loss)
    print(f"average mean squared loss of log transformed data with linear model and target encoding {loss}")


    def to_codes(X):
        cols = []
        for col in ['StockCode', 'CustomerID']:
            lookup = {c: i + 1 for i, c in enumerate(X_train[col].unique())}
            cols.append(X[col].map(lookup).fillna(0).to_numpy())
        return np.stack(cols, axis=1)

    C_train, C_val, C_test = to_codes(X_train), to_codes(X_val), to_codes(X_test)
    n_stock, n_cust = int(C_train[:, 0].max()) + 1, int(C_train[:, 1].max()) + 1

    emb_loss = run_embedding_model(C_train, C_val, C_test, y_train, y_val, y_test, n_stock, n_cust)
    print(f"average mean squared loss of log transformed data with embedding model {emb_loss}")

    # 2d: embeddings + target-encoded features
    both_loss = run_embedding_model(np.hstack([C_train, new_X_train]),
                                    np.hstack([C_val, new_X_val]),
                                    np.hstack([C_test, new_X_test]),
                                    y_train, y_val, y_test, n_stock, n_cust, n_extra=2)
    print(f"average mean squared loss of log transformed data with embedding + target encoding model {both_loss}")


def problem3():
    from models import sweep_rks_model  # imported here because models.py imports from problems.py

    ds = HW2Dataset.from_csv("train.csv", "val.csv", "test.csv")

    plt.figure()
    sc = plt.scatter(ds.train.X[:, 0], ds.train.X[:, 1], c=ds.train.y)
    plt.colorbar(sc, label='y')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('training data 3a')
    plt.show()

    sweep_D = [2, 4, 8, 16, 32, 64, 128, 256]
    rks, rks_val, best_D, rks_tr, rks_va = sweep_rks_model(sweep_D, ds.train.X, ds.train.y, ds.val.X, ds.val.y)
    print('3b) RKS best value for D', best_D, rks_val)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(sweep_D, rks_tr, 'o-')
    ax[0].set(title='RKS train MSE', xlabel='D', xscale='log')
    ax[1].plot(sweep_D, rks_va, 'o-')
    ax[1].set(title='RKS val MSE', xlabel='D', xscale='log')
    plt.show()




    tr = make_loader(ds.train.X, ds.train.y.reshape(-1, 1), batch_size=32)
    va = make_loader(ds.val.X, ds.val.y.reshape(-1, 1), batch_size=200, shuffle=False)
    def train_mlp(h):
        torch.manual_seed(69)
        m = NeuralNetwork(in_features=2, hidden_sizes=[h, 1])
        opt = torch.optim.SGD(m.parameters(), lr=1e-2)
        fit(m, tr, va, nn.functional.mse_loss, opt, num_epochs=100, bad_limit=20)
        return m, test_loop(tr, m, nn.functional.mse_loss), test_loop(va, m, nn.functional.mse_loss)
    sweep_h = [4, 8, 16, 32, 64, 128, 256]
    res = [train_mlp(h) for h in sweep_h]
    mlp_tr, mlp_va = [r[1] for r in res], [r[2] for r in res]
    best_h = sweep_h[int(np.argmin(mlp_va))]
    print('3c) MLP best value for h', best_h, min(mlp_va))
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(sweep_h, mlp_tr, 'o-')
    ax[0].set(title='MLP train MSE', xlabel='hidden size', xscale='log')
    ax[1].plot(sweep_h, mlp_va, 'o-')
    ax[1].set(title='MLP val MSE', xlabel='hidden size', xscale='log')
    plt.show()





    mlp, _, mlp_val = train_mlp(best_h)
    print('val MSE RKS', rks_val, ' MLP', mlp_val)
    mlp_pred = mlp(torch.tensor(ds.test.X, dtype=torch.float32)).detach().numpy().ravel()
    rks_pred = rks.predict(ds.test.X)
    np.savetxt('rks_test_predictions.csv', rks_pred, delimiter=',')
    np.savetxt('mlp_test_predictions.csv', mlp_pred, delimiter=',')
    plot_surface_2d(rks.predict, (-3, 3), (-3, 3), 'RKS surface', 'rks_surface.png')
    plot_surface_2d(mlp, (-3, 3), (-3, 3), 'MLP surface', 'mlp_surface.png')

