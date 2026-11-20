import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.neighbors import KNeighborsRegressor


from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

def problem2():
    df = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

    X = df.drop(columns='quality')
    y = df['quality']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=69)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.66, random_state=69)

    rand_for_grid = ParameterGrid({
        "n_estimators": [64, 128, 256, 512],
        "max_depth": [None, 1, 2, 4, 8, 16, 32],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
        "min_samples_split": [2, 4, 6, 16, 32],
    })
    
    best_model = None
    best_val = float("inf")
    best_params = None
    print(len(rand_for_grid))
    for params in tqdm(rand_for_grid):
        model = RandomForestRegressor(random_state=69, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        val_mse = mean_squared_error(y_val, model.predict(X_val))
        if val_mse < best_val:
            best_model = model
            best_params = params
            best_val = val_mse

    test_mse = mean_squared_error(y_test, best_model.predict(X_test))
    print("best params randforest:", best_params)
    print("val best MSE rand forest: ", best_val)
    print("test best  MSE: rand forest", test_mse)
   

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    nn_grid = ParameterGrid({
        'n_neighbors': list(range(1, 69)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4]
    })   
    best_model = None
    best_val = float('inf')
    best_params = None
    
    for params in tqdm(nn_grid):
        model = KNeighborsRegressor(**params)
        model.fit(X_train_s, y_train)
        val_mse = mean_squared_error(y_val, model.predict(X_val_s))
        if val_mse < best_val:
            best_model = model
            best_params = params
            best_val = val_mse

    test_mse = mean_squared_error(y_test, best_model.predict(X_test_s))
    print("best_params nn:", best_params)
    print("val best MSE nn:", best_val)
    print("test best MSE nn:", test_mse) 

def problem3():
    df0 = pd.read_csv('dset0-1.csv', header=None)
    df1 = pd.read_csv('dset1-1.csv', header=None)

    y0 = df0.iloc[:, 0]
    y1 = df1.iloc[:, 0]
    X0 = df0.iloc[:, 1:]
    X1 = df1.iloc[:, 1:]


    
    X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.2, random_state=69)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=69)

    num_trees = [1, 10, 100, 300, 1000]
    
    outs0 = []
    outs1 = []

    for i in num_trees:
        clf0 = GradientBoostingClassifier(n_estimators=i, random_state=69).fit(X0_train, y0_train)
        clf1 = GradientBoostingClassifier(n_estimators=i, random_state=69).fit(X1_train, y1_train)

        outs0.append(clf0.score(X0_test, y0_test))
        outs1.append(clf1.score(X1_test, y1_test))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xscale('log')
    ax.scatter(num_trees, outs0, label='dataset 0', c='tab:blue')
    ax.scatter(num_trees, outs1, label='dataset 1', c='tab:orange')
    
    ax.set_xlabel('num trees')
    ax.set_ylabel('accuracy')
    ax.legend()
    plt.show()
