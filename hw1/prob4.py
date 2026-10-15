import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

def regress_and_predict(data_csv, regressor):
    df = pd.read_csv(data_csv)

    X = df[['anchor_ratio', 'trans_range', 'node_density', 'iterations']]
    Y = df['ale']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)

    regressor.fit(X_train, y_train)
    predicts = regressor.predict(X_test)
    
    return y_test, predicts
    

def problem4():
    lin = linear_model.LinearRegression()
    svm_reg = svm.SVR()
    neigh = KNeighborsRegressor(n_neighbors=2)
    regressors = [lin, svm_reg, neigh]

    for i in regressors:
        y_test, predicts = regress_and_predict('ale.csv', i)
        

        residuals = y_test - predicts
        stats.probplot(residuals, dist='norm', plot=plt)
        plt.title("residual qqplot for predictor" + str(type(i).__name__))
        plt.show()
        plt.close()
        stat, p = stats.normaltest(residuals)
        print(f"statistic = {stat:.4f}, pval = {p:.4g}")
