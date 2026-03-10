import numpy as np
from ml_from_scratch.base import Regressor

class LinearRegression(Regressor):

    def __init__(self,lr = 0.01, epoch = 1000):
            self.lr = lr
            self.epoch = epoch

    def fit(self,X,y):
        n = X.shape[0]

        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initialize m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape((-1,1))

        for _ in range(self.epoch):
            prediction = X @ self.m
            error = prediction -y
            gradiant = (2/n) * (X.T @ error)
            self.m = self.m - self.lr * gradiant
        
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
    