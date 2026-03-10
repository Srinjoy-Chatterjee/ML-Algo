import numpy as np
from ml_from_scratch.base import Regressor

class LassoRegression(Regressor):
    def __init__(self,lr = 0.01, epoch = 1000, alpha = 1):
            self.lr = lr
            self.epoch = epoch
            self.alpha = alpha

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
            gradiant = (2/n) * (X.T @ error) + self.alpha * np.sign(self.m)
            self.m = self.m - self.lr * gradiant
        
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
    
class RidgeRegression(Regressor):
    def __init__(self,lr = 0.01,epoch=1000,alpha=1):
        self.lr = lr
        self.epoch = epoch
        self.alpha = alpha

    def fit(self,X,y):
        n = X.shape[0]
        
        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initializing m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape(-1,1)

        for _ in range(self.epoch):
            prediction = X @ self.m
            error = prediction - y
            gradiant = 2/n * X.T @ error + 2 * self.alpha * self.m
            self.m = self.m - self.lr * gradiant
            
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
    
class ElasticNetRegression(Regressor):
    def __init__(self,lr=0.01,epochs=1000,l1_lambda=1,l2_lambda=1):
        self.lr = lr
        self.epochs = epochs
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda 
    
    def fit(self,X,y):
        n = X.shape[0]

        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initialize m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape(-1,1)

        for _ in range(self.epochs):
            prediction = X @ self.m
            error = prediction - y
            gradiant = (2/n) * (X.T @ error) + self.l1_lambda * np.sign(self.m) + 2 * self.l2_lambda * self.m
            self.m = self.m - self.lr * gradiant
          
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m        
   