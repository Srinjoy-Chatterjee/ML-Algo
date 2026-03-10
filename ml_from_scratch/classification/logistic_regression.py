import numpy as np
from ml_from_scratch.base import Classifier

class LogisticRegression(Classifier) :

    def __init__(self,epoch=1000,lr=0.1,l1 = 0, l2 = 0):
        self.epoch = epoch
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        
    def fit(self,X,y):
        n = X.shape[0]
        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))
        #initialize m
        self.M = np.zeros((X.shape[1],1))
        #reshape y
        y = np.reshape(y,(-1,1))

        for _ in range(self.epoch):
            z = X @ self.M
            prediction = np.where(z>=0,1/(1+np.exp(-z)),np.exp(z)/(1+np.exp(z)))
            error = prediction - y 
            gradiant = 1/n *  (X.T @ error) + 2 * self.l2 * self.M + self.l1 * np.sign(self.M)
            self.M = self.M- self.lr * gradiant

    def predict(self,X):
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))
        z = X @ self.M
        prediction = np.where(z>=0,1/(1+np.exp(-z)),np.exp(z)/(1+np.exp(z)))
        return (prediction>=0.5).astype(int)
 