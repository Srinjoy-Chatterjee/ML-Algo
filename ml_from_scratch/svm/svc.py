import numpy as np
from ml_from_scratch.base import Classifier

class SVC:
    def __init__(self, epoch=1000, lr=0.01,C=100):
        self.epoch = epoch
        self.lr = lr
        self.C = C
        
    def fit(self, X, y):
        n = X.shape[0]
        y = y.reshape((-1,1))
        
        self.alpha = np.zeros((n,1))
        
        Y = y @ y.T                 # (n,n)
        K = X @ X.T                 # (n,n)
        Q = Y * K                   # (n,n)
        
        ones = np.ones((n,1))
        
        for _ in range(self.epoch):
            
            # gradient = 1 - Q alpha
            gradient = ones - Q @ self.alpha      
            # gradient ascent
            self.alpha = self.alpha + self.lr * gradient           
            # enforce alpha >= 0 and alpha <= C
            self.alpha = np.clip(self.alpha,0,self.C)     
            # enforce y^T alpha = 0
            self.alpha = self.alpha - (y.T @ self.alpha)/(y.T @ y) * y # (y and alpha should be perpendicular so we remove the parallel part)
        # compute w
        self.w = X.T @ (self.alpha * y) 
        # compute b
        support = (self.alpha.flatten() > 1e-5)
        self.b = np.mean(y[support] - X[support] @ self.w)

    def predict(self, X):
        return np.sign(X @ self.w + self.b)
