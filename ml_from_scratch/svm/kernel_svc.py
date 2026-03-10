import numpy as np
from ml_from_scratch.base import Classifier
from utils.helper import Kernel

class KernelSVC(Classifier):
    def __init__(self, epoch=1000, lr=0.01,C=100,Kernel = Kernel.LINEAR,**kwargs):
        self.epoch = epoch
        self.lr = lr
        self.C = C
        self.Kernel = Kernel
        self.kwargs = kwargs
        
    def fit(self, X, y):
        self.X_train = X
        y = y.reshape((-1,1))
        self.y_train = y
        n = X.shape[0]
        
        self.alpha = np.zeros((n,1))
        
        Y = y @ y.T                               # (n,n)
        K = self.Kernel(X,X,**self.kwargs)        # (n,n)
        Q = Y * K                                 # (n,n)
        
        ones = np.ones((n,1))
        
        for _ in range(self.epoch):
            
            # gradient = 1 - Q alpha
            gradient = ones - Q @ self.alpha      
            # gradient ascent
            self.alpha = self.alpha + self.lr * gradient           
            # enforce alpha >= 0 and alpha <= C
            self.alpha = np.clip(self.alpha,0,self.C)     
            # enforce y^T alpha = 0
            self.alpha = self.alpha - (y.T @ self.alpha)/(y.T @ y) * y
        # compute b
        support = (self.alpha.flatten() > 1e-5) & (self.alpha.flatten() < self.C - 1e-5)
        self.b = np.mean(y[support] - (K @ (self.alpha * y))[support])

    def predict(self, X):
        K = self.Kernel(self.X_train,X)
        return np.sign((self.alpha * self.y_train) @ K+ self.b)
