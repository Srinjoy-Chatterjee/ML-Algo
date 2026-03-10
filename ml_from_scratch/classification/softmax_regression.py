import numpy as np

class Softmax :
    def __init__(self,epoch=1000,lr=0.1,l1 = 0, l2 = 0):
        self.epoch = epoch
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        
    def softmax(self,z):
        z = z - np.max(z,axis=1,keepdims=True)
        exp_z = np.exp(z)
        return exp_z/np.sum(exp_z,axis=1,keepdims=True)

    def fit(self,X,y):
        n = X.shape[0]
        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))
        #one shot encode y
        k = len(np.unique(y))
        Y = np.zeros((n,k))
        Y[np.arange(n),y] = 1
        #initialize m
        self.M = np.zeros((X.shape[1],k))

        for _ in range(self.epoch):
            z = X @ self.M
            prediction = self.softmax(z)
            error = prediction - Y 
            gradiant = 1/n *  (X.T @ error) + 2 * self.l2 * self.M + self.l1 * np.sign(self.M)
            self.M = self.M- self.lr * gradiant

    def predict(self,X):
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))
        z = X @ self.M
        prediction = self.softmax(z)
        return np.argmax(prediction, axis=1)
