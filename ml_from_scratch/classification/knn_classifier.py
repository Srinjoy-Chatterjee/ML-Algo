import numpy as np
from ml_from_scratch.base import Classifier

class KNN(Classifier):
    def __init__(self,k=3):
        self.k = k
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    def predict(self,X):
        distances = np.linalg.norm(self.X_train[None,:,:] - X[:,None,:],axis=2)
        nearest_neighbors = np.argsort(distances,axis=1)[:,:self.k]
        nearest_labels = self.y_train[nearest_neighbors]
        return np.array([np.bincount(labels).argmax() for labels in nearest_labels])
    