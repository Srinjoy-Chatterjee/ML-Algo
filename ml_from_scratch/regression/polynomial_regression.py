import numpy as np
from itertools import combinations_with_replacement
from ml_from_scratch.base import Regressor
from ml_from_scratch.regression import LinearRegression

class PolinomialRegression(Regressor):

    def __init__(self,lr=0.01,epoch=1000,degree=1,interaction=False):
        self.lr = lr
        self.epoch = epoch
        self.degree = degree
        self.interaction = interaction
        self.lr =  LinearRegression()

    def generate_feature_set(self,X):
        _,X_features = X.shape
        features = []
        if self.interaction:
            for k in range(1,self.degree+1):
                for comb in combinations_with_replacement(range(X_features),k):
                    for index in comb:
                        new_feature *=X[:,index]
                    features.append(new_feature)
        else:
            
            for k in range(1,self.degree+1):
                new_feature = X**k
                features.append(new_feature)

        return np.column_stack(features)

    def fit(self,X,y):
        features = self.generate_feature_set(X)
        return self.lr.fit(features,y)

    def predict(self,X):
        features = self.generate_feature_set(X)
        return self.lr.predict(features)
 