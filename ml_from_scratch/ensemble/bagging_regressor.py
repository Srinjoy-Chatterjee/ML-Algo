import numpy as np
from ml_from_scratch.base import Regressor
from copy import deepcopy

class BaggingRegressor(Regressor):
    def __init__(self,model,n_estimators):
        self.model = model
        self.n_estimators = n_estimators
        self.models = []

    def fit(self,X,y):
        n = X.shape[0]
        for _ in range(self.n_estimators):
            indx = np.random.choice(n,n,replace=True)
            new_X = X[indx]
            new_Y = y[indx]
            model = deepcopy(self.model)
            model.fit(new_X,new_Y)
            self.models.append(model)

    def predict(self,X):
        y_pred = np.array([model.predict(X) for model in self.models])        
        return np.mean(y_pred,axis=0)
  