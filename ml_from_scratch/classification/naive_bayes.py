import numpy as np
from ml_from_scratch.base import classifier

class NaiveBayesMultinomial(classifier):
    def __init__(self):
        pass
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.groups,group_index = np.unique(y,return_inverse=True)
        n_groups = len(self.groups)
        probability_groups = np.bincount(group_index)/n_samples
        self.log_probability_groups = np.log(probability_groups)

        y_onehot = np.eye(n_groups)[group_index]

        feature_count = y_onehot.T @ X
        smooth_fc = feature_count+1

        smooth_total = np.sum(smooth_fc,axis=1,keepdims=True)

        self.log_probability_features_per_group = np.log(smooth_fc/smooth_total)
        
    def predict(self,X):
        log_prob = self.log_probability_groups +X @ self.log_probability_features_per_group.T
        return self.groups[np.argmax(log_prob,axis=1)]
    
class NaiveBayesGaussian(classifier):
    def __init__(self):
        pass   
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.groups,group_index = np.unique(y,return_inverse=True)
        n_groups = len(self.groups)
        probability_groups = np.bincount(group_index)/n_samples
        self.log_probability_groups = np.log(probability_groups) 

        y_onehot = np.eye(n_groups)[group_index]

        feature_count = y_onehot.T @ X  
        self.means = feature_count /np.bincount(group_index)[:,None]
        diff = X[:,None,:]-self.means
        diff_sqr = diff ** 2
        weighted_sqr_diff = y_onehot[:,:,None] - diff_sqr
        var_sum = np.sum(weighted_sqr_diff,axis=0)
        self.var = var_sum/np.bincount(group_index)[:,None]

        self.var+=1e-9

    def predict(self,X):
        log_likelihood = self.log_probability_groups+ -0.5 * (np.log(2*np.pi*self.var)+((X[:, None, :] - self.means) ** 2)/self.var)
        return self.groups[np.argmax(log_likelihood,axis=1)]
  