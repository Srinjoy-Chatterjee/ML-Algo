from enum import Enum
import numpy as np
class Kernel(Enum):
    def linear(x,z): 
        return x @ z.T
    
    def polynomial(x,z,degree = 2,constant = 0) : 
        xz = x @ z.T
        return (xz+constant)**degree
    
    def rbf(x,z,gamma = 1) : 
        x_sq = np.sum(x**2,axis = 1).reshape(-1,1)
        z_sq = np.sum(z**2,axis = 1).reshape(1,-1)
        xz = x @ z.T
        sq_diff = x_sq + z_sq - 2 * xz
        return np.exp(-gamma * sq_diff)

    def sigmoid(x,z,gamma = 1,constant = 0) : 
        xz = x @ z.T
        return np.tanh(gamma * xz + constant)

    LINEAR = linear
    POLYNOMIAL = polynomial
    RBF = rbf
    SIGMOID = sigmoid

    def __call__(self,x,z):
        return self.value(x,z)
    
class Node:
    def __init__(self,left=None,right=None,feature_index=None,threshold=None,value=None):
        self.Left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value