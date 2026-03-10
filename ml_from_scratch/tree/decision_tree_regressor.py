import numpy as np
from utils.helper import Node
from ml_from_scratch.base import Regressor

class DecisionTreeRegression(Regressor):
    def __init__(self,min=20,depth=5):
        self.min = min
        self.depth = depth
        self.root : Node

    def find_best_split(self,X,y):
        lowest_error = float("inf")
        best_threshold = None
        best_feature_index = None
        n,m = X.shape

        for index in range(m): #--------->O(nlogn)

            sorted_indises = np.argsort(X[:,index])
            X_sorted = X[sorted_indises,index]
            y_sorted = y[sorted_indises]
            left_sum = 0
            left_sq_sum = 0
            right_sum = np.sum(y_sorted)
            right_sq_sum = np.sum(y_sorted**2)

            for i in range(1,n):
                if(X_sorted[i] == X_sorted[i-1]): continue

                y_val = y_sorted[i-1]
                left_sum+=y_val
                left_sq_sum+=y_val**2
                right_sum-=y_val
                right_sq_sum-=y_val**2

                left_sq_mean = left_sum**2/i
                right_sq_mean = right_sum**2/(n-i)
                left_error= left_sq_sum - left_sq_mean
                right_error = right_sq_sum - right_sq_mean
                total_error = left_error+right_error

                if total_error < lowest_error :
                    lowest_error = total_error
                    best_threshold = (X_sorted[i] + X_sorted[i-1])/2
                    best_feature_index = index
        
        return best_feature_index,best_threshold

    def build_tree(self,X,y,depth):
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        if depth >= self.depth or X.shape[0] <= self.min :
            return Node(value=np.mean(y))
             
        feature_index,threshold = self.find_best_split(X,y)

        if feature_index is None:
            return Node(value=np.mean(y))

        left = X[:,feature_index] <= threshold
        right = X[:,feature_index] > threshold

        left_tree = self.build_tree(X[left],y[left],depth+1)
        right_tree = self.build_tree(X[right],y[right],depth+1)

        return Node(left=left_tree,right=right_tree,feature_index=feature_index,threshold=threshold)

    def fit(self,X,y):
        self.root = self.build_tree(X,y,0)

    def traverse(self,X,node:Node):
        if(node.value != None): return node.value
        if(X[node.feature_index]<=node.threshold): return self.traverse(X,node.left)
        return self.traverse(X,node.right)

    def predict(self,X):
        n = X.shape[0]
        y = np.zeros(n)
        for row in range(n):
            y[row] = self.traverse(X[row,:],self.root)
        return y
