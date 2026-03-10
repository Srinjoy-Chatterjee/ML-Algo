import numpy as np
from utils.helper import Node
from ml_from_scratch.base import Classifier

class DecisionTreeClassification(Classifier):
    def __init__(self,min=20,depth=5,method='CART'):
        self.min = min
        self.depth = depth
        self.method = method
        self.root : Node

    # def find_gini(self,vector):
    #     _,count = np.unique(vector,return_counts=True)
    #     total = len(vector)
    #     prob = count/total
    #     return 1- np.sum(prob ** 2)

    # def compute_weighted_gini(self,left,right):
    #     left_weight = self.find_gini(left)
    #     right_weight = self.find_gini(right)
    #     n = len(left)+len(right)
    #     left_prob = len(left)/n
    #     right_prob = len(right)/n
    #     return left_prob*left_weight+right_prob*right_weight

    def find_best_split_ginni(self,X,y):
        lowest_gini = float("inf")
        best_threshold = None
        best_feature_index = None
        n,m = X.shape

        # for index,feature in enumerate(X.T): ----->O(n2)
            # thresholds = np.unique(feature)
            # for t in thresholds:
            #     left = y[feature <= t]
            #     right = y[feature > t]
            #     gini = self.compute_weighted_gini(left,right)
            #     if gini < lowest_gini :
            #         lowest_gini = gini
            #         best_threshold = t
            #         best_feature_index = index

        for index in range(m): #--------->O(nlogn)

            sorted_indises = np.argsort(X[:,index])
            X_sorted = X[sorted_indises,index]
            y_sorted = y[sorted_indises]
            unique_y_len = int(np.max(y))+1 # cause we are using y value as index for l/r count
            left_count = np.zeros(unique_y_len)
            right_count = np.bincount(y_sorted,minlength=unique_y_len)

            for i in range(1,n):
                if(X_sorted[i] == X_sorted[i-1]): continue

                y_val = y_sorted[i-1]
                left_count[y_val]+=1
                right_count[y_val]-=1
                left_size = i
                right_size = n-i

                left_prob = left_count/left_size
                right_prob = right_count/right_size
                left_gini = 1 - np.sum(left_prob ** 2)
                right_gini = 1 - np.sum(right_prob ** 2)
                weighted_gini = ((left_size / n) * left_gini+ (right_size / n) * right_gini)

                if weighted_gini < lowest_gini :
                    lowest_gini = weighted_gini
                    best_threshold = (X_sorted[i] + X_sorted[i-1])/2
                    best_feature_index = index
        
        return best_feature_index,best_threshold

    def find_best_split_entropy(self,X,y):
        lowest_entropy = float("inf")
        best_threshold = None
        best_feature_index = None
        n,m = X.shape

        for index in range(m): #--------->O(nlogn)

            sorted_indises = np.argsort(X[:,index])
            X_sorted = X[sorted_indises,index]
            y_sorted = y[sorted_indises]
            unique_y_len = int(np.max(y))+1 # cause we are using y value as index for l/r count
            left_count = np.zeros(unique_y_len)
            right_count = np.bincount(y_sorted,minlength=unique_y_len)

            for i in range(1,n):
                if(X_sorted[i] == X_sorted[i-1]): continue

                y_val = y_sorted[i-1]
                left_count[y_val]+=1
                right_count[y_val]-=1
                left_size = i
                right_size = n-i

                left_prob = left_count/left_size
                left_prob = left_prob[left_prob>0]
                right_prob = right_count/right_size
                right_prob = right_prob[right_prob>0]
                left_entropy = - np.sum(left_prob ** np.log(left_prob))
                right_entropy = - np.sum(right_prob ** np.log(right_prob))
                weighted_entropy = ((left_size / n) * left_entropy+ (right_size / n) * right_entropy)

                if weighted_entropy < lowest_entropy :
                    lowest_entropy = weighted_entropy
                    best_threshold = (X_sorted[i] + X_sorted[i-1])/2
                    best_feature_index = index

        return best_feature_index,best_threshold

    def build_tree(self,X,y,depth):
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        if depth >= self.depth or X.shape[0] <= self.min :
            counts = np.bincount(y)
            return Node(value=np.argmax(counts))
             
        feature_index,threshold = self.find_best_split_ginni(X,y) if self.method == 'CART' else self.find_best_split_entropy(X,y)

        if feature_index is None:
            counts = np.bincount(y)
            return Node(value=np.argmax(counts))

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
