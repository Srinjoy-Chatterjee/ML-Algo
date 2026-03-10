import numpy as np
from utils.helper import Node
from ml_from_scratch.base import Classifier

class RandomForestClassification(Classifier):

    def __init__(self,n_estimator,depth=5,min=20):
        self.n_estimator = n_estimator
        self.depth = depth
        self.min = min
        self.roots = []

    def find_best_split(self,X,y):

        n_samples,n_features = X.shape
        n_classes = np.max(y)+1
        k_features = np.random.choice(n_features,int(np.sqrt(n_features)),replace=False)
        min_ginni = float('inf')
        best_thresold = None
        best_feature_indx = None

        for feature in k_features:

            sorted_indx = np.argsort(X[:,feature])
            X_sorted = X[sorted_indx,feature]
            y_sorted = y[sorted_indx]
            
            left_count = np.zeros(n_classes)
            right_count = np.bincount(y_sorted,minlength=n_classes)

            for i in range(1,n_samples):

                if(X_sorted[i]==X_sorted[i-1]): continue

                y_val = y_sorted[i-1]
                left_count[y_val]+=1
                right_count[y_val]-=1
                left = i
                right = n_samples-i
                left_w = 1-np.sum((left_count/left)**2)
                right_w = 1-np.sum((right_count/right)**2)
                ginni = left/n_samples * left_w + right/n_samples * right_w

                if(ginni<min_ginni):
                    min_ginni = ginni
                    best_thresold = (X_sorted[i-1] + X_sorted[i])/2
                    best_feature_indx = feature

        return best_feature_indx,best_thresold

    def build_tree(self,X,y,depth):

        n_samples,_ = X.shape

        if(depth==self.depth or n_samples <= self.min or len(np.unique(y))==1):
            return Node(value=np.argmax(np.bincount(y)))
        
        feature_indx,thresold = self.find_best_split(X,y)

        if feature_indx is None :
            counts = np.bincount(y)
            return Node(value=np.argmax(counts))     
               
        left_indx = X[:,feature_indx]<=thresold
        right_indx = ~left_indx
        left_x = X[left_indx]
        left_y = y[left_indx]
        right_x = X[right_indx]
        right_y = y[right_indx]
        left_node = self.build_tree(left_x,left_y,depth=depth+1)
        right_node = self.build_tree(right_x,right_y,depth=depth+1)

        return Node(left=left_node,right=right_node,feature_index=feature_indx,threshold=thresold)

    def fit(self,X,y):

        n,_ = X.shape

        for _ in range(self.n_estimator):
            indx = np.random.choice(n,n,replace=True)
            new_X = X[indx]
            new_y = y[indx]
            self.roots.append(self.build_tree(new_X,new_y,0))

    def traverse(self,X,node:Node):
        if(node.value is not None) : return node.value
        if(X[node.feature_index]<=node.threshold) : return self.traverse(X,node.left)
        else : return self.traverse(X,node.right)

    def predict(self,X):
        n = X.shape[0]
        y_pred = np.zeros(n, dtype=int)
        for i in range(n):
            preds = np.array([self.traverse(X[i],root) for root in self.roots])
            counts = np.bincount(preds)
            y_pred[i] = np.argmax(counts)
        
        return y_pred
  