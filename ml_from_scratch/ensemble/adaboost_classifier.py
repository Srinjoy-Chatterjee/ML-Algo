import numpy as np
from utils.helper import Node
from ml_from_scratch.base import Classifier

class AdaBoostClassifier(Classifier):

    def __init__(self,n_estimator):

        self.n_estimator = n_estimator
        self.roots = []
        self.weights = []
        self.feature_sorted = []

    def find_best_split(self,X,y):

        n_samples,n_features = X.shape
        best_ginni = float("inf")
        best_thresold = None
        best_feature = None
        for feature in range(n_features-1):

            sorted_indx = self.feature_sorted[:,feature]
            X_sorted = X[sorted_indx,feature]
            y_sorted = y[sorted_indx]
            weight_sorted = X[sorted_indx,-1]
            n_classes = int(np.max(y))+1
            left_count = np.zeros(n_classes)
            right_count = np.bincount(y_sorted,weights=weight_sorted,minlength=n_classes)
            left_sum = 0
            right_sum = np.sum(weight_sorted)

            for i in range(1,n_samples):

                if(X_sorted[i]==X_sorted[i-1]): continue

                y_val = y_sorted[i-1]
                w = weight_sorted[i-1]
                left_count[y_val]+=w
                right_count[y_val]-=w

                left_sum+=w
                right_sum-=w

                prob_left = left_count/(left_sum+1e-6)
                prob_right = right_count/(right_sum+1e-6)

                left_ginni = 1-np.sum(prob_left**2)
                right_ginni = 1-np.sum(prob_right**2)


                ginni = left_sum * left_ginni + right_sum * right_ginni

                if(best_ginni>ginni):
                    best_ginni = ginni
                    best_feature = feature
                    best_thresold = (X_sorted[i] + X_sorted[i-1])/2
            
        return best_feature,best_thresold
            
    def build_stump(self,X,y):

        feature,thresold = self.find_best_split(X,y)

        left_indx = X[:,feature] <= thresold
        right_indx = ~left_indx

        left_val = np.argmax(np.bincount(y[left_indx]))
        right_val = np.argmax(np.bincount(y[right_indx]))

        # predictions of the stump
        preds = np.where(left_indx, left_val, right_val)

        # misclassified samples
        misclassified = preds != y

        # weighted error
        total_error = np.sum(X[misclassified, -1]) + 1e-6

        model_weight = 0.5 * np.log((1 - total_error) / total_error)

        # weight update
        X[misclassified, -1] *= np.exp(model_weight)
        X[~misclassified, -1] *= np.exp(-model_weight)

        #normalize
        total = np.sum(X[:,-1]) + 1e-6
        X[:,-1] = X[:,-1]/total

        left = Node(value=left_val)
        right = Node(value=right_val)
        root = Node(feature_index=feature,threshold=thresold,left=left,right=right)

        return root,model_weight

    def fit(self,X,y):

        n_samples = X.shape[0]
        self.n_classes = int(np.max(y))+1
        # add bias
        ones = np.ones((n_samples,1))
        bias = ones/n_samples
        X = np.hstack((X,bias))
        self.feature_sorted = np.argsort(X[:,:-1],axis=0)
        for _ in range(self.n_estimator):
            root,weight = self.build_stump(X,y)
            self.roots.append(root)
            self.weights.append(weight)

    def predict(self,X):

        n_samples = X.shape[0]
        scores = np.zeros((n_samples,self.n_classes))

        for root,weight in zip(self.roots,self.weights):

            left_idx = X[:,root.feature_index] <= root.threshold
            preds = np.where(left_idx, root.left.value, root.right.value)

            scores[np.arange(n_samples), preds] += weight

        return np.argmax(scores,axis=1)
   