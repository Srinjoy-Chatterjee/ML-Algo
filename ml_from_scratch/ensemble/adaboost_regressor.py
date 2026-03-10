import numpy as np
from utils.helper import Node
from ml_from_scratch.base import Regressor

class AdaBoostRegressor(Regressor):

    def __init__(self,n_estimator):

        self.n_estimator = n_estimator
        self.roots = []
        self.weights = []
        self.feature_sorted = []

    def find_best_split(self,X,y):

        n_samples,n_features = X.shape
        best_sse = float("inf")
        best_thresold = None
        best_feature = None
        for feature in range(n_features-1):

            sorted_indx = self.feature_sorted[:,feature]
            X_sorted = X[sorted_indx,feature]
            y_sorted = y[sorted_indx]
            weight_sorted = X[sorted_indx,-1]
            left_sum = 0
            left_sq_sum = 0
            left_weight_sum = 0
            right_sum = np.sum( weight_sorted * y_sorted)
            right_sq_sum = np.sum( weight_sorted * y_sorted**2)
            right_weight_sum = np.sum(weight_sorted)

            for i in range(1,n_samples):

                if(X_sorted[i]==X_sorted[i-1]): continue

                y_val = y_sorted[i-1]
                w = weight_sorted[i-1]
                
                left_sum += w*y_val
                right_sum -= w*y_val

                left_sq_sum += w*y_val**2
                right_sq_sum-= w*y_val**2

                left_weight_sum += w
                right_weight_sum -=w

                left = left_sq_sum - (left_sum**2)/(left_weight_sum+1e-6)
                right = right_sq_sum - (right_sum**2)/(right_sq_sum+1e-6)

                sse = left + right_sum * right

                if(best_sse>sse):
                    best_sse = sse
                    best_feature = feature
                    best_thresold = (X_sorted[i] + X_sorted[i-1])/2
            
        return best_feature,best_thresold
            
    def build_stump(self,X,y):

        feature,thresold = self.find_best_split(X,y)

        left_indx = X[:,feature] <= thresold
        right_indx = ~left_indx

        left_val = np.mean(y[left_indx])
        right_val = np.mean(y[right_indx])

        # predictions of the stump
        preds = np.where(left_indx, left_val, right_val)

        # weighted error
        total_error = np.abs(y-preds) + 1e-6
        model_weight = 0.5 * np.log((1 - total_error) / total_error)
        beta = np.sum(model_weight * total_error)
        beta = beta/(1-beta)

        # weight update
        X[:, -1] *= beta**(1-total_error)

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
        preds = np.zeros(n_samples)
        weight_sum = np.sum(self.weights)

        for root,alpha in zip(self.roots,self.weights):

            left_idx = X[:,root.feature_index] <= root.threshold
            stump_pred = np.where(left_idx, root.left.value, root.right.value)

            preds += alpha * stump_pred

        return preds / weight_sum
