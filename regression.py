from copy import deepcopy
import numpy as np
from itertools import combinations_with_replacement
from helper import Kernel,Node

class LinearRegression:

    def __init__(self,lr = 0.01, epoch = 1000):
            self.lr = lr
            self.epoch = epoch

    def fit(self,X,y):
        n = X.shape[0]

        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initialize m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape((-1,1))

        for _ in range(self.epoch):
            prediction = X @ self.m
            error = prediction -y
            gradiant = (2/n) * (X.T @ error)
            self.m = self.m - self.lr * gradiant
        
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
    
class LassoRegression:
    def __init__(self,lr = 0.01, epoch = 1000, alpha = 1):
            self.lr = lr
            self.epoch = epoch
            self.alpha = alpha

    def fit(self,X,y):
        n = X.shape[0]

        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initialize m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape((-1,1))

        for _ in range(self.epoch):
            prediction = X @ self.m
            error = prediction -y
            gradiant = (2/n) * (X.T @ error) + self.alpha * np.sign(self.m)
            self.m = self.m - self.lr * gradiant
        
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
    
class RidgeRegression:
    def __init__(self,lr = 0.01,epoch=1000,alpha=1):
        self.lr = lr
        self.epoch = epoch
        self.alpha = alpha

    def fit(self,X,y):
        n = X.shape[0]
        
        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initializing m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape(-1,1)

        for _ in range(self.epoch):
            prediction = X @ self.m
            error = prediction - y
            gradiant = 2/n * X.T @ error + 2 * self.alpha * self.m
            self.m = self.m - self.lr * gradiant
            
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
    
class ElasticNetRegression:
    def __init__(self,lr=0.01,epochs=1000,l1_lambda=1,l2_lambda=1):
        self.lr = lr
        self.epochs = epochs
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda 
    
    def fit(self,X,y):
        n = X.shape[0]

        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initialize m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape(-1,1)

        for _ in range(self.epochs):
            prediction = X @ self.m
            error = prediction - y
            gradiant = (2/n) * (X.T @ error) + self.l1_lambda * np.sign(self.m) + 2 * self.l2_lambda * self.m
            self.m = self.m - self.lr * gradiant
          
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m        
    
class PolinomialRegression:
    def __init__(self,lr=0.01,epoch=1000,degree=1,interaction=False):
        self.lr = lr
        self.epoch = epoch
        self.degree = degree
        self.interaction = interaction
        self.lr =  LinearRegression()
    def fit(self,X,y):
        features = self.generate_feature_set(X)
        return self.lr.fit(features,y)

    def predict(self,X):
        features = self.generate_feature_set(X)
        return self.lr.predict(features)
    
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
    
class KernelSVR:
    def __init__(self, epoch=1000, lr=0.001, C=100, epsilon=0.1, 
                 Kernel=Kernel.LINEAR, **kwargs):
        
        self.epoch = epoch
        self.lr = lr
        self.C = C
        self.epsilon = epsilon
        self.Kernel = Kernel
        self.kwargs = kwargs

    def fit(self, X, y):
        self.X_train = X
        y = y.reshape(-1,1)
        self.y_train = y
        
        n = X.shape[0]
        
        # Two alpha vectors
        self.alpha = np.zeros((n,1))
        self.alpha_star = np.zeros((n,1))
        
        # Kernel matrix
        K = self.Kernel(X, X, **self.kwargs)
        
        for _ in range(self.epoch):
            
            beta = self.alpha - self.alpha_star   # (n,1)
            f = K @ beta                          # (n,1)
            
            # Gradients from SVR dual
            grad_alpha = y - f - self.epsilon
            grad_alpha_star = -y + f - self.epsilon
            
            # Gradient ascent
            self.alpha += self.lr * grad_alpha
            self.alpha_star += self.lr * grad_alpha_star
            
            # Box constraints
            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)
            
            # Enforce equality constraint: sum(alpha - alpha*) = 0
            beta = self.alpha - self.alpha_star
            correction = np.sum(beta) / n
            
            self.alpha -= correction / 2
            self.alpha_star += correction / 2
            
            # Clip again
            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)
        
        # Final beta
        self.beta = self.alpha - self.alpha_star
        
        # -------- Compute bias b --------
        
        f_train = K @ self.beta
        
        idx1 = np.where((self.alpha > 1e-5) & 
                        (self.alpha < self.C-1e-5))[0]
        
        idx2 = np.where((self.alpha_star > 1e-5) & 
                        (self.alpha_star < self.C-1e-5))[0]
        
        b_vals = []
        
        for i in idx1:
            b_vals.append(y[i] - self.epsilon - f_train[i])
        
        for i in idx2:
            b_vals.append(y[i] + self.epsilon - f_train[i])
        
        self.b = np.mean(b_vals) if b_vals else 0

    def predict(self, X):
        K = self.Kernel(self.X_train, X, **self.kwargs)
        return (K.T @ self.beta + self.b)
    
class DecisionTreeRegression:
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


# Bagging

class Bagging:
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
    
class RandomForestRegression:
    def __init__(self,n_estimator,depth=5,min=20):
        self.n_estimator = n_estimator
        self.depth = depth
        self.min = min
        self.roots = []

    def find_best_split(self,X,y):
        n_samples,n_features = X.shape
        k_features = np.random.choice(n_features,int(np.sqrt(n_features)),replace=False)
        min_sse = float('inf')
        best_thresold = None
        best_feature_indx = None

        for feature in k_features:

            sorted_indx = np.argsort(X[:,feature])
            X_sorted = X[sorted_indx,feature]
            y_sorted = y[sorted_indx]
            
            left_sum = 0
            left_sq_sum = 0
            right_sum = np.sum(y_sorted)
            right_sq_sum = np.sum(y_sorted**2)

            for i in range(1,n_samples):

                if(X_sorted[i]==X_sorted[i-1]): continue

                y_val = y_sorted[i-1]
                left_sum+=y_val
                right_sum-=y_val
                left_sq_sum+=y_val**2
                right_sq_sum-=y_val**2
                left_w = left_sq_sum - (left_sum)**2/i
                right_w = right_sq_sum - (right_sum)**2/(n_samples-i)
                sse = left_w + right_w

                if(sse<min_sse):
                    min_sse = sse
                    best_thresold = (X_sorted[i-1] + X_sorted[i])/2
                    best_feature_indx = feature

        return best_feature_indx,best_thresold

    def build_tree(self,X,y,depth):

        n_samples,_ = X.shape

        if(depth==self.depth or n_samples <= self.min):
            return Node(value=np.mean(y))
        
        feature_indx,thresold = self.find_best_split(X,y)

        if feature_indx is None :
            return Node(value=np.mean(y))     
               
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
        y_pred = np.zeros(n)
        for i in range(n):
            pred = []
            for root in self.roots:
                pred.append(self.traverse(X[i],root))
            y_pred[i] = np.mean(pred)
        
        return y_pred
    
# Boosting 

class AdaBoost:

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
    
