from copy import deepcopy
import numpy as np
from helper import Kernel,Node

class LogisticRegression :

    def __init__(self,epoch=1000,lr=0.1,l1 = 0, l2 = 0):
        self.epoch = epoch
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        
    def fit(self,X,y):
        n = X.shape[0]
        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))
        #initialize m
        self.M = np.zeros((X.shape[1],1))
        #reshape y
        y = np.reshape(y,(-1,1))

        for _ in range(self.epoch):
            z = X @ self.M
            prediction = np.where(z>=0,1/(1+np.exp(-z)),np.exp(z)/(1+np.exp(z)))
            error = prediction - y + 2 * self.l1 * self.M + self.l2 * np.sign(self.M) 
            gradiant = 1/n *  (X.T @ error)
            self.M = self.M- self.lr * gradiant

    def predict(self,X):
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))
        z = X @ self.M
        prediction = np.where(z>=0,1/(1+np.exp(-z)),np.exp(z)/(1+np.exp(z)))
        return (prediction>=0.5).astype(int)
    
class Softmax :
    def __init__(self,epoch=1000,lr=0.1,l1 = 0, l2 = 0):
        self.epoch = epoch
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        
    def softmax(z):
        z = z - np.max(z,axis=1,keepdims=True)
        exp_z = np.exp(z)
        return exp_z/np.sum(exp_z,axis=1,keepdims=True)

    def fit(self,X,y):
        n = X.shape[0]
        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))
        #one shot encode y
        k = len(np.unique(y))
        Y = np.zeros((n,k))
        Y[np.arange(n),y] = 1
        #initialize m
        self.M = np.zeros((X.shape[1],k))

        for _ in range(self.epoch):
            z = X @ self.M
            prediction = self.softmax(z)
            error = prediction - y + 2 * self.l1 * self.M + self.l2 * np.sign(self.M)
            gradiant = 1/n *  (X.T @ error)
            self.M = self.M- self.lr * gradiant

    def predict(self,X):
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))
        z = X @ self.M
        prediction = self.softmax(z)
        return (prediction>=0.5).astype(int)

class KNN:
    def __init__(self,k=3):
        self.k = k
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    def predict(self,X):
        distances = np.linalg.norm(self.X_train[None,:,:] - X[:,None,:],axis=2)
        nearest_neighbors = np.argsort(distances,axis=1)[:,:self.k]
        nearest_labels = self.y_train[nearest_neighbors]
        return np.array([np.bincount(labels).argmax() for labels in nearest_labels])
    
class NaiveBayesMultinomial:
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
    
class NaiveBayesGaussian:
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
    
class SVC:
    def __init__(self, epoch=1000, lr=0.01,C=100):
        self.epoch = epoch
        self.lr = lr
        self.C = C
        
    def fit(self, X, y):
        n = X.shape[0]
        y = y.reshape((-1,1))
        
        self.alpha = np.zeros((n,1))
        
        Y = y @ y.T                 # (n,n)
        K = X @ X.T                 # (n,n)
        Q = Y * K                   # (n,n)
        
        ones = np.ones((n,1))
        
        for _ in range(self.epoch):
            
            # gradient = 1 - Q alpha
            gradient = ones - Q @ self.alpha      
            # gradient ascent
            self.alpha = self.alpha + self.lr * gradient           
            # enforce alpha >= 0 and alpha <= C
            self.alpha = np.clip(self.alpha,0,self.C)     
            # enforce y^T alpha = 0
            self.alpha = self.alpha - (y.T @ self.alpha)/(y.T @ y) * y # (y and alpha should be perpendicular so we remove the parallel part)
        # compute w
        self.w = X.T @ (self.alpha * y) 
        # compute b
        support = (self.alpha.flatten() > 1e-5)
        self.b = np.mean(y[support] - X[support] @ self.w)

    def predict(self, X):
        return np.sign(X @ self.w + self.b)
    
class KernelSVC:
    def __init__(self, epoch=1000, lr=0.01,C=100,Kernel = Kernel.LINEAR,**kwargs):
        self.epoch = epoch
        self.lr = lr
        self.C = C
        self.Kernel = Kernel
        self.kwargs = kwargs
        
    def fit(self, X, y):
        self.X_train = X
        y = y.reshape((-1,1))
        self.y_train = y
        n = X.shape[0]
        
        self.alpha = np.zeros((n,1))
        
        Y = y @ y.T                               # (n,n)
        K = self.Kernel(X,X,**self.kwargs)        # (n,n)
        Q = Y * K                                 # (n,n)
        
        ones = np.ones((n,1))
        
        for _ in range(self.epoch):
            
            # gradient = 1 - Q alpha
            gradient = ones - Q @ self.alpha      
            # gradient ascent
            self.alpha = self.alpha + self.lr * gradient           
            # enforce alpha >= 0 and alpha <= C
            self.alpha = np.clip(self.alpha,0,self.C)     
            # enforce y^T alpha = 0
            self.alpha = self.alpha - (y.T @ self.alpha)/(y.T @ y) * y
        # compute b
        support = (self.alpha.flatten() > 1e-5) & (self.alpha.flatten() < self.C - 1e-5)
        self.b = np.mean(y[support] - (K @ (self.alpha * y))[support])

    def predict(self, X):
        K = self.Kernel(self.X_train,X)
        return np.sign((self.alpha * self.y_train) @ K+ self.b)

class DecisionTreeClassification:
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
        final = []
        for column in y_pred.T:
            final.append(np.bincount(column.astype(int)).argmax())
        
        return final
        
class RandomForestClassification:

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
    
#Boosting

class AdaBoost:

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
        
class GradiantBoost:
    def __init__(self):
        pass

