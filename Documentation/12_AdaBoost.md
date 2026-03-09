# Algorithm
AdaBoost (Adaptive Boosting)

AdaBoost is an ensemble learning algorithm that combines multiple **weak learners** to create a strong model.

The key idea is:

• Train models sequentially  
• Increase the importance of misclassified samples  
• Combine models using weighted voting  

Each new model focuses more on the samples that previous models predicted incorrectly.

In this implementation the weak learner is a **decision stump** (a decision tree with depth 1).

Two variants are implemented:

AdaBoost Classification  
AdaBoost Regression

---

# Model

The final prediction is a weighted combination of weak learners.

Classification

f(x) = sign( Σ αₜ hₜ(x) )

Regression

f(x) = ( Σ αₜ hₜ(x) ) / ( Σ αₜ )

Where

| Symbol | Meaning |
|------|------|
| hₜ(x) | prediction from weak learner t |
| αₜ | weight of learner t |
| T | number of learners |

Weak learners with lower error receive higher weights.

---

# Loss Function

AdaBoost minimizes **exponential loss**.

Classification loss

L = Σ exp(−y f(x))

This loss increases rapidly when predictions are incorrect, forcing the next learner to focus on difficult samples.

---

# Gradient

AdaBoost does not explicitly compute gradients.

Instead it performs **multiplicative weight updates** on training samples.

Samples that are misclassified receive larger weights.

---

# Gradient Descent Update

Not applicable.

AdaBoost uses **adaptive weight updates instead of gradient descent**.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| n_estimator | number of weak learners |

### Derived

| Variable | Meaning |
|------|------|
| roots | decision stumps |
| weights | model weights |
| sample_weights | importance of each training sample |
| threshold | split threshold |
| feature_index | selected feature |

---

# Vectorized Form

Weighted error

```
total_error = sum(sample_weights[misclassified])
```

Model weight

```
alpha = 0.5 * log((1 - error) / error)
```

Weight update

```
w_i = w_i * exp(±alpha)
```

Normalization

```
w_i = w_i / sum(w_i)
```

---

# Algorithm Steps

Training

1 Initialize sample weights equally  
2 Train weak learner  
3 Compute weighted error  
4 Compute model weight  
5 Increase weight of misclassified samples  
6 Normalize weights  
7 Repeat for all estimators  

Prediction

1 Collect predictions from each weak learner  
2 Combine predictions using model weights  

Classification → weighted vote  
Regression → weighted average

---

# Time Complexity

Training

O(n_estimators × tree_training_cost)

Prediction

O(n_estimators × tree_depth)

Where

| Symbol | Meaning |
|------|------|
| n_estimators | number of weak learners |
| n | number of samples |
| d | number of features |

---

# Implementation

## AdaBoost Classification

```python
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
   
```

---

## AdaBoost Regression

Regression version updates weights based on prediction error magnitude.

```python
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
    

```