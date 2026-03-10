# Algorithm
Random Forest (Regression and Classification)

Random Forest is an **ensemble learning algorithm** that combines multiple decision trees to improve predictive performance.

The algorithm reduces overfitting and variance using two key mechanisms:

1. **Bootstrap Aggregation (Bagging)**  
   Each tree is trained on a random dataset sampled **with replacement** from the original dataset.

2. **Random Feature Selection**  
   At each split in the tree, only a **random subset of features** is considered.

These two sources of randomness reduce correlation between trees, which improves ensemble generalization.

Two variants are implemented:

RandomForestRegression → predicts continuous values  
RandomForestClassification → predicts discrete classes

---

# Model

Random Forest combines predictions from multiple decision trees.

Assume there are **M trees**.

Each tree learns a function

hᵢ(x)

Where

| Symbol | Meaning |
|------|------|
| x | input feature vector |
| hᵢ(x) | prediction of tree i |
| M | number of trees |

---

### Regression Prediction

Predictions are averaged.

ŷ = (1/M) Σ hᵢ(x)

---

### Classification Prediction

Predictions are determined by majority vote.

ŷ = mode(h₁(x), h₂(x), … , hₘ(x))

---

# Math Implementation

## Bootstrap Sampling

Training datasets are generated using bootstrap sampling.

Given dataset

D = {x₁, x₂, … , xₙ}

A bootstrap sample is created as

D₁ = sample(D, n, replace=True)

This process is repeated **M times** to create datasets for M trees.

---

## Random Feature Selection

At each split only a subset of features is considered.

If the dataset has **d features**

k = √d

Then a random subset of **k features** is selected.

---

## Regression Split Criterion

Decision trees minimize **sum of squared errors (SSE)**.

SSE = Σ (yᵢ − ȳ)²

Where

| Symbol | Meaning |
|------|------|
| yᵢ | true target |
| ȳ | mean of node values |

The best split minimizes

SSE_left + SSE_right

---

## Classification Split Criterion

Random Forest classification uses **Gini impurity**.

Gini = 1 − Σ pᵢ²

Where

| Symbol | Meaning |
|------|------|
| pᵢ | probability of class i |

The best split minimizes weighted impurity

Gini_split = (n_left/n) Gini_left + (n_right/n) Gini_right

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| n_estimator | number of trees |
| depth | maximum tree depth |
| min | minimum samples required for split |

---

### Derived During Training

| Variable | Meaning |
|------|------|
| roots | list of tree roots |
| k_features | randomly selected features |
| threshold | split threshold |
| feature_index | selected feature |

---

### Derived During Prediction

| Variable | Meaning |
|------|------|
| preds | predictions from trees |
| y_pred | final ensemble prediction |

---

# Loop / Vectorized Form

Bootstrap sampling

```
indices = np.random.choice(n, n, replace=True)
```

Random feature selection

```
k_features = np.random.choice(d, sqrt(d), replace=False)
```

Prediction aggregation

Regression

```
mean(predictions)
```

Classification

```
majority_vote(predictions)
```

---

# Complete Algorithm Steps

## Training

1 Initialize empty list of trees.

2 Repeat **n_estimators times**

Create bootstrap dataset

X_sample = sample(X)

Train decision tree using sampled data.

At each node

Select random subset of features.

Find best split among selected features.

Grow tree recursively.

Store tree root.

---

## Prediction

For each input sample

Run sample through all trees.

Collect predictions.

Regression

Compute average prediction.

Classification

Compute majority vote.

---

# Time Complexity

Let

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |
| M | number of trees |
| depth | tree depth |

---

Training complexity

O(M × n log n × depth)

---

Prediction complexity

O(M × depth)

---

# Code Implementation

## Random Forest Regression

```python
class RandomForestRegression(Regressor):
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
```

---

## Random Forest Classification

Classification trees use **Gini impurity**.

```
Gini = 1 − Σ pᵢ²
```

```python
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
```