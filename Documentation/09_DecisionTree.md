# Algorithm
Decision Tree (Regression and Classification)

Decision Trees are supervised learning algorithms that split the dataset into smaller subsets based on feature values.

The model builds a **tree structure** consisting of:

Root node → internal decision nodes → leaf nodes

Each internal node represents a **feature split** and each leaf node represents a **prediction**.

Two variants:

DecisionTreeRegression → predicts continuous values  
DecisionTreeClassification → predicts discrete classes

---

# Model

A decision tree represents a function

f(x) = prediction at leaf node

Prediction depends on a sequence of feature comparisons.

Example

if x₁ ≤ t₁  
 if x₂ ≤ t₂ → leaf₁  
 else → leaf₂  
else → leaf₃

Where

| Symbol | Meaning |
|------|------|
| x | input feature vector |
| t | threshold value |
| leaf | predicted value |

---

# Loss Function

### Regression

Decision Tree Regression minimizes **sum of squared errors (SSE)**.

Error in a node

SSE = Σ (yᵢ − ȳ)²

Where

| Symbol | Meaning |
|------|------|
| yᵢ | true target |
| ȳ | mean of node targets |

Split quality is measured by minimizing

SSE_left + SSE_right

---

### Classification

Two common impurity measures

Gini Index

Gini = 1 − Σ pᵢ²

Entropy

Entropy = − Σ pᵢ log(pᵢ)

Where

| Symbol | Meaning |
|------|------|
| pᵢ | probability of class i |

The best split minimizes weighted impurity.

---

# Gradient

Decision Trees do **not use gradients**.

Splits are selected by evaluating impurity measures directly.

---

# Gradient Descent Update

Not applicable.

Decision Trees use **greedy splitting** instead of gradient optimization.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| depth | maximum tree depth |
| min | minimum samples to split |
| method | splitting criterion |

### Derived

| Variable | Meaning |
|------|------|
| threshold | best split threshold |
| feature_index | feature used for split |
| left_tree | left subtree |
| right_tree | right subtree |

---

# Vectorized Form

Sorting feature values

```
sorted_indices = np.argsort(X[:,feature])
```

Efficiently updating class counts

```
left_count += 1
right_count -= 1
```

Regression error computation

```
left_error = left_sq_sum - left_mean²
```

---

# Algorithm Steps

Training

1 Start with full dataset  
2 Find best feature and threshold  
3 Split dataset into left and right subsets  
4 Recursively repeat splitting  
5 Stop when depth limit or minimum samples reached  

Prediction

1 Start at root node  
2 Compare feature value with threshold  
3 Move left or right child node  
4 Repeat until leaf node reached  

---

# Time Complexity

Finding best split

O(n log n)

Tree construction

O(n log n × depth)

Prediction

O(depth)

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |

---

# Implementation

## Decision Tree Regression

```python
class DecisionTreeRegression:
    def __init__(self,min=20,depth=5):
        self.min = min
        self.depth = depth
        self.root : Node
```

```python
    def find_best_split(self,X,y):

        lowest_error = float("inf")

        best_threshold = None

        best_feature_index = None

        n,m = X.shape

        for index in range(m):

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
```

---

## Decision Tree Classification

Gini Index

```
Gini = 1 − Σ pᵢ²
```

Entropy

```
Entropy = − Σ pᵢ log(pᵢ)
```

```python
class DecisionTreeClassification:
    def __init__(self,min=20,depth=5,method='CART'):
        self.min = min
        self.depth = depth
        self.method = method
        self.root : Node

    def find_best_split_ginni(self,X,y):
        lowest_gini = float("inf")
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

```