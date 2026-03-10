# Algorithm
Decision Tree (Regression and Classification)

Decision Trees are supervised learning algorithms that recursively partition the feature space into smaller regions.

The algorithm builds a **tree structure** consisting of:

Root node → internal decision nodes → leaf nodes

Each internal node represents a **feature split**, and each leaf node produces a **prediction**.

The splitting process is **greedy**: at each node, the algorithm selects the split that minimizes impurity.

Two variants are implemented:

• **DecisionTreeRegression** → predicts continuous values  
• **DecisionTreeClassification** → predicts class labels  

Decision Trees are widely used because they are:

• interpretable  
• non-linear  
• able to handle mixed feature types  

---

# Model

A decision tree represents a function

f(x) = prediction stored in the leaf node reached by x

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

Each internal node partitions the feature space into two regions.

---

# Math Implementation

## Decision Tree Regression

Decision Tree Regression chooses splits that minimize **sum of squared errors (SSE)**.

Error in a node

SSE = Σ (yᵢ − ȳ)²

Where

| Symbol | Meaning |
|------|------|
| yᵢ | true target |
| ȳ | mean target value in node |

Equivalent form used in the implementation

SSE = Σ yᵢ² − (Σ yᵢ)² / n

This formula allows fast incremental computation when scanning thresholds.

For a candidate split

Total error

SSE_total = SSE_left + SSE_right

The split with **minimum total error** is selected.

---

## Decision Tree Classification

For classification the algorithm minimizes **impurity**.

Two impurity measures are implemented.

### Gini Index

Gini = 1 − Σ pᵢ²

Where

| Symbol | Meaning |
|------|------|
| pᵢ | probability of class i |

Weighted split impurity

Gini_split = (n_left / n) Gini_left + (n_right / n) Gini_right

---

### Entropy

Entropy = − Σ pᵢ log(pᵢ)

Where

| Symbol | Meaning |
|------|------|
| pᵢ | class probability |

Weighted entropy

Entropy_split = (n_left / n) Entropy_left + (n_right / n) Entropy_right

The split with **minimum impurity** is selected.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| depth | maximum tree depth |
| min | minimum number of samples required to split |
| method | splitting criterion (CART or entropy) |

---

### Derived During Training

| Variable | Meaning |
|------|------|
| threshold | candidate split threshold |
| feature_index | feature used for split |
| left_tree | left subtree |
| right_tree | right subtree |
| root | root node of the tree |

---

# Loop / Vectorized Form

Efficient tree training evaluates candidate split thresholds by **sorting feature values once and updating statistics incrementally** while scanning through the sorted samples.

## Sorting Feature Values

For each feature, samples are sorted so potential thresholds can be checked sequentially.

```
sorted_indices = np.argsort(X[:, feature])
X_sorted = X[sorted_indices, feature]
y_sorted = y[sorted_indices]
```

Sorting ensures that split candidates are evaluated in order.

---

## Regression Tree Updates (SSE)

Regression trees minimize **Sum of Squared Errors (SSE)**.

Instead of recomputing statistics for every split, maintain running sums.

### Running Sum Updates

```
left_sum  += y_val
right_sum -= y_val
```

### Squared Sum Updates

```
left_sq_sum  += y_val**2
right_sq_sum -= y_val**2
```

### Error Computation

```
left_error  = left_sq_sum  - (left_sum**2 / left_size)
right_error = right_sq_sum - (right_sum**2 / right_size)

total_error = left_error + right_error
```

These incremental updates allow evaluating splits efficiently.

---

## Classification Tree Updates (Gini / Entropy)

Classification trees track **class counts** in each partition.

### Class Count Updates

```
left_count[y_val]  += 1
right_count[y_val] -= 1
```

### Gini Impurity

```
p = class_count / node_size
gini = 1 - sum(p**2)
```

Weighted split impurity

```
weighted_gini =
(left_size / n) * gini_left +
(right_size / n) * gini_right
```

### Entropy

```
entropy = -sum(p * log(p))
```

Weighted entropy

```
weighted_entropy =
(left_size / n) * entropy_left +
(right_size / n) * entropy_right
```

# Algorithm Steps

## Training

1 Start with full dataset at root node  

2 For each feature

Sort feature values

3 For each possible threshold

Split dataset into

Left subset  
Right subset  

4 Compute impurity

Regression

SSE_left + SSE_right

Classification

Weighted Gini or Entropy

5 Select split with lowest impurity.

6 Create child nodes

Left child  
Right child

7 Recursively repeat until stopping condition

Stopping conditions

• maximum depth reached  
• minimum sample size reached  
• node is pure

---

## Prediction

1 Start from root node

2 Compare feature value with threshold

x[feature_index] ≤ threshold → go left  
x[feature_index] > threshold → go right

3 Continue until reaching a leaf node.

4 Return prediction stored in leaf node.

---

# Time Complexity

Best split search

O(n log n)

Where sorting is required per feature.

Tree construction

O(n log n × depth)

Prediction complexity

O(depth)

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |

---

# Code Implementation

## Decision Tree Regression

```python
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
```

---

## Decision Tree Classification

```python
class DecisionTreeClassification(Classifier):
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
            unique_y_len = int(np.max(y))+1 
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