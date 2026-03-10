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

Sorting feature values

```
sorted_indices = np.argsort(X[:,feature])
```

Sorted features allow evaluating candidate thresholds in **O(n)** time.

Incremental statistics are used to update errors efficiently.

Example regression updates

```
left_sum += y
right_sum -= y
```

Squared sum updates

```
left_sq_sum += y²
right_sq_sum -= y²
```

These updates allow computing SSE without recomputing from scratch.

---

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
        self.min = min                  # minimum samples required to split
        self.depth = depth              # maximum tree depth
        self.root : Node                # root node of the tree
```

```python
    def find_best_split(self,X,y):

        lowest_error = float("inf")     # track lowest SSE

        best_threshold = None

        best_feature_index = None

        n,m = X.shape                   # n samples, m features

        for index in range(m):

            sorted_indises = np.argsort(X[:,index])   # sort samples by feature value

            X_sorted = X[sorted_indises,index]

            y_sorted = y[sorted_indises]

            left_sum = 0
            left_sq_sum = 0

            right_sum = np.sum(y_sorted)
            right_sq_sum = np.sum(y_sorted**2)

            for i in range(1,n):

                if(X_sorted[i] == X_sorted[i-1]): continue   # skip identical thresholds

                y_val = y_sorted[i-1]

                # update left partition statistics
                left_sum+=y_val
                left_sq_sum+=y_val**2

                # update right partition statistics
                right_sum-=y_val
                right_sq_sum-=y_val**2

                # compute squared mean terms
                left_sq_mean = left_sum**2/i
                right_sq_mean = right_sum**2/(n-i)

                # compute SSE for both partitions
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
        self.min = min                 # minimum samples required to split
        self.depth = depth             # maximum tree depth
        self.method = method           # split criterion
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

                # update class counts
                left_count[y_val]+=1
                right_count[y_val]-=1

                left_size = i
                right_size = n-i

                # compute class probabilities
                left_prob = left_count/left_size
                right_prob = right_count/right_size

                # compute Gini impurity
                left_gini = 1 - np.sum(left_prob ** 2)
                right_gini = 1 - np.sum(right_prob ** 2)

                weighted_gini = ((left_size / n) * left_gini+ (right_size / n) * right_gini)

                if weighted_gini < lowest_gini :
                    lowest_gini = weighted_gini
                    best_threshold = (X_sorted[i] + X_sorted[i-1])/2
                    best_feature_index = index
        
        return best_feature_index,best_threshold
```

```python
    def find_best_split_entropy(self,X,y):

        lowest_entropy = float("inf")
        best_threshold = None
        best_feature_index = None

        n,m = X.shape

        for index in range(m):

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
```