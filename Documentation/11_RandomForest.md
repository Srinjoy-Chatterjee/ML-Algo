# Algorithm
Random Forest (Regression and Classification)

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

It improves prediction accuracy and reduces overfitting by using two key ideas:

1. **Bagging (Bootstrap Sampling)**  
   Each tree is trained on a random bootstrap sample of the dataset.

2. **Feature Randomness**  
   At each split, only a random subset of features is considered.

This randomness makes the trees less correlated, which improves ensemble performance.

Two versions are implemented:

RandomForestRegression → predicts continuous values  
RandomForestClassification → predicts class labels

---

# Model

Random Forest combines predictions from multiple decision trees.

If there are **M trees**

Regression prediction

ŷ = (1/M) Σ hᵢ(x)

Classification prediction

ŷ = mode(h₁(x), h₂(x), … , hₘ(x))

Where

| Symbol | Meaning |
|------|------|
| hᵢ(x) | prediction from tree i |
| M | number of trees |

---

# Loss Function

Each tree optimizes the same objective as a standard decision tree.

### Regression

Sum of Squared Errors

SSE = Σ (yᵢ − ȳ)²

### Classification

Gini impurity

Gini = 1 − Σ pᵢ²

Where

| Symbol | Meaning |
|------|------|
| pᵢ | probability of class i |

Random Forest does not introduce a new loss function.

---

# Gradient

Random Forest does not use gradients.

It relies on greedy splitting in decision trees.

---

# Gradient Descent Update

Not applicable.

Random Forest uses **bootstrap aggregation and random feature selection**.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| n_estimator | number of trees |
| depth | maximum tree depth |
| min | minimum samples per node |

### Derived

| Variable | Meaning |
|------|------|
| roots | list of tree roots |
| k_features | randomly selected feature subset |
| threshold | split threshold |
| feature_index | selected feature |

---

# Vectorized Form

Bootstrap sampling

```
indices = np.random.choice(n,n,replace=True)
```

Random feature selection

```
k_features = random subset of sqrt(d)
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

# Algorithm Steps

Training

1 Repeat n_estimators times  
2 Create bootstrap sample of dataset  
3 Build decision tree on sampled data  
4 At each split choose random subset of features  
5 Store the root of each tree  

Prediction

1 Run input through each tree  
2 Collect predictions  
3 Aggregate predictions  

Regression → average  
Classification → majority vote

---

# Time Complexity

Training

O(n_estimators × tree_training_cost)

Prediction

O(n_estimators × tree_depth)

Where

| Symbol | Meaning |
|------|------|
| n_estimators | number of trees |
| n | number of samples |
| d | number of features |

---

# Implementation

## Random Forest Regression

```python
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
```

```python
    def fit(self,X,y):

        n,_ = X.shape

        for _ in range(self.n_estimator):

            indx = np.random.choice(n,n,replace=True)

            new_X = X[indx]

            new_y = y[indx]

            self.roots.append(self.build_tree(new_X,new_y,0))
```

```python
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
```

```python
    def fit(self,X,y):

        n,_ = X.shape

        for _ in range(self.n_estimator):

            indx = np.random.choice(n,n,replace=True)

            new_X = X[indx]

            new_y = y[indx]

            self.roots.append(self.build_tree(new_X,new_y,0))
```

```python
    def predict(self,X):

        n = X.shape[0]

        y_pred = np.zeros(n, dtype=int)

        for i in range(n):

            preds = np.array([self.traverse(X[i],root) for root in self.roots])

            counts = np.bincount(preds)

            y_pred[i] = np.argmax(counts)

        return y_pred
```