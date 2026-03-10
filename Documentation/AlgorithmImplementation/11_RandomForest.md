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
class RandomForestRegression:
    def __init__(self,n_estimator,depth=5,min=20):
        self.n_estimator = n_estimator     # number of trees
        self.depth = depth                 # max tree depth
        self.min = min                     # minimum samples to split
        self.roots = []                    # list of tree roots

    def find_best_split(self,X,y):

        n_samples,n_features = X.shape

        # random subset of features
        k_features = np.random.choice(n_features,int(np.sqrt(n_features)),replace=False)

        min_sse = float('inf')

        best_thresold = None
        best_feature_indx = None
```

```python
    def fit(self,X,y):

        n,_ = X.shape

        for _ in range(self.n_estimator):

            # bootstrap sampling
            indx = np.random.choice(n,n,replace=True)

            new_X = X[indx]

            new_y = y[indx]

            # build tree on sampled dataset
            self.roots.append(self.build_tree(new_X,new_y,0))
```

```python
    def predict(self,X):

        n = X.shape[0]

        y_pred = np.zeros(n)

        for i in range(n):

            pred = []

            # collect predictions from all trees
            for root in self.roots:

                pred.append(self.traverse(X[i],root))

            # average prediction
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

        self.n_estimator = n_estimator     # number of trees
        self.depth = depth                 # max depth
        self.min = min                     # minimum samples
        self.roots = []                    # list of tree roots
```

```python
    def fit(self,X,y):

        n,_ = X.shape

        for _ in range(self.n_estimator):

            # bootstrap sampling
            indx = np.random.choice(n,n,replace=True)

            new_X = X[indx]

            new_y = y[indx]

            # train decision tree
            self.roots.append(self.build_tree(new_X,new_y,0))
```

```python
    def predict(self,X):

        n = X.shape[0]

        y_pred = np.zeros(n, dtype=int)

        for i in range(n):

            # predictions from all trees
            preds = np.array([self.traverse(X[i],root) for root in self.roots])

            # majority voting
            counts = np.bincount(preds)

            y_pred[i] = np.argmax(counts)

        return y_pred
```