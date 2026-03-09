# Algorithm
K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a **non-parametric supervised learning algorithm** used for classification and regression.

The main idea is simple:

A data point is classified based on the **labels of its nearest neighbors in the feature space**.

For classification:
The class with the **majority vote among the k nearest neighbors** is assigned.

For regression:
The prediction is the **average value of the k nearest neighbors**.

---

# Model

KNN does not learn explicit model parameters.

Instead, the entire training dataset is stored.

Prediction is based on **distance between points**.

For two samples

x and z

distance is typically computed using **Euclidean distance**

d(x,z) = √ Σ (xᵢ − zᵢ)²

Where

| Symbol | Meaning |
|------|------|
| x | query sample |
| z | training sample |
| d(x,z) | distance between samples |

---

# Loss Function

KNN does not optimize a loss function during training.

Training simply stores the dataset.

The prediction rule minimizes **local distance in feature space**.

---

# Gradient

KNN is a **lazy learning algorithm** and does not use gradient descent.

No parameters are optimized.

---

# Gradient Descent Update

Not applicable.

No parameter updates occur during training.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| k | number of nearest neighbors |

### Derived During Prediction

| Variable | Meaning |
|------|------|
| distances | distance between query and training points |
| nearest_neighbors | indices of k closest points |
| nearest_labels | labels of nearest neighbors |

---

# Vectorized Form

Distances are computed in a fully vectorized way.

Distance matrix

```
distances = np.linalg.norm(self.X_train[None,:,:] - X[:,None,:], axis=2)
```

Shape

```
(n_test , n_train)
```

Each row contains distances from a test sample to all training samples.

---

# Algorithm Steps

Training

1 Store training data  
2 Store labels  

Prediction

1 Compute distance between test samples and training samples  
2 Select k smallest distances  
3 Retrieve labels of nearest neighbors  
4 Choose the most frequent label  

---

# Time Complexity

Training

O(1)

Prediction

O(n_train × d)

Where

| Symbol | Meaning |
|------|------|
| n_train | number of training samples |
| d | number of features |

Sorting neighbors adds

O(n_train log n_train)

---

# Implementation

```python
class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):

        distances = np.linalg.norm(
            self.X_train[None,:,:] - X[:,None,:],
            axis=2
        )

        nearest_neighbors = np.argsort(distances,axis=1)[:,:self.k]

        nearest_labels = self.y_train[nearest_neighbors]

        return np.array([
            np.bincount(labels).argmax()
            for labels in nearest_labels
        ])
```