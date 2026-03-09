# Algorithm
K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a **non-parametric supervised learning algorithm** used for both classification and regression tasks.

Unlike many other algorithms, KNN **does not learn explicit model parameters during training**.  
Instead, it stores the entire training dataset and makes predictions by comparing new samples with the stored data.

The key idea is:

A sample is predicted based on the **labels of its closest neighbors in feature space**.

### Classification

The predicted class is the **majority label among the k nearest neighbors**.

### Regression

The predicted value is the **average value of the k nearest neighbors**.

KNN is called a **lazy learning algorithm** because computation happens mainly during prediction rather than training.

---

# Model

KNN does not learn a mathematical model.

Instead, predictions are made based on **distance between points in feature space**.

Given two samples

x and z

the most common distance metric is **Euclidean distance**.

d(x,z) = √ Σ (xᵢ − zᵢ)²

Where

| Symbol | Meaning |
|------|------|
| x | query sample |
| z | training sample |
| d(x,z) | distance between two samples |
| xᵢ | feature i of sample x |

The algorithm finds the **k closest training samples** to the query point.

---

# Math Implementation

### Distance Computation

For a query sample x and training sample z

d(x,z) = √ Σ (xᵢ − zᵢ)²

Vector form

d(x,z) = √((x − z)ᵀ(x − z))

This distance measures similarity in the feature space.

---

### Classification Rule

Let

Nₖ(x) be the set of k nearest neighbors of sample x.

Prediction

ŷ = argmax Σ I(yᵢ = c)

Where

| Symbol | Meaning |
|------|------|
| I | indicator function |
| c | class label |
| yᵢ | label of neighbor |

The class with the **largest count among neighbors** is selected.

---

### Regression Rule

Prediction

ŷ = (1/k) Σ yᵢ

Where

| Symbol | Meaning |
|------|------|
| yᵢ | neighbor values |

The predicted value is the **average of neighbor values**.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| k | number of nearest neighbors |

---

### Stored During Training

| Variable | Meaning |
|------|------|
| X_train | training feature matrix |
| y_train | training labels |

---

### Derived During Prediction

| Variable | Meaning |
|------|------|
| distances | distance matrix between test and training samples |
| nearest_neighbors | indices of k closest training samples |
| nearest_labels | labels of the nearest neighbors |

---

# Loop / Vectorized Form

Distance matrix computation uses **NumPy broadcasting**.

```
distances = np.linalg.norm(self.X_train[None,:,:] - X[:,None,:], axis=2)
```

Shapes involved

```
X_train shape → (n_train , d)
X shape       → (n_test , d)
```

After broadcasting

```
X[:,None,:]           → (n_test , 1 , d)
self.X_train[None,:,:] → (1 , n_train , d)
```

Subtraction result

```
(n_test , n_train , d)
```

Then Euclidean norm across feature dimension

```
axis = 2
```

Final distance matrix shape

```
(n_test , n_train)
```

Each row contains distances from **one test sample to all training samples**.

---

# Algorithm Steps

### Training

1 Store training feature matrix

X_train = X

2 Store training labels

y_train = y

No parameter learning occurs.

---

### Prediction

1 Compute distance between test samples and training samples

D[i,j] = distance(test_i , train_j)

2 Sort distances

3 Select indices of the **k smallest distances**

4 Retrieve labels of those neighbors

5 Compute majority vote

Predicted class = most frequent label

---

# Time Complexity

### Training Complexity

O(1)

Training only stores the dataset.

---

### Prediction Complexity

Distance computation

O(n_test × n_train × d)

Sorting neighbors

O(n_train log n_train)

Total prediction complexity

O(n_train × d)

for each test sample.

---

# Code Implementation

```python
class KNN:
    def __init__(self,k=3):
        self.k = k                     # number of nearest neighbors

    def fit(self,X,y):
        self.X_train = X               # store training features
        self.y_train = y               # store training labels

    def predict(self,X):

        # compute pairwise Euclidean distances between
        # test samples and training samples
        distances = np.linalg.norm(
            self.X_train[None,:,:] - X[:,None,:],
            axis=2
        )

        # find indices of k nearest neighbors
        nearest_neighbors = np.argsort(distances,axis=1)[:,:self.k]

        # retrieve labels of nearest neighbors
        nearest_labels = self.y_train[nearest_neighbors]

        # majority voting for classification
        return np.array([
            np.bincount(labels).argmax()
            for labels in nearest_labels
        ])
```