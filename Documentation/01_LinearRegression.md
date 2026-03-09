# Algorithm
Linear Regression

Linear Regression is a supervised learning algorithm used to predict a **continuous target variable** from input features.  
It assumes a **linear relationship** between features and the target.

Example

y = w1x1 + w2x2 + ... + wdxd + b

The algorithm learns parameters that minimize prediction error.

---

# Model

The linear model is

ŷ = Xθ

Where

| Symbol | Meaning |
|------|------|
| X | feature matrix (n × d) |
| θ | parameter vector |
| ŷ | predicted values |
| y | true target |
| n | number of samples |
| d | number of features |

Bias is handled by adding a column of ones.

X' = [X 1]

Then

ŷ = X'θ

---

# Loss Function

Linear regression minimizes **Mean Squared Error (MSE)**.

L(θ) = (1/n) Σ (ŷ − y)²

This measures the average squared difference between predictions and true values.

Vector form

L(θ) = (1/n)(Xθ − y)ᵀ(Xθ − y)

---

# Gradient

Let

e = Xθ − y

Then the gradient of the loss with respect to θ is

∇L = (2/n) Xᵀ(Xθ − y)

This gradient tells how parameters should change to reduce error.

---

# Gradient Descent Update

Parameters are updated iteratively.

θ = θ − η ∇L

Substituting the gradient

θ = θ − η (2/n) Xᵀ(Xθ − y)

Where

| Symbol | Meaning |
|------|------|
| η | learning rate |

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| lr | learning rate |
| epoch | number of training iterations |

### Derived During Training

| Variable | Meaning |
|------|------|
| m | parameter vector |
| prediction | predicted values |
| error | difference between prediction and true value |
| gradient | derivative of loss |

---

# Vectorized Form

Prediction

```
prediction = X @ m
```

Error

```
error = prediction - y
```

Gradient

```
gradient = (2/n) * (X.T @ error)
```

Vectorization removes loops and allows fast matrix operations.

---

# Algorithm Steps

1 Add bias column to feature matrix  
2 Initialize parameters θ = 0  

Repeat for each epoch

prediction = Xθ  
error = prediction − y  
gradient = (2/n) Xᵀ error  
θ = θ − lr × gradient  

---

# Time Complexity

The main cost is matrix multiplication.

O(n × d)

Where

n = number of samples  
d = number of features

---

# Implementation

```python
class LinearRegression:

    def __init__(self,lr = 0.01, epoch = 1000):
            self.lr = lr
            self.epoch = epoch

    def fit(self,X,y):
        n = X.shape[0]

        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        self.m = np.zeros((X.shape[1],1))

        y = y.reshape((-1,1))

        for _ in range(self.epoch):
            prediction = X @ self.m
            error = prediction -y
            gradiant = (2/n) * (X.T @ error)
            self.m = self.m - self.lr * gradiant
        
    def predict(self,X):

        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
```