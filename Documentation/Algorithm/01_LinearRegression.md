# Algorithm
Linear Regression

Linear Regression is a supervised learning algorithm used to predict a **continuous target variable** from input features.

It assumes that the relationship between the input features and the target variable is **linear**.

Example linear relationship

y = w₁x₁ + w₂x₂ + ... + w_d x_d + b

The goal of the algorithm is to learn the parameters (weights and bias) that minimize the prediction error between the predicted value and the true value.

Linear Regression is commonly used in:

• house price prediction  
• sales forecasting  
• trend estimation  

---

# Model

The linear regression model can be written as

ŷ = Xθ

Where

| Symbol | Meaning |
|------|------|
| X | feature matrix (n × d) |
| θ | parameter vector (d × 1) |
| ŷ | predicted output |
| y | true output |
| n | number of samples |
| d | number of features |

Expanded form

ŷ = w₁x₁ + w₂x₂ + ... + w_d x_d + b

Where

| Symbol | Meaning |
|------|------|
| w | feature weights |
| b | bias |

### Bias Handling

Instead of treating bias separately, it is included inside the parameter vector.

Add a column of ones to X:

X' = [X 1]

Then

ŷ = X'θ

Where

θ = [w₁, w₂, ... , w_d , b]

This allows the model to be written using a single matrix multiplication.

---

# Math Implementation

### Loss Function

Linear regression minimizes **Mean Squared Error (MSE)**.

L(θ) = (1/n) Σ (ŷ − y)²

Where

| Symbol | Meaning |
|------|------|
| L(θ) | loss function |
| ŷ | predicted value |
| y | true value |

Vector form

L(θ) = (1/n)(Xθ − y)ᵀ(Xθ − y)

---

### Gradient

Let

e = Xθ − y

Then

∇L(θ) = (2/n) Xᵀ(Xθ − y)

Where

| Symbol | Meaning |
|------|------|
| ∇L | gradient of loss |
| Xᵀ | transpose of X |
|e   | error |

This gradient indicates how the parameters should change to minimize the loss.

---

### Gradient Descent Update

Parameters are updated iteratively using gradient descent.

θ = θ − η ∇L(θ)

Substituting the gradient

θ = θ − η (2/n) Xᵀ(Xθ − y)

Where

| Symbol | Meaning |
|------|------|
| η | learning rate |

Learning rate controls how large each parameter update step is.

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
| m | parameter vector θ |
| prediction | predicted values |
| error | difference between prediction and true value |
| gradient | derivative of loss function |

---

# Loop / Vectorized Form

The implementation uses **vectorized matrix operations** instead of loops over samples.

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

Vectorization allows the computation to be done efficiently using linear algebra operations.

---

# Algorithm Steps

1. Add bias column to feature matrix  

X ← [X 1]

2. Initialize parameter vector

θ = 0

3. Repeat for each iteration

Compute prediction

ŷ = Xθ

Compute error

e = ŷ − y

Compute gradient

∇L = (2/n) Xᵀ e

Update parameters

θ = θ − η ∇L

4. Stop after the specified number of epochs.

---

# Time Complexity

Training complexity

O(n × d)

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |

Prediction complexity

O(n × d)

The main cost comes from matrix multiplication.

---

# Code Implementation

```python
class LinearRegression:

    def __init__(self,lr = 0.01, epoch = 1000):
            self.lr = lr              # learning rate for gradient descent
            self.epoch = epoch        # number of training iterations

    def fit(self,X,y):
        n = X.shape[0]               # number of samples

        ones = np.ones((n,1))        # create bias column
        X = np.hstack((X,ones))      # add bias to feature matrix

        self.m = np.zeros((X.shape[1],1))   # initialize parameter vector θ

        y = y.reshape((-1,1))        # reshape target to column vector

        for _ in range(self.epoch):

            prediction = X @ self.m        # compute predictions (Xθ)

            error = prediction - y         # compute residual error

            gradiant = (2/n) * (X.T @ error)   # compute gradient of MSE

            self.m = self.m - self.lr * gradiant   # update parameters using gradient descent
        
    def predict(self,X):

        ones = np.ones((X.shape[0],1))     # add bias column for prediction
        X = np.hstack((X,ones))

        return X @ self.m                  # return predicted values
```