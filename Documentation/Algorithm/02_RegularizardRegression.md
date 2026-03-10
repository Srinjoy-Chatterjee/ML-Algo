# Algorithm
Regularized Linear Regression (Lasso, Ridge, ElasticNet)

Regularization is a technique used to **reduce overfitting** in machine learning models by penalizing large parameter values.

**NOTE:** Regularization can be performend of any model which has loss function

In standard Linear Regression, the model tries to minimize prediction error.  
However, if the model becomes too complex, it may fit the training data very well but perform poorly on unseen data.

Regularization addresses this by adding a **penalty term to the loss function** that discourages large weights.

Three common types of regularization are:

• **Lasso Regression (L1 Regularization)**  
• **Ridge Regression (L2 Regularization)**  
• **ElasticNet Regression (Combination of L1 and L2)**

---

# Model

The prediction model remains identical to Linear Regression.

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

Bias is handled by adding a column of ones to the feature matrix.

X' = [X 1]

Then

ŷ = X'θ

---

# Math Implementation

## Lasso Regression (L1 Regularization)

### Loss Function

L(θ) = (1/n) Σ (y − Xθ)² + α‖θ‖₁

Where

| Symbol | Meaning |
|------|------|
| α | regularization strength |
| ‖θ‖₁ | L1 norm = Σ |θᵢ| |

The L1 penalty forces some weights to become **exactly zero**, effectively performing **feature selection**.

---

### Gradient

∇L = (2/n)Xᵀ(Xθ − y) + α sign(θ)

Where

| Symbol | Meaning |
|------|------|
| sign(θ) | element-wise sign function |

sign(θᵢ) =

1 if θᵢ > 0  
−1 if θᵢ < 0  
0 if θᵢ = 0

---

### Gradient Descent Update

θ = θ − η[(2/n)Xᵀ(Xθ − y) + α sign(θ)]

---

## Ridge Regression (L2 Regularization)

### Loss Function

L(θ) = (1/n) Σ (y − Xθ)² + α‖θ‖²

Where

| Symbol | Meaning |
|------|------|
| α | regularization parameter |
| ‖θ‖² | L2 norm = Σ θᵢ² |

The L2 penalty **shrinks parameter values toward zero**, but rarely makes them exactly zero.

---

### Gradient

∇L = (2/n)Xᵀ(Xθ − y) + 2αθ

---

### Gradient Descent Update

θ = θ − η[(2/n)Xᵀ(Xθ − y) + 2αθ]

---

## ElasticNet Regression

ElasticNet combines both L1 and L2 penalties.

### Loss Function

L(θ) = (1/n) Σ (y − Xθ)² + λ₁‖θ‖₁ + λ₂‖θ‖²

Where

| Symbol | Meaning |
|------|------|
| λ₁ | L1 regularization strength |
| λ₂ | L2 regularization strength |

---

### Gradient

∇L = (2/n)Xᵀ(Xθ − y) + λ₁ sign(θ) + 2λ₂θ

---

### Gradient Descent Update

θ = θ − η[(2/n)Xᵀ(Xθ − y) + λ₁ sign(θ) + 2λ₂θ]

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| lr | learning rate |
| epoch | number of training iterations |
| alpha | regularization strength (L1 or L2) |
| l1_lambda | L1 regularization parameter |
| l2_lambda | L2 regularization parameter |

### Derived During Training

| Variable | Meaning |
|------|------|
| m | parameter vector θ |
| prediction | predicted output |
| error | residual error (prediction − y) |
| gradient | derivative of loss function |

---

# Loop / Vectorized Form

Prediction

```
prediction = X @ m
```

Error

```
error = prediction - y
```

Gradient (Lasso)

```
gradient = (2/n) * X.T @ error + alpha * sign(m)
```

Gradient (Ridge)

```
gradient = (2/n) * X.T @ error + 2 * alpha * m
```

Gradient (ElasticNet)

```
gradient = (2/n) * X.T @ error + l1_lambda * sign(m) + 2 * l2_lambda * m
```

These operations are fully **vectorized**, allowing efficient computation using matrix multiplication.

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

Lasso

∇L = (2/n)Xᵀe + α sign(θ)

Ridge

∇L = (2/n)Xᵀe + 2αθ

ElasticNet

∇L = (2/n)Xᵀe + λ₁ sign(θ) + 2λ₂θ

Update parameters

θ = θ − η∇L

4. Stop after reaching the specified number of iterations.

---

# Time Complexity

Training complexity

O(n × d)

Prediction complexity

O(n × d)

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |

The dominant operation is matrix multiplication.

---

# Code Implementation

```python
class LassoRegression:
    def __init__(self,lr = 0.01, epoch = 1000, alpha = 1):
            self.lr = lr                # learning rate
            self.epoch = epoch          # number of gradient descent iterations
            self.alpha = alpha          # L1 regularization strength

    def fit(self,X,y):
        n = X.shape[0]

        # add bias column
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        # initialize parameter vector
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape((-1,1))

        for _ in range(self.epoch):

            prediction = X @ self.m             # compute prediction

            error = prediction -y               # compute residual error

            # gradient with L1 penalty
            gradiant = (2/n) * (X.T @ error) + self.alpha * np.sign(self.m)

            # parameter update
            self.m = self.m - self.lr * gradiant
        
    def predict(self,X):

        # add bias column for prediction
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
```

```python
class RidgeRegression:
    def __init__(self,lr = 0.01,epoch=1000,alpha=1):
        self.lr = lr                  # learning rate
        self.epoch = epoch            # number of iterations
        self.alpha = alpha            # L2 regularization strength

    def fit(self,X,y):
        n = X.shape[0]
        
        # add bias column
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        # initialize parameters
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape(-1,1)

        for _ in range(self.epoch):

            prediction = X @ self.m      # compute prediction

            error = prediction - y       # compute residual error

            # gradient with L2 regularization
            gradiant = 2/n * X.T @ error + 2 * self.alpha * self.m

            # update parameters
            self.m = self.m - self.lr * gradiant
            
    def predict(self,X):

        # add bias column
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
```

```python
class ElasticNetRegression:
    def __init__(self,lr=0.01,epochs=1000,l1_lambda=1,l2_lambda=1):
        self.lr = lr                      # learning rate
        self.epochs = epochs              # number of iterations
        self.l1_lambda = l1_lambda        # L1 regularization parameter
        self.l2_lambda = l2_lambda        # L2 regularization parameter
    
    def fit(self,X,y):
        n = X.shape[0]

        # add bias column
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        # initialize parameters
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape(-1,1)

        for _ in range(self.epochs):

            prediction = X @ self.m

            error = prediction - y

            # gradient with combined L1 + L2 penalties
            gradiant = (2/n) * (X.T @ error) + self.l1_lambda * np.sign(self.m) + 2 * self.l2_lambda * self.m

            # parameter update
            self.m = self.m - self.lr * gradiant
          
    def predict(self,X):

        # add bias column
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m        
```