# Algorithm
Regularized Linear Regression (Lasso, Ridge, ElasticNet)

Regularization is used to **reduce overfitting** by penalizing large model weights.

The base model remains

ŷ = Xθ

Regularization adds a penalty term to the loss function.

---

# Model

ŷ = Xθ

Regularization modifies the loss function but does not change the prediction formula.

---

# Loss Function

### Lasso Regression (L1)

L(θ) = (1/n) Σ (y − Xθ)² + α|θ|

L1 penalty encourages **sparse weights**, meaning some weights become zero.

---

### Ridge Regression (L2)

L(θ) = (1/n) Σ (y − Xθ)² + α||θ||²

L2 penalty **shrinks weights** but rarely makes them zero.

---

### ElasticNet

L(θ) = (1/n) Σ (y − Xθ)² + λ₁|θ| + λ₂||θ||²

Combines both L1 and L2 regularization.

---

# Gradient

### Lasso

∇L = (2/n)Xᵀ(Xθ − y) + α sign(θ)

---

### Ridge

∇L = (2/n)Xᵀ(Xθ − y) + 2αθ

---

### ElasticNet

∇L = (2/n)Xᵀ(Xθ − y) + λ₁ sign(θ) + 2λ₂θ

---

# Gradient Descent Update

θ = θ − η ∇L

Example (Ridge)

θ = θ − η[(2/n)Xᵀ(Xθ − y) + 2αθ]

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| lr | learning rate |
| epoch | number of iterations |
| alpha | regularization strength |
| l1_lambda | L1 penalty |
| l2_lambda | L2 penalty |

---

# Vectorized Form

```
prediction = X @ m
error = prediction - y
gradient = (2/n) * X.T @ error + regularization
```

---

# Algorithm Steps

1 Add bias column  
2 Initialize parameters  
3 Compute prediction  
4 Compute gradient including penalty  
5 Update parameters  

---

# Time Complexity

O(n × d)

---

# Implementation

```python
class LassoRegression:
    def __init__(self,lr = 0.01, epoch = 1000, alpha = 1):
            self.lr = lr
            self.epoch = epoch
            self.alpha = alpha

    def fit(self,X,y):
        n = X.shape[0]

        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initialize m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape((-1,1))

        for _ in range(self.epoch):
            prediction = X @ self.m
            error = prediction -y
            gradiant = (2/n) * (X.T @ error) + self.alpha * np.sign(self.m)
            self.m = self.m - self.lr * gradiant
        
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
   
```

```python
class RidgeRegression:
    def __init__(self,lr = 0.01,epoch=1000,alpha=1):
        self.lr = lr
        self.epoch = epoch
        self.alpha = alpha

    def fit(self,X,y):
        n = X.shape[0]
        
        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initializing m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape(-1,1)

        for _ in range(self.epoch):
            prediction = X @ self.m
            error = prediction - y
            gradiant = 2/n * X.T @ error + 2 * self.alpha * self.m
            self.m = self.m - self.lr * gradiant
            
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m
   
```

```python
class ElasticNetRegression:
    def __init__(self,lr=0.01,epochs=1000,l1_lambda=1,l2_lambda=1):
        self.lr = lr
        self.epochs = epochs
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda 
    
    def fit(self,X,y):
        n = X.shape[0]

        #bias
        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        #initialize m
        self.m = np.zeros((X.shape[1],1))

        y = y.reshape(-1,1)

        for _ in range(self.epochs):
            prediction = X @ self.m
            error = prediction - y
            gradiant = (2/n) * (X.T @ error) + self.l1_lambda * np.sign(self.m) + 2 * self.l2_lambda * self.m
            self.m = self.m - self.lr * gradiant
          
    def predict(self,X):
        #bias
        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        return X @ self.m        
    
```