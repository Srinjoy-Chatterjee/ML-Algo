# Algorithm
Support Vector Machines (SVC, Kernel SVC, Kernel SVR)

Support Vector Machines are supervised learning algorithms used for **classification and regression**.

The main idea is to find a **hyperplane that maximizes the margin between classes**.

Margin = distance between the separating hyperplane and the nearest training points.

Those nearest points are called **support vectors**.

Two main variants:

SVC → classification  
SVR → regression

Kernel methods allow SVM to solve **non-linear problems**.

---

# Model

For classification the model is

f(x) = wᵀx + b

Prediction

y = sign(wᵀx + b)

Where

| Symbol | Meaning |
|------|------|
| w | weight vector |
| b | bias |
| x | feature vector |

In the **dual formulation**

w = Σ αᵢ yᵢ xᵢ

Where

| Symbol | Meaning |
|------|------|
| α | Lagrange multipliers |
| yᵢ | class label |
| xᵢ | support vectors |

For kernel SVM

f(x) = Σ αᵢ yᵢ K(xᵢ , x) + b

Where

K(xᵢ , x) is the kernel function.

---

# Loss Function

The optimization objective of SVM is

min (1/2) ||w||² + C Σ ξᵢ

Subject to

yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ

Where

| Symbol | Meaning |
|------|------|
| ξᵢ | slack variable |
| C | regularization parameter |

Dual objective maximized in implementation

max Σ αᵢ − (1/2) Σ αᵢ αⱼ yᵢ yⱼ xᵢ·xⱼ

---

# Gradient

For the dual objective

gradient = 1 − Qα

Where

Q = yᵢ yⱼ xᵢ·xⱼ

In matrix form

Q = (y yᵀ) ⊙ (X Xᵀ)

Where

⊙ denotes element-wise multiplication.

---

# Gradient Descent Update

The dual parameters α are updated using gradient ascent.

α = α + η (1 − Qα)

Constraints must also be enforced

0 ≤ α ≤ C

yᵀα = 0

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| epoch | number of iterations |
| lr | learning rate |
| C | regularization strength |
| epsilon | SVR margin width |
| Kernel | kernel function |

### Derived

| Variable | Meaning |
|------|------|
| alpha | Lagrange multipliers |
| Q | dual matrix |
| w | weight vector |
| b | bias |
| support | support vector indices |

---

# Vectorized Form

Dual matrix

```
Y = y @ y.T
K = X @ X.T
Q = Y * K
```

Gradient

```
gradient = ones - Q @ alpha
```

Parameter update

```
alpha = alpha + lr * gradient
```

Projection constraint

```
alpha = alpha - (y.T @ alpha)/(y.T @ y) * y
```

---

# Algorithm Steps

Training

1 Initialize α = 0  
2 Compute Q matrix  
3 Perform gradient ascent updates  
4 Enforce constraints on α  
5 Compute weight vector  
6 Compute bias from support vectors  

Prediction

1 Compute decision function  
2 Apply sign function for classification  

---

# Time Complexity

Kernel matrix computation

O(n²d)

Training

O(epoch × n²)

Prediction

O(n_support × d)

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |

---

# Implementation

## Linear SVC

```python
class SVC:
    def __init__(self, epoch=1000, lr=0.01,C=100):
        self.epoch = epoch
        self.lr = lr
        self.C = C
        
    def fit(self, X, y):

        n = X.shape[0]
        y = y.reshape((-1,1))
        
        self.alpha = np.zeros((n,1))
        
        Y = y @ y.T
        K = X @ X.T
        Q = Y * K
        
        ones = np.ones((n,1))
        
        for _ in range(self.epoch):

            gradient = ones - Q @ self.alpha

            self.alpha = self.alpha + self.lr * gradient

            self.alpha = np.clip(self.alpha,0,self.C)

            self.alpha = self.alpha - (y.T @ self.alpha)/(y.T @ y) * y

        self.w = X.T @ (self.alpha * y)

        support = (self.alpha.flatten() > 1e-5)

        self.b = np.mean(y[support] - X[support] @ self.w)

    def predict(self, X):

        return np.sign(X @ self.w + self.b)
```

---

## Kernel SVC

```python
class KernelSVC:
    def __init__(self, epoch=1000, lr=0.01,C=100,Kernel = Kernel.LINEAR,**kwargs):
        self.epoch = epoch
        self.lr = lr
        self.C = C
        self.Kernel = Kernel
        self.kwargs = kwargs
        
    def fit(self, X, y):

        self.X_train = X
        y = y.reshape((-1,1))
        self.y_train = y

        n = X.shape[0]

        self.alpha = np.zeros((n,1))

        Y = y @ y.T
        K = self.Kernel(X,X,**self.kwargs)
        Q = Y * K

        ones = np.ones((n,1))

        for _ in range(self.epoch):

            gradient = ones - Q @ self.alpha

            self.alpha = self.alpha + self.lr * gradient

            self.alpha = np.clip(self.alpha,0,self.C)

            self.alpha = self.alpha - (y.T @ self.alpha)/(y.T @ y) * y

        support = (self.alpha.flatten() > 1e-5) & (self.alpha.flatten() < self.C - 1e-5)

        self.b = np.mean(y[support] - (K @ (self.alpha * y))[support])

    def predict(self, X):

        K = self.Kernel(self.X_train,X)

        return np.sign((self.alpha * self.y_train) @ K+ self.b)
```

---

## Kernel SVR

SVR predicts continuous values using an **epsilon-insensitive loss**.

Only errors larger than ε contribute to the loss.

```
|y - f(x)| ≤ ε
```

```python
class KernelSVR:
    def __init__(self, epoch=1000, lr=0.001, C=100, epsilon=0.1, 
                 Kernel=Kernel.LINEAR, **kwargs):
        
        self.epoch = epoch
        self.lr = lr
        self.C = C
        self.epsilon = epsilon
        self.Kernel = Kernel
        self.kwargs = kwargs

    def fit(self, X, y):
        self.X_train = X
        y = y.reshape(-1,1)
        self.y_train = y
        
        n = X.shape[0]
        
        # Two alpha vectors
        self.alpha = np.zeros((n,1))
        self.alpha_star = np.zeros((n,1))
        
        # Kernel matrix
        K = self.Kernel(X, X, **self.kwargs)
        
        for _ in range(self.epoch):
            
            beta = self.alpha - self.alpha_star   # (n,1)
            f = K @ beta                          # (n,1)
            
            # Gradients from SVR dual
            grad_alpha = y - f - self.epsilon
            grad_alpha_star = -y + f - self.epsilon
            
            # Gradient ascent
            self.alpha += self.lr * grad_alpha
            self.alpha_star += self.lr * grad_alpha_star
            
            # Box constraints
            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)
            
            # Enforce equality constraint: sum(alpha - alpha*) = 0
            beta = self.alpha - self.alpha_star
            correction = np.sum(beta) / n
            
            self.alpha -= correction / 2
            self.alpha_star += correction / 2
            
            # Clip again
            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)
        
        # Final beta
        self.beta = self.alpha - self.alpha_star
        
        # -------- Compute bias b --------
        
        f_train = K @ self.beta
        
        idx1 = np.where((self.alpha > 1e-5) & 
                        (self.alpha < self.C-1e-5))[0]
        
        idx2 = np.where((self.alpha_star > 1e-5) & 
                        (self.alpha_star < self.C-1e-5))[0]
        
        b_vals = []
        
        for i in idx1:
            b_vals.append(y[i] - self.epsilon - f_train[i])
        
        for i in idx2:
            b_vals.append(y[i] + self.epsilon - f_train[i])
        
        self.b = np.mean(b_vals) if b_vals else 0

    def predict(self, X):
        K = self.Kernel(self.X_train, X, **self.kwargs)
        return (K.T @ self.beta + self.b)
 
```

