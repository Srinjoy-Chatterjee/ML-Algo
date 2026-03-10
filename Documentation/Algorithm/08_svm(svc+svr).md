# Algorithm
Support Vector Machines (SVC, Kernel SVC, Kernel SVR)

Support Vector Machines (SVM) are supervised learning algorithms used for **classification and regression**.

The central idea of SVM is to find a **decision boundary (hyperplane) that maximizes the margin between classes**.

Margin = distance between the separating hyperplane and the closest training samples.

The samples closest to the decision boundary are called **support vectors**, and they determine the final model.

SVM can solve both:

• **Linear problems**  
• **Non-linear problems using kernel functions**

Variants implemented:

| Variant | Task |
|------|------|
SVC | Linear classification |
Kernel SVC | Nonlinear classification |
Kernel SVR | Regression |

---

# Model

## Linear SVM Decision Function

The linear decision function is

```
f(x) = wᵀx + b
```

Prediction

```
y = sign(wᵀx + b)
```

Where

| Symbol | Meaning |
|------|------|
| w | weight vector |
| b | bias |
| x | input feature vector |

The decision boundary is

```
wᵀx + b = 0
```

---

## Margin Geometry

Two margin boundaries exist

```
wᵀx + b = 1
wᵀx + b = -1
```

Margin width

```
Margin = 2 / ||w||
```

Maximizing margin is equivalent to **minimizing ||w||²**.

---

# Math Implementation

## Hard Margin SVM

If the data is perfectly separable:

Optimization objective

```
min (1/2) ||w||²
```

Subject to

```
yᵢ (w·xᵢ + b) ≥ 1
```

This ensures all points are correctly classified.

---

## Soft Margin SVM

Real data is rarely perfectly separable.

Introduce **slack variables**

```
ξᵢ ≥ 0
```

Optimization becomes

```
min (1/2)||w||² + C Σ ξᵢ
```

Subject to

```
yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ
```

Where

| Symbol | Meaning |
|------|------|
| C | regularization parameter |
| ξᵢ | classification error |

Large C → less tolerance to errors  
Small C → wider margin but more errors

---

# Dual Formulation

Instead of solving for **w**, SVM solves the **dual optimization problem**.

Introduce Lagrange multipliers:

```
αᵢ ≥ 0
```

Dual objective

```
max Σ αᵢ − ½ Σ Σ αᵢ αⱼ yᵢ yⱼ xᵢ·xⱼ
```

Subject to

```
0 ≤ αᵢ ≤ C
Σ αᵢ yᵢ = 0
```

---

## Weight Vector from Dual

After solving α:

```
w = Σ αᵢ yᵢ xᵢ
```

Only points with

```
αᵢ > 0
```

contribute to w.

These points are the **support vectors**.

---

# Kernel Trick

For nonlinear data we map inputs to higher dimensional space

```
φ(x)
```

Instead of computing φ explicitly we use

```
K(xᵢ,xⱼ) = φ(xᵢ)ᵀ φ(xⱼ)
```

This is called the **kernel trick**.

Decision function becomes

```
f(x) = Σ αᵢ yᵢ K(xᵢ , x) + b
```

---

# Common Kernels

| Kernel | Formula |
|------|------|
Linear | xᵀz |
Polynomial | (xᵀz + c)^d |
RBF | exp(-γ||x−z||²) |
Sigmoid | tanh(γxᵀz + c) |

---

# SVC (Support Vector Classification)

SVC performs **binary classification**.

Prediction

```
y = sign(wᵀx + b)
```

Loss used:

**Hinge Loss**

```
L = max(0, 1 − y f(x))
```

Optimization minimizes

```
½ ||w||² + C Σ hinge_loss
```

---

# Kernel SVC

Kernel SVC extends SVC to nonlinear boundaries.

Instead of

```
xᵢ · xⱼ
```

use

```
K(xᵢ , xⱼ)
```

Decision function

```
f(x) = Σ αᵢ yᵢ K(xᵢ , x) + b
```

Kernel SVC allows SVM to learn **complex nonlinear boundaries**.

---

# SVR (Support Vector Regression)

SVR adapts SVM for **regression problems**.

Instead of classification margin, SVR uses an **epsilon-insensitive region**.

Errors inside this region are ignored.

```
|y − f(x)| ≤ ε
```

---

## SVR Optimization

Primal objective

```
min (1/2)||w||² + C Σ (ξᵢ + ξᵢ*)
```

Subject to

```
yᵢ − (w·xᵢ + b) ≤ ε + ξᵢ
(w·xᵢ + b) − yᵢ ≤ ε + ξᵢ*
```

Where

| Symbol | Meaning |
|------|------|
| ε | tolerance margin |
| ξᵢ , ξᵢ* | slack variables |

---

## SVR Dual Variables

Two Lagrange multipliers exist

```
αᵢ
αᵢ*
```

Define

```
β = α − α*
```

Prediction becomes

```
f(x) = Σ βᵢ K(xᵢ , x) + b
```

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| epoch | training iterations |
| lr | learning rate |
| C | regularization strength |
| epsilon | SVR margin width |
| Kernel | kernel function |

---

### Derived

| Variable | Meaning |
|------|------|
| alpha | Lagrange multipliers |
| alpha_star | SVR dual variables |
| beta | α − α* |
| w | weight vector |
| b | bias |
| support | support vector indices |
| Q | dual matrix |

---

# Vectorized Form

Dual matrix

```
Y = y yᵀ
K = X Xᵀ
Q = Y ⊙ K
```

Gradient

```
gradient = 1 − Qα
```

Update

```
α = α + lr * gradient
```

Constraint

```
α = clip(α,0,C)
```

Projection

```
α = α − (yᵀα)/(yᵀy) y
```

---

# Algorithm Steps

## SVC

1 Initialize α = 0  
2 Compute dual matrix Q  
3 Perform gradient ascent updates  
4 Enforce constraints  
5 Compute weight vector  
6 Compute bias  
7 Predict using sign(wᵀx+b)

---

## Kernel SVC

1 Compute kernel matrix  
2 Solve dual optimization  
3 Identify support vectors  
4 Predict using kernel decision function

---

## Kernel SVR

1 Initialize α and α*  
2 Compute kernel matrix  
3 Update dual variables  
4 Enforce constraints  
5 Compute β = α − α*  
6 Compute bias  
7 Predict using regression function

---

# Differences

| Model | Task | Kernel Support | Loss |
|------|------|------|------|
SVC | Classification | Optional | Hinge Loss |
Kernel SVC | Classification | Yes | Hinge Loss |
Kernel SVR | Regression | Yes | Epsilon-insensitive loss |

---

# Time Complexity

Kernel matrix

```
O(n²d)
```

Training

```
O(epoch × n²)
```

Prediction

```
O(n_support × d)
```

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |

---

# Code Implementation

## Linear SVC

```python
class SVC:
    def __init__(self, epoch=1000, lr=0.01,C=100):
        self.epoch = epoch        # number of optimization iterations
        self.lr = lr              # learning rate
        self.C = C                # regularization parameter
        
    def fit(self, X, y):

        n = X.shape[0]
        y = y.reshape((-1,1))     # convert labels to column vector
        
        self.alpha = np.zeros((n,1))   # initialize Lagrange multipliers
        
        Y = y @ y.T               # label outer product
        K = X @ X.T               # kernel matrix for linear kernel
        Q = Y * K                 # dual matrix
        
        ones = np.ones((n,1))
        
        for _ in range(self.epoch):

            gradient = ones - Q @ self.alpha   # gradient of dual objective

            self.alpha = self.alpha + self.lr * gradient   # gradient ascent step

            self.alpha = np.clip(self.alpha,0,self.C)      # enforce 0 ≤ α ≤ C

            # enforce constraint y^T α = 0
            self.alpha = self.alpha - (y.T @ self.alpha)/(y.T @ y) * y

        # compute weight vector
        self.w = X.T @ (self.alpha * y)

        # identify support vectors
        support = (self.alpha.flatten() > 1e-5)

        # compute bias
        self.b = np.mean(y[support] - X[support] @ self.w)

    def predict(self, X):

        return np.sign(X @ self.w + self.b)   # classification prediction
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

        self.X_train = X        # store training data
        y = y.reshape((-1,1))
        self.y_train = y

        n = X.shape[0]

        self.alpha = np.zeros((n,1))

        Y = y @ y.T
        K = self.Kernel(X,X,**self.kwargs)   # kernel matrix
        Q = Y * K

        ones = np.ones((n,1))

        for _ in range(self.epoch):

            gradient = ones - Q @ self.alpha

            self.alpha = self.alpha + self.lr * gradient

            self.alpha = np.clip(self.alpha,0,self.C)

            self.alpha = self.alpha - (y.T @ self.alpha)/(y.T @ y) * y

        # identify support vectors
        support = (self.alpha.flatten() > 1e-5) & (self.alpha.flatten() < self.C - 1e-5)

        # compute bias using support vectors
        self.b = np.mean(y[support] - (K @ (self.alpha * y))[support])

    def predict(self, X):

        K = self.Kernel(self.X_train,X)

        return np.sign((self.alpha * self.y_train) @ K+ self.b)
```

---

## Kernel SVR

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
            
            beta = self.alpha - self.alpha_star   # dual variable
            f = K @ beta                          # prediction
            
            # gradients of SVR dual objective
            grad_alpha = y - f - self.epsilon
            grad_alpha_star = -y + f - self.epsilon
            
            # gradient ascent updates
            self.alpha += self.lr * grad_alpha
            self.alpha_star += self.lr * grad_alpha_star
            
            # enforce box constraints
            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)
            
            # enforce equality constraint
            beta = self.alpha - self.alpha_star
            correction = np.sum(beta) / n
            
            self.alpha -= correction / 2
            self.alpha_star += correction / 2
            
            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)
        
        # compute final beta
        self.beta = self.alpha - self.alpha_star
        
        # compute bias
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