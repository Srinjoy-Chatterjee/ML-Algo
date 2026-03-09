# Algorithm
Support Vector Machines (SVC, Kernel SVC, Kernel SVR)

Support Vector Machines (SVM) are supervised learning algorithms used for **classification and regression**.

The key idea of SVM is to find a **hyperplane that separates classes with the maximum margin**.

Margin = distance between the separating hyperplane and the closest training points.

The training samples that lie closest to the hyperplane are called **support vectors**, and they determine the final model.

SVM can solve **linear and non-linear problems**.

Linear SVM → uses linear hyperplane  
Kernel SVM → uses kernel functions to map data into higher-dimensional space

Two variants implemented here:

• **SVC** → classification  
• **SVR** → regression using epsilon-insensitive loss  

---

# Model

## Linear SVM

The decision function is

f(x) = wᵀx + b

Prediction

y = sign(wᵀx + b)

Where

| Symbol | Meaning |
|------|------|
| w | weight vector |
| b | bias |
| x | feature vector |

---

## Dual Representation

Instead of directly solving for **w**, SVM solves a **dual optimization problem**.

w = Σ αᵢ yᵢ xᵢ

Where

| Symbol | Meaning |
|------|------|
| αᵢ | Lagrange multiplier |
| yᵢ | class label |
| xᵢ | training sample |

Only samples with **αᵢ > 0** contribute to the solution → these are the **support vectors**.

---

## Kernel SVM

Kernel functions allow SVM to learn **non-linear decision boundaries**.

Decision function

f(x) = Σ αᵢ yᵢ K(xᵢ , x) + b

Where

| Symbol | Meaning |
|------|------|
| K(xᵢ , x) | kernel function |

Common kernels

• Linear  
• Polynomial  
• RBF (Gaussian)  

---

# Math Implementation

## Primal Optimization Problem

SVM solves

min (1/2) ||w||² + C Σ ξᵢ

subject to

yᵢ (w·xᵢ + b) ≥ 1 − ξᵢ

Where

| Symbol | Meaning |
|------|------|
| ξᵢ | slack variable |
| C | regularization parameter |

The first term maximizes the **margin**, while the second penalizes classification errors.

---

## Dual Optimization Problem

The dual objective is

max Σ αᵢ − (1/2) Σ Σ αᵢ αⱼ yᵢ yⱼ xᵢ·xⱼ

subject to

0 ≤ αᵢ ≤ C  
Σ αᵢ yᵢ = 0

---

## Dual Matrix Form

Define

Q = (y yᵀ) ⊙ (X Xᵀ)

Where

| Symbol | Meaning |
|------|------|
| ⊙ | element-wise multiplication |

The dual objective becomes

max 1ᵀα − (1/2) αᵀ Q α

---

## Gradient of Dual Objective

∇L = 1 − Qα

This gradient is used in **gradient ascent**.

---

## Gradient Ascent Update

α = α + η (1 − Qα)

---

## Constraint Enforcement

After updating α, two constraints must be satisfied.

### Box Constraint

0 ≤ α ≤ C

Implemented with

```
clip(alpha, 0, C)
```

---

### Equality Constraint

yᵀα = 0

To enforce this constraint the projection step is used

α ← α − proj_y(α)

Projection formula

proj_y(α) = (yᵀα / yᵀy) y

Thus

α = α − (yᵀα)/(yᵀy) y

This removes the component of α parallel to y.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| epoch | number of training iterations |
| lr | learning rate |
| C | regularization parameter |
| epsilon | SVR margin width |
| Kernel | kernel function |

---

### Derived During Training

| Variable | Meaning |
|------|------|
| alpha | Lagrange multipliers |
| alpha_star | dual variables in SVR |
| Q | dual matrix |
| w | weight vector |
| b | bias |
| support | support vector indices |
| beta | α − α* (SVR dual variable) |

---

# Loop / Vectorized Form

Dual matrix computation

```
Y = y @ y.T
K = X @ X.T
Q = Y * K
```

Gradient

```
gradient = ones - Q @ alpha
```

Gradient ascent update

```
alpha = alpha + lr * gradient
```

Constraint enforcement

```
alpha = clip(alpha,0,C)
alpha = alpha - (y.T @ alpha)/(y.T @ y) * y
```

---

# Algorithm Steps

## SVC Training

1 Initialize α = 0  

2 Compute dual matrix  

Q = (y yᵀ) ⊙ (X Xᵀ)

3 Repeat for each iteration

Compute gradient

gradient = 1 − Qα

Update α

α = α + η gradient

Clip α to satisfy

0 ≤ α ≤ C

Project α to satisfy

yᵀα = 0

4 Compute weight vector

w = Xᵀ(α ⊙ y)

5 Identify support vectors

αᵢ > 0

6 Compute bias

b = mean(yᵢ − wᵀxᵢ)

---

## Kernel SVC Training

Same procedure as SVC, but replace

X Xᵀ

with kernel matrix

K(X,X)

Prediction uses

f(x) = Σ αᵢ yᵢ K(xᵢ , x) + b

---

## SVR Training

SVR uses **epsilon-insensitive loss**.

Errors smaller than ε are ignored

|y − f(x)| ≤ ε

Two dual variables are introduced

α and α*

Define

β = α − α*

Prediction

f(x) = Σ βᵢ K(xᵢ , x) + b

The algorithm updates α and α* with gradient ascent while enforcing

0 ≤ α, α* ≤ C  
Σ (α − α*) = 0

---

# Time Complexity

Kernel matrix computation

O(n² d)

Training complexity

O(epoch × n²)

Prediction complexity

O(n_support × d)

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