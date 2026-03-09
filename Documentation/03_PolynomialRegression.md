# Algorithm
Polynomial Regression

Polynomial Regression models **nonlinear relationships** by transforming input features into polynomial features and then applying linear regression.

Example

y = ax² + bx + c

---

# Model

After polynomial feature expansion

ŷ = X_poly θ

Where

| Symbol | Meaning |
|------|------|
| X_poly | polynomial feature matrix |
| θ | parameter vector |

---

# Loss Function

Polynomial regression still uses **Mean Squared Error**.

L(θ) = (1/n) Σ (ŷ − y)²

---

# Gradient

The gradient is identical to linear regression.

∇L = (2/n) Xᵀ(Xθ − y)

---

# Gradient Descent Update

θ = θ − η (2/n) Xᵀ(Xθ − y)

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| degree | polynomial degree |
| interaction | whether to include interaction terms |
| lr | learning rate |
| epoch | iterations |

---

# Vectorized Form

Polynomial features

```
X² = X ** 2
X³ = X ** 3
```

Stack features

```
X_poly = np.column_stack(features)
```

---

# Algorithm Steps

1 Generate polynomial features  
2 Train linear regression  
3 Predict using transformed features  

---

# Time Complexity

Feature generation

O(n × d × degree)

Training

O(n × d_poly)

---

# Implementation

```python
class PolinomialRegression:
    def __init__(self,lr=0.01,epoch=1000,degree=1,interaction=False):
        self.lr = lr
        self.epoch = epoch
        self.degree = degree
        self.interaction = interaction
        self.lr =  LinearRegression()

    def fit(self,X,y):
        features = self.generate_feature_set(X)
        return self.lr.fit(features,y)

    def predict(self,X):
        features = self.generate_feature_set(X)
        return self.lr.predict(features)
    
    def generate_feature_set(self,X):
        _,X_features = X.shape
        features = []
        if self.interaction:
            for k in range(1,self.degree+1):
                for comb in combinations_with_replacement(range(X_features),k):
                    for index in comb:
                        new_feature *=X[:,index]
                    features.append(new_feature)
        else:
            
            for k in range(1,self.degree+1):
                new_feature = X**k
                features.append(new_feature)

        return np.column_stack(features)
```