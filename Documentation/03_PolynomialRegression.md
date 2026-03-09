# Algorithm
Polynomial Regression

Polynomial Regression is used to model **non-linear relationships** between input features and the target variable.

Instead of fitting a straight line, the algorithm fits a **polynomial curve** by transforming the original features into higher-degree polynomial features and then applying **Linear Regression** on those transformed features.

Example

y = ax² + bx + c

Even though the relationship is nonlinear in **x**, the model is still **linear in the parameters**, which allows it to be solved using Linear Regression.

Example transformation

Original feature

x

Polynomial expansion (degree = 3)

[x, x², x³]

After transformation the model becomes linear in θ.

---

# Model

After polynomial feature expansion, the model becomes

ŷ = X_poly θ

Where

| Symbol | Meaning |
|------|------|
| X_poly | transformed polynomial feature matrix |
| θ | parameter vector |
| ŷ | predicted output |
| y | true output |
| n | number of samples |
| d | number of original features |

Example

If

X = [x]

and degree = 3

Then

X_poly = [x, x², x³]

For multiple features with interaction terms:

X = [x₁, x₂]

Polynomial expansion (degree 2)

[x₁, x₂, x₁², x₁x₂, x₂²]

---

# Math Implementation

Polynomial regression still uses **Linear Regression optimization**.

The only difference is that the input matrix is replaced by **polynomial features**.

### Loss Function

Mean Squared Error

L(θ) = (1/n) Σ (ŷ − y)²

Where

| Symbol | Meaning |
|------|------|
| L(θ) | loss function |
| ŷ | predicted output |
| y | true output |

Vector form

L(θ) = (1/n)(X_poly θ − y)ᵀ(X_poly θ − y)

---

### Gradient

Let

e = X_poly θ − y

Then

∇L = (2/n) X_polyᵀ (X_poly θ − y)

---

### Gradient Descent Update

θ = θ − η ∇L

Substituting the gradient

θ = θ − η (2/n) X_polyᵀ (X_poly θ − y)

Where

| Symbol | Meaning |
|------|------|
| η | learning rate |

Thus polynomial regression **reduces to linear regression on transformed features**.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| degree | maximum polynomial degree |
| interaction | include cross-feature interaction terms |
| lr | learning rate |
| epoch | number of gradient descent iterations |

### Derived During Training

| Variable | Meaning |
|------|------|
| X_poly | transformed polynomial feature matrix |
| features | list of generated polynomial features |
| θ (m in code) | parameter vector |
| prediction | predicted values |

---

# Loop / Vectorized Form

Polynomial feature generation without interaction

```
new_feature = X ** k
```

Example

```
X² = X ** 2
X³ = X ** 3
```

Feature stacking

```
X_poly = np.column_stack(features)
```

For interaction features the algorithm uses

```
combinations_with_replacement(...)
```

to generate feature combinations such as

x₁²  
x₁x₂  
x₂²

---

# Algorithm Steps

1 Transform input features into polynomial features.

For degree = k

Generate

x, x², x³ … xᵏ

If interaction terms are enabled, generate combinations of features.

2 Construct transformed feature matrix

X_poly

3 Train Linear Regression on X_poly

4 During prediction

Transform input features again into polynomial features.

5 Compute prediction

ŷ = X_poly θ

---

# Time Complexity

Feature generation complexity

O(n × d × degree)

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |

Training complexity

O(n × d_poly)

Where

d_poly = number of generated polynomial features

Prediction complexity

O(n × d_poly)

---

# Code Implementation

```python
class PolinomialRegression:
    def __init__(self,lr=0.01,epoch=1000,degree=1,interaction=False):
        self.lr = lr                     # learning rate (passed but LinearRegression object is used internally)
        self.epoch = epoch               # number of training iterations
        self.degree = degree             # maximum polynomial degree
        self.interaction = interaction   # whether to include interaction terms
        self.lr =  LinearRegression()    # underlying linear regression model

    def fit(self,X,y):
        features = self.generate_feature_set(X)   # generate polynomial feature matrix
        return self.lr.fit(features,y)            # train linear regression on expanded features

    def predict(self,X):
        features = self.generate_feature_set(X)   # generate polynomial features for prediction
        return self.lr.predict(features)          # predict using trained linear model
    
    def generate_feature_set(self,X):
        _,X_features = X.shape
        features = []

        if self.interaction:

            # generate polynomial interaction terms
            for k in range(1,self.degree+1):

                for comb in combinations_with_replacement(range(X_features),k):

                    # multiply features in the combination
                    for index in comb:
                        new_feature *=X[:,index]

                    features.append(new_feature)

        else:
            
            # generate simple polynomial powers (x, x², x³ ...)
            for k in range(1,self.degree+1):

                new_feature = X**k

                features.append(new_feature)

        # stack generated features into a single matrix
        return np.column_stack(features)
```