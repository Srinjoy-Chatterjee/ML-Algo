# Algorithm
Logistic Regression

Logistic Regression is a supervised learning algorithm used for **binary classification** problems.

Instead of predicting a continuous value, it predicts a **probability between 0 and 1**.

The model output is passed through the **sigmoid function**, which converts any real value into a probability.

Example:

P(y = 1 | x)

If probability ≥ 0.5 → class 1  
If probability < 0.5 → class 0

---

# Model

The linear part of logistic regression is

z = Xθ

Where

| Symbol | Meaning |
|------|------|
| X | feature matrix (n × d) |
| θ | parameter vector |
| z | linear score |

The sigmoid function converts the score into a probability

σ(z) = 1 / (1 + e⁻ᶻ)

Final prediction

p = σ(Xθ)

Where

| Symbol | Meaning |
|------|------|
| p | probability of class 1 |

---

# Loss Function

Logistic regression uses **Binary Cross Entropy Loss**.

L(θ) = -(1/n) Σ [y log(p) + (1-y) log(1-p)]

Where

| Symbol | Meaning |
|------|------|
| y | true label |
| p | predicted probability |

This loss penalizes incorrect probability predictions.

---

# Gradient

Derivative of the loss with respect to θ

∇L = (1/n) Xᵀ (p − y)

Where

| Symbol | Meaning |
|------|------|
| p | predicted probabilities |
| y | true labels |

---

# Gradient Descent Update

Parameters are updated using gradient descent

θ = θ − η ∇L

Substituting gradient

θ = θ − η (1/n) Xᵀ (p − y)

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
| l1 | L1 regularization weight |
| l2 | L2 regularization weight |

### Derived

| Variable | Meaning |
|------|------|
| z | linear score |
| prediction | sigmoid output |
| error | prediction − y |
| gradient | derivative of loss |

---

# Vectorized Form

Linear score

```
z = X @ M
```

Sigmoid function

```
prediction = 1 / (1 + exp(-z))
```

Error

```
error = prediction - y
```

Gradient

```
gradient = (1/n) * X.T @ error
```

---

# Algorithm Steps

1 Add bias column to X  
2 Initialize parameters θ = 0  

Repeat for each epoch

z = Xθ  
p = sigmoid(z)  
error = p − y  
gradient = (1/n) Xᵀ error  
θ = θ − lr × gradient  

---

# Time Complexity

Matrix multiplication dominates the computation.

O(n × d)

Where

n = number of samples  
d = number of features

---

# Implementation

```python
class LogisticRegression :

    def __init__(self,epoch=1000,lr=0.1,l1 = 0, l2 = 0):
        self.epoch = epoch
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        
    def fit(self,X,y):

        n = X.shape[0]

        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        self.M = np.zeros((X.shape[1],1))

        y = np.reshape(y,(-1,1))

        for _ in range(self.epoch):

            z = X @ self.M

            prediction = np.where(
                z>=0,
                1/(1+np.exp(-z)),
                np.exp(z)/(1+np.exp(z))
            )

            error = prediction - y + 2 * self.l1 * self.M + self.l2 * np.sign(self.M)

            gradient = 1/n * (X.T @ error)

            self.M = self.M - self.lr * gradient


    def predict(self,X):

        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        z = X @ self.M

        prediction = np.where(
            z>=0,
            1/(1+np.exp(-z)),
            np.exp(z)/(1+np.exp(z))
        )

        return (prediction>=0.5).astype(int)
```