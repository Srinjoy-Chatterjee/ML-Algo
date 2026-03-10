# Algorithm
Logistic Regression

Logistic Regression is a supervised learning algorithm used for **binary classification problems**.

Instead of predicting a continuous value, the model predicts the **probability that a sample belongs to class 1**.

The output of a linear model is transformed using the **sigmoid function**, which maps any real number to the range (0,1).

Example

P(y = 1 | x)

Classification rule

If P ≥ 0.5 → class 1  
If P < 0.5 → class 0

Logistic Regression is widely used in:

• medical diagnosis  
• spam detection  
• credit risk analysis  

---

# Model

The model first computes a **linear score**

z = Xθ

Where

| Symbol | Meaning |
|------|------|
| X | feature matrix (n × d) |
| θ | parameter vector (d × 1) |
| z | linear score |
| n | number of samples |
| d | number of features |

This score is converted to probability using the **sigmoid function**

σ(z) = 1 / (1 + e⁻ᶻ)

Final model

p = σ(Xθ)

Where

| Symbol | Meaning |
|------|------|
| p | probability that y = 1 |

Thus

P(y=1|x) = σ(Xθ)

---

# Math Implementation

## Sigmoid Function

σ(z) = 1 / (1 + e⁻ᶻ)

Where

| Symbol | Meaning |
|------|------|
| σ(z) | sigmoid function |
| z | linear score |

The sigmoid converts any real number to a probability between **0 and 1**.

---

## Loss Function

Logistic regression minimizes **Binary Cross Entropy (Log Loss)**.

L(θ) = -(1/n) Σ [y log(p) + (1-y) log(1-p)]

Where

| Symbol | Meaning |
|------|------|
| y | true label |
| p | predicted probability |

This loss heavily penalizes confident incorrect predictions.

---

## Gradient

Derivative of the loss with respect to θ

∇L = (1/n) Xᵀ (p − y)

Where

| Symbol | Meaning |
|------|------|
| p | predicted probabilities |
| y | true labels |
| Xᵀ | transpose of X |

---

## Regularization

The implementation supports **L1 and L2 regularization**.

### L2 Regularization

Penalty

λ₂‖θ‖²

Gradient

2λ₂θ

---

### L1 Regularization

Penalty

λ₁‖θ‖₁

Gradient

λ₁ sign(θ)

---

### Combined Gradient

The code combines all terms

∇L = (1/n)Xᵀ(p − y + 2λ₂θ + λ₁ sign(θ))

---

## Gradient Descent Update

θ = θ − η ∇L

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

### Derived During Training

| Variable | Meaning |
|------|------|
| z | linear score |
| prediction | sigmoid output |
| error | prediction − y with regularization |
| gradient | derivative of loss |

---

# Loop / Vectorized Form

Linear score

```
z = X @ M
```

Sigmoid computation

```
prediction = 1/(1+exp(-z))
```

Error

```
error = prediction - y
```

Gradient

```
gradient = (1/n) * X.T @ error
```

The implementation uses a **numerically stable sigmoid**.

Instead of directly computing

exp(-z)

the code uses a piecewise form

```
if z >= 0:
    1/(1+exp(-z))
else:
    exp(z)/(1+exp(z))
```

This avoids overflow when z is very large or very negative.

---

# Algorithm Steps

1 Add bias column to feature matrix

X ← [X 1]

2 Initialize parameters

θ = 0

3 Repeat for each iteration

Compute linear score

z = Xθ

Compute sigmoid probability

p = σ(z)

Compute error

error = p − y

Add regularization

error = p − y + 2λ₂θ + λ₁ sign(θ)

Compute gradient

∇L = (1/n) Xᵀ error

Update parameters

θ = θ − η ∇L

4 Stop after reaching the specified number of iterations.

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

Matrix multiplication dominates the computation.

---

# Code Implementation

```python
class LogisticRegression :

    def __init__(self,epoch=1000,lr=0.1,l1 = 0, l2 = 0):
        self.epoch = epoch          # number of gradient descent iterations
        self.lr = lr                # learning rate
        self.l1 = l1                # L1 regularization weight
        self.l2 = l2                # L2 regularization weight
        
    def fit(self,X,y):

        n = X.shape[0]              # number of samples

        ones = np.ones((n,1))       # create bias column
        X = np.hstack((X,ones))     # add bias to feature matrix

        self.M = np.zeros((X.shape[1],1))   # initialize parameter vector θ

        y = np.reshape(y,(-1,1))    # reshape target vector

        for _ in range(self.epoch):

            z = X @ self.M          # compute linear score

            # numerically stable sigmoid computation
            prediction = np.where(
                z>=0,
                1/(1+np.exp(-z)),
                np.exp(z)/(1+np.exp(z))
            )

            # compute error 
            error = prediction - y 

            gradient = 1/n * (X.T @ error) + 2 * self.l2 * self.M + self.l1 * np.sign(self.M)   # compute gradient with regularization

            self.M = self.M - self.lr * gradient   # gradient descent update


    def predict(self,X):

        ones = np.ones((X.shape[0],1))   # add bias column
        X = np.hstack((X,ones))

        z = X @ self.M                   # compute linear score

        # sigmoid probability
        prediction = np.where(
            z>=0,
            1/(1+np.exp(-z)),
            np.exp(z)/(1+np.exp(z))
        )

        return (prediction>=0.5).astype(int)   # convert probability to class label
```