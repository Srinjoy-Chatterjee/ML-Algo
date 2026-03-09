# Algorithm
Softmax Regression

Softmax Regression is a generalization of Logistic Regression used for **multi-class classification**.

Instead of predicting the probability of a single class, Softmax Regression predicts a **probability distribution across multiple classes**.

Example

Classes = {0,1,2}

Softmax output

P(y=0), P(y=1), P(y=2)

The predicted class is the one with the **highest probability**.

Softmax Regression is commonly used in:

• image classification  
• document classification  
• speech recognition  

---

# Model

The model first computes a **linear score for each class**.

z = XΘ

Where

| Symbol | Meaning |
|------|------|
| X | feature matrix (n × d) |
| Θ | parameter matrix (d × k) |
| z | class score matrix (n × k) |
| n | number of samples |
| d | number of features |
| k | number of classes |

Each column of Θ represents the parameters for a specific class.

---

## Softmax Function

The linear scores are converted into probabilities using the **softmax function**.

Softmax

pᵢⱼ = exp(zᵢⱼ) / Σ exp(zᵢk)

Where

| Symbol | Meaning |
|------|------|
| pᵢⱼ | probability that sample i belongs to class j |
| zᵢⱼ | score for class j |
| k | number of classes |

Important property

Σ pᵢⱼ = 1

This ensures that the outputs form a **valid probability distribution**.

---

# Math Implementation

## Numerical Stability

Direct computation of softmax can overflow if z is very large.

To prevent this, the implementation subtracts the maximum value from each row.

z' = z − max(z)

This does not change the probabilities but stabilizes exponentiation.

---

## Loss Function

Softmax regression minimizes **Categorical Cross Entropy Loss**.

L(Θ) = -(1/n) Σ Σ yᵢⱼ log(pᵢⱼ)

Where

| Symbol | Meaning |
|------|------|
| yᵢⱼ | one-hot encoded true label |
| pᵢⱼ | predicted probability |

---

## Gradient

Derivative of the loss with respect to Θ

∇L = (1/n) Xᵀ (P − Y)

Where

| Symbol | Meaning |
|------|------|
| P | predicted probability matrix |
| Y | one-hot encoded label matrix |

---

## Regularization

The implementation supports **L1 and L2 regularization**.

### L2 Regularization

Penalty

λ₂‖Θ‖²

Gradient

2λ₂Θ

---

### L1 Regularization

Penalty

λ₁‖Θ‖₁

Gradient

λ₁ sign(Θ)

---

### Combined Gradient

The implemented gradient becomes

∇L = (1/n) Xᵀ(P − Y + 2λ₂Θ + λ₁ sign(Θ))

---

## Gradient Descent Update

Parameters are updated iteratively

Θ = Θ − η ∇L

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

---

### Derived During Training

| Variable | Meaning |
|------|------|
| z | class score matrix |
| prediction | softmax probability matrix |
| error | prediction − Y with regularization |
| gradient | derivative of loss |
| Y | one-hot encoded label matrix |

---

# Loop / Vectorized Form

Linear score

```
z = X @ M
```

Numerical stability adjustment

```
z = z - max(z)
```

Softmax computation

```
exp_z = exp(z)
prediction = exp_z / sum(exp_z)
```

Gradient

```
gradient = (1/n) * X.T @ (prediction - Y)
```

Vectorization allows computing probabilities for **all samples and classes simultaneously**.

---

# Algorithm Steps

1 Add bias column to feature matrix

X ← [X 1]

2 Convert labels into one-hot encoded vectors

Y[i,j] = 1 if sample i belongs to class j

3 Initialize parameter matrix

Θ = 0

4 Repeat for each iteration

Compute linear scores

z = XΘ

Apply numerical stability shift

z = z − max(z)

Compute softmax probabilities

P = softmax(z)

Compute error

error = P − Y

Add regularization

error = P − Y + 2λ₂Θ + λ₁ sign(Θ)

Compute gradient

∇L = (1/n) Xᵀ error

Update parameters

Θ = Θ − η ∇L

5 Stop after reaching the specified number of iterations.

---

# Time Complexity

Training complexity

O(n × d × k)

Prediction complexity

O(n × d × k)

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |
| k | number of classes |

Matrix multiplication dominates the computation.

---

# Code Implementation

```python
class Softmax :
    def __init__(self,epoch=1000,lr=0.1,l1 = 0, l2 = 0):
        self.epoch = epoch            # number of gradient descent iterations
        self.lr = lr                  # learning rate
        self.l1 = l1                  # L1 regularization weight
        self.l2 = l2                  # L2 regularization weight
        
    def softmax(z):
        z = z - np.max(z,axis=1,keepdims=True)  # numerical stability adjustment
        exp_z = np.exp(z)                       # exponentiate scores
        return exp_z/np.sum(exp_z,axis=1,keepdims=True)  # normalize to probabilities

    def fit(self,X,y):
        n = X.shape[0]               # number of samples

        ones = np.ones((n,1))        # create bias column
        X = np.hstack((X,ones))      # add bias to feature matrix

        k = len(np.unique(y))        # number of classes

        # convert labels to one-hot encoding
        Y = np.zeros((n,k))
        Y[np.arange(n),y] = 1

        self.M = np.zeros((X.shape[1],k))   # initialize parameter matrix Θ

        for _ in range(self.epoch):

            z = X @ self.M                   # compute class scores

            prediction = self.softmax(z)     # compute softmax probabilities

            # compute error with regularization
            error = prediction - y + 2 * self.l1 * self.M + self.l2 * np.sign(self.M)

            gradient = 1/n * (X.T @ error)   # compute gradient

            self.M = self.M - self.lr * gradient   # update parameters


    def predict(self,X):

        ones = np.ones((X.shape[0],1))   # add bias column
        X = np.hstack((X,ones))

        z = X @ self.M                   # compute class scores

        prediction = self.softmax(z)     # compute probabilities

        return (prediction>=0.5).astype(int)   # convert probabilities to class predictions
```