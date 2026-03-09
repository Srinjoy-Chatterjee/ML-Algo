# Algorithm
Softmax Regression

Softmax Regression is a generalization of Logistic Regression for **multi-class classification**.

Instead of predicting the probability of only one class, it predicts a **probability distribution over multiple classes**.

Example

Classes = {0,1,2}

Softmax outputs

P(y=0), P(y=1), P(y=2)

The class with the **highest probability** is selected as the prediction.

---

# Model

The linear model is

z = XΘ

Where

| Symbol | Meaning |
|------|------|
| X | feature matrix (n × d) |
| Θ | parameter matrix (d × k) |
| z | class scores |
| n | number of samples |
| d | number of features |
| k | number of classes |

Each column of Θ represents parameters for one class.

---

# Loss Function

Softmax regression uses **Categorical Cross Entropy Loss**.

L(Θ) = -(1/n) Σ Σ yᵢⱼ log(pᵢⱼ)

Where

| Symbol | Meaning |
|------|------|
| yᵢⱼ | true label (one-hot encoded) |
| pᵢⱼ | predicted probability |

This loss encourages the model to assign high probability to the correct class.

---

# Gradient

The gradient of the loss with respect to Θ is

∇L = (1/n) Xᵀ (P − Y)

Where

| Symbol | Meaning |
|------|------|
| P | predicted probabilities |
| Y | one-hot encoded labels |

---

# Gradient Descent Update

Parameters are updated using gradient descent.

Θ = Θ − η ∇L

Substituting gradient

Θ = Θ − η (1/n) Xᵀ (P − Y)

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
| epoch | training iterations |
| l1 | L1 regularization weight |
| l2 | L2 regularization weight |

### Derived

| Variable | Meaning |
|------|------|
| z | class score matrix |
| prediction | softmax probabilities |
| error | prediction − Y |
| gradient | derivative of loss |

---

# Vectorized Form

Linear score

```
z = X @ M
```

Numerical stability adjustment

```
z = z - max(z)
```

Softmax function

```
exp_z = exp(z)
prediction = exp_z / sum(exp_z)
```

Gradient

```
gradient = (1/n) * X.T @ (prediction - Y)
```

Vectorization computes predictions for **all samples and classes simultaneously**.

---

# Algorithm Steps

1 Add bias column to X  
2 Convert labels to one-hot vectors  
3 Initialize parameter matrix Θ  

Repeat for each epoch

z = XΘ  
P = softmax(z)  
error = P − Y  
gradient = (1/n) Xᵀ error  
Θ = Θ − lr × gradient  

---

# Time Complexity

Matrix multiplication dominates.

O(n × d × k)

Where

n = number of samples  
d = number of features  
k = number of classes

---

# Implementation

```python
class Softmax :
    def __init__(self,epoch=1000,lr=0.1,l1 = 0, l2 = 0):
        self.epoch = epoch
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        
    def softmax(z):
        z = z - np.max(z,axis=1,keepdims=True)
        exp_z = np.exp(z)
        return exp_z/np.sum(exp_z,axis=1,keepdims=True)

    def fit(self,X,y):
        n = X.shape[0]

        ones = np.ones((n,1))
        X = np.hstack((X,ones))

        k = len(np.unique(y))
        Y = np.zeros((n,k))
        Y[np.arange(n),y] = 1

        self.M = np.zeros((X.shape[1],k))

        for _ in range(self.epoch):

            z = X @ self.M
            prediction = self.softmax(z)

            error = prediction - y + 2 * self.l1 * self.M + self.l2 * np.sign(self.M)

            gradient = 1/n * (X.T @ error)

            self.M = self.M - self.lr * gradient


    def predict(self,X):

        ones = np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

        z = X @ self.M

        prediction = self.softmax(z)

        return (prediction>=0.5).astype(int)
```