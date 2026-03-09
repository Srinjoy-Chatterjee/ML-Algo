# Algorithm
Naive Bayes (Multinomial and Gaussian)

Naive Bayes is a probabilistic classification algorithm based on **Bayes’ Theorem**.

The key assumption is that **features are conditionally independent given the class**.

This assumption simplifies probability computation and allows efficient classification.

Two common variants are used:

Multinomial Naive Bayes → used for count data (text classification, word counts)

Gaussian Naive Bayes → used for continuous features

---

# Model

The model is based on **Bayes' theorem**

P(y | x) = ( P(x | y) P(y) ) / P(x)

Where

| Symbol | Meaning |
|------|------|
| P(y | x) | posterior probability |
| P(x | y) | likelihood |
| P(y) | prior probability |
| P(x) | evidence |

Since P(x) is the same for all classes, prediction uses

P(y | x) ∝ P(x | y) P(y)

The predicted class is

argmax_y P(x | y) P(y)

---

# Loss Function

Naive Bayes does not explicitly optimize a loss function.

Instead it directly estimates probabilities:

• class prior probabilities  
• class conditional probabilities

Prediction selects the class with the **maximum posterior probability**.

---

# Gradient

Naive Bayes does not use gradient descent.

Parameters are estimated using **frequency statistics from the dataset**.

---

# Gradient Descent Update

Not applicable.

No iterative optimization is performed.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| X | feature matrix |
| y | class labels |

### Derived

| Variable | Meaning |
|------|------|
| groups | unique classes |
| probability_groups | prior probabilities |
| feature_count | class feature counts |
| means | mean of features per class |
| var | variance of features per class |

---

# Vectorized Form

Class priors

```
probability_groups = bincount(labels) / n_samples
```

Feature counts

```
feature_count = y_onehot.T @ X
```

Prediction

```
log_prob = log_prior + X @ log_feature_prob.T
```

Using logarithms avoids **numerical underflow** when multiplying many probabilities.

---

# Algorithm Steps

Training

1 Identify unique classes  
2 Compute class prior probabilities  
3 Compute class conditional probabilities  

Prediction

1 Compute posterior probability for each class  
2 Select class with highest probability  

---

# Time Complexity

Training

O(n × d)

Prediction

O(n_test × d × k)

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |
| k | number of classes |

---

# Implementation

## Multinomial Naive Bayes

```python
class NaiveBayesMultinomial:
    def __init__(self):
        pass

    def fit(self,X,y):

        n_samples,n_features = X.shape

        self.groups,group_index = np.unique(y,return_inverse=True)

        n_groups = len(self.groups)

        probability_groups = np.bincount(group_index)/n_samples

        self.log_probability_groups = np.log(probability_groups)

        y_onehot = np.eye(n_groups)[group_index]

        feature_count = y_onehot.T @ X

        smooth_fc = feature_count + 1

        smooth_total = np.sum(smooth_fc,axis=1,keepdims=True)

        self.log_probability_features_per_group = np.log(smooth_fc/smooth_total)
        
    def predict(self,X):

        log_prob = self.log_probability_groups + X @ self.log_probability_features_per_group.T

        return self.groups[np.argmax(log_prob,axis=1)]
```

---

## Gaussian Naive Bayes

Gaussian Naive Bayes assumes that features follow a **normal distribution**.

Probability density function

P(x | y) = (1 / √(2πσ²)) exp(-(x-μ)² / 2σ²)

Where

| Symbol | Meaning |
|------|------|
| μ | class mean |
| σ² | class variance |

```python
class NaiveBayesGaussian:
    def __init__(self):
        pass   

    def fit(self,X,y):

        n_samples,n_features = X.shape

        self.groups,group_index = np.unique(y,return_inverse=True)

        n_groups = len(self.groups)

        probability_groups = np.bincount(group_index)/n_samples

        self.log_probability_groups = np.log(probability_groups) 

        y_onehot = np.eye(n_groups)[group_index]

        feature_count = y_onehot.T @ X  

        self.means = feature_count / np.bincount(group_index)[:,None]

        diff = X[:,None,:] - self.means

        diff_sqr = diff ** 2

        weighted_sqr_diff = y_onehot[:,:,None] - diff_sqr

        var_sum = np.sum(weighted_sqr_diff,axis=0)

        self.var = var_sum / np.bincount(group_index)[:,None]

        self.var += 1e-9

    def predict(self,X):

        log_likelihood = self.log_probability_groups + -0.5 * (
            np.log(2*np.pi*self.var) +
            ((X[:, None, :] - self.means) ** 2)/self.var
        )

        return self.groups[np.argmax(log_likelihood,axis=1)]
```