# Algorithm
Naive Bayes (Multinomial and Gaussian)

Naive Bayes is a probabilistic classification algorithm based on **Bayes’ Theorem**.

The algorithm assumes that all features are **conditionally independent given the class label**.

Although this assumption is rarely true in real-world datasets, it significantly simplifies probability computation and often works well in practice.

Naive Bayes classifiers are widely used for:

• spam detection  
• document classification  
• sentiment analysis  

Two common variants are implemented:

Multinomial Naive Bayes → used for **count data** (text classification, word counts)

Gaussian Naive Bayes → used for **continuous features**

---

# Model

The model is derived from **Bayes' Theorem**.

P(y | x) = ( P(x | y) P(y) ) / P(x)

Where

| Symbol | Meaning |
|------|------|
| P(y \| x) | posterior probability |
| P(x \| y) | likelihood |
| P(y) | prior probability |
| P(x) | evidence |

Since P(x) is identical for all classes, prediction simplifies to

P(y | x) ∝ P(x | y) P(y)

Final classification rule

ŷ = argmax_y P(x | y) P(y)

Where

| Symbol | Meaning |
|------|------|
| ŷ | predicted class |

---

# Math Implementation

## Class Prior Probability

The prior probability of a class is estimated from the dataset.

P(y = c) = count(y = c) / n

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |

---

# Multinomial Naive Bayes

Used when features represent **counts or frequencies**.

Example

• word counts in documents  
• term frequency vectors  

---

## Likelihood (Multinomial Distribution)

P(x | y=c) = ∏ P(xᵢ | y=c)

Where

| Symbol | Meaning |
|------|------|
| xᵢ | feature count |
| P(xᵢ \| y=c) | probability of feature i in class c |

Feature probabilities are estimated using **frequency counts**.

---

## Laplace Smoothing

To avoid zero probabilities, **Laplace smoothing** is applied.

P(xᵢ | y=c) = (countᵢ + 1) / (total_count + V)

Where

| Symbol | Meaning |
|------|------|
| countᵢ | count of feature i in class c |
| V | vocabulary size |

---

## Log Probability Trick

Instead of multiplying many small probabilities

P(x|y) = Π P(xᵢ|y)

we compute

log P(x|y) = Σ log P(xᵢ|y)

This prevents **numerical underflow**.

Final prediction

log P(y|x) = log P(y) + Σ log P(xᵢ | y)

---

# Gaussian Naive Bayes

Gaussian Naive Bayes assumes that features follow a **normal (Gaussian) distribution**.

For each class and feature the algorithm estimates

• mean  
• variance

---

## Gaussian Likelihood

P(x | y=c) = (1 / √(2πσ²)) exp(-(x-μ)² / 2σ²)

Where

| Symbol | Meaning |
|------|------|
| μ | class mean |
| σ² | class variance |
| x | feature value |

---

## Log Likelihood

Using log probabilities

log P(x|y) = -½ [ log(2πσ²) + (x-μ)² / σ² ]

This is the form used in the implementation.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| X | feature matrix |
| y | class labels |

---

### Derived During Training

| Variable | Meaning |
|------|------|
| groups | unique class labels |
| group_index | class index of each sample |
| probability_groups | prior class probabilities |
| log_probability_groups | log of class priors |
| feature_count | feature frequency per class |
| means | mean feature value per class |
| var | variance of features per class |

---

# Loop / Vectorized Form

Class prior computation

```
probability_groups = bincount(labels) / n_samples → (k)
```

One-hot encoding

```
y_onehot = eye(n_classes)[group_index] → (n_samples , k)
```

Feature counting

```
feature_count = y_onehot.T @ X → (k , d)
```

Multinomial

```
smooth_total = np.sum(smooth_fc,axis=1,keepdims=True) → (k , 1) 
```

```
self.log_probability_features_per_group = np.log(smooth_fc/smooth_total) → (k , d)
```

Gaussian

```
self.means = feature_count / np.bincount(group_index)[:,None] → (k , d)
```

```
 diff = X[:,None,:] - self.means → (n_samples , k , d) → (n_samples , k , d)
```

```
weighted_sqr_diff = y_onehot[:,:,None] * diff_sqr → (n_samples , k , d)
```

```
var_sum = np.sum(weighted_sqr_diff,axis=0) → (k , d)
```

```
self.var = var_sum / np.bincount(group_index)[:,None] → (k , d)
```

Prediction (Multinomial)

```
log_prob = log_prior + X @ log_feature_prob.T → (n_test , k)
```

Prediction (Gaussian)

```
        log_likelihood = -0.5 * (
            np.log(2*np.pi*self.var) +
            ((X[:, None, :] - self.means) ** 2)/self.var
        ) → (n_test , k , d)
```

```
log_likelihood = log_prior + gaussian_log_density → (n_test , k)
```

Vectorization allows computing probabilities for **all samples and classes simultaneously**.

---

# Algorithm Steps

## Multinomial Naive Bayes

### Training

1 Identify unique classes  
2 Compute class prior probabilities  
3 Convert labels to one-hot encoding  
4 Compute feature counts per class  
5 Apply Laplace smoothing  
6 Compute log probabilities  

---

### Prediction

1 Compute log likelihood of each class  

log P(y|x) = log P(y) + X log P(feature|class)

2 Select class with maximum probability.

---

## Gaussian Naive Bayes

### Training

1 Identify unique classes  
2 Compute class priors  
3 Compute class mean for each feature  
4 Compute class variance for each feature  

---

### Prediction

1 Compute Gaussian log likelihood  

log P(x|y)

2 Add class prior

log P(y|x) = log P(y) + log P(x|y)

3 Select class with highest probability.

---

# Time Complexity

Training complexity

O(n × d)

Prediction complexity

O(n_test × d × k)

Where

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |
| k | number of classes |

---

# Code Implementation

## Multinomial Naive Bayes

```python
class NaiveBayesMultinomial:
    def __init__(self):
        pass

    def fit(self,X,y):

        n_samples,n_features = X.shape

        # find unique classes and map labels to indices
        self.groups,group_index = np.unique(y,return_inverse=True)

        n_groups = len(self.groups)

        # compute class prior probabilities
        probability_groups = np.bincount(group_index)/n_samples

        # store log prior probabilities
        self.log_probability_groups = np.log(probability_groups)

        # convert labels to one-hot encoding
        y_onehot = np.eye(n_groups)[group_index]

        # compute feature counts per class
        feature_count = y_onehot.T @ X

        # apply Laplace smoothing
        smooth_fc = feature_count + 1

        smooth_total = np.sum(smooth_fc,axis=1,keepdims=True)

        # compute log feature probabilities
        self.log_probability_features_per_group = np.log(smooth_fc/smooth_total)
        
    def predict(self,X):

        # compute log posterior probabilities
        log_prob = self.log_probability_groups + X @ self.log_probability_features_per_group.T

        # select class with maximum probability
        return self.groups[np.argmax(log_prob,axis=1)]
```

---

## Gaussian Naive Bayes

```python
class NaiveBayesGaussian:
    def __init__(self):
        pass   

    def fit(self,X,y):

        n_samples,n_features = X.shape

        # find unique classes
        self.groups,group_index = np.unique(y,return_inverse=True)

        n_groups = len(self.groups)

        # compute class prior probabilities
        probability_groups = np.bincount(group_index)/n_samples

        self.log_probability_groups = np.log(probability_groups) 

        # one-hot encoding of labels
        y_onehot = np.eye(n_groups)[group_index]

        # compute class means
        feature_count = y_onehot.T @ X  

        self.means = feature_count / np.bincount(group_index)[:,None]

        # compute variance
        diff = X[:,None,:] - self.means

        diff_sqr = diff ** 2

        weighted_sqr_diff = y_onehot[:,:,None] * diff_sqr

        var_sum = np.sum(weighted_sqr_diff,axis=0)

        self.var = var_sum / np.bincount(group_index)[:,None]

        # add small value to avoid division by zero
        self.var += 1e-9

    def predict(self,X):

        # compute Gaussian log likelihood
        log_likelihood = -0.5 * (
            np.log(2*np.pi*self.var) +
            ((X[:, None, :] - self.means) ** 2)/self.var
        )

        # sum across features
        log_prob = self.log_probability_groups + np.sum(log_likelihood, axis=2)

        return self.groups[np.argmax(log_prob, axis=1)]
```