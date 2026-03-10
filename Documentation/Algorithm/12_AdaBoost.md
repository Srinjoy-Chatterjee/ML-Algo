# Algorithm
AdaBoost (Adaptive Boosting)

AdaBoost is an ensemble learning algorithm that combines multiple **weak learners** to produce a strong predictive model.

The key idea is **sequential training with adaptive sample weights**.

Instead of training models independently, AdaBoost trains them **one after another**.  
After each model:

• Misclassified samples receive **higher weights**  
• Correctly predicted samples receive **lower weights**

This forces the next weak learner to focus more on the difficult samples.

In this implementation the weak learner is a **decision stump** (a decision tree with depth = 1).

Two variants are implemented:

• **AdaBoostClassifier** → classification  
• **AdaBoostRegressor** → regression

---

# Model

AdaBoost constructs the final model as a **weighted combination of weak learners**.

Assume there are **T weak learners**

h₁(x), h₂(x), … , hₜ(x)

Each learner receives a weight **αₜ** based on its performance.

---

## Classification Prediction

Final classifier

```
f(x) = sign( Σ αₜ hₜ(x) )
```

Where

| Symbol | Meaning |
|------|------|
| hₜ(x) | prediction of weak learner t |
| αₜ | weight of weak learner |
| T | number of learners |

Learners with **lower error receive larger weights**.

---

## Regression Prediction

Regression uses a weighted average.

```
f(x) = ( Σ αₜ hₜ(x) ) / ( Σ αₜ )
```

Where

| Symbol | Meaning |
|------|------|
| αₜ | learner weight |
| hₜ(x) | regression prediction |

---

# Math Implementation

AdaBoost uses **different formulations for classification and regression**.

---

# AdaBoost Classification

AdaBoost classification minimizes **exponential loss**.

```
L = Σ exp(-y f(x))
```

Where

| Symbol | Meaning |
|------|------|
| y | true label |
| f(x) | ensemble prediction |

---

## Sample Weight Initialization

All samples start with equal importance.

```
wᵢ = 1 / n
```

Where

| Symbol | Meaning |
|------|------|
| wᵢ | weight of sample i |
| n | number of samples |

---

## Weighted Error

Weak learner error is computed using **weighted misclassification**.

```
error = Σ wᵢ I(yᵢ ≠ h(xᵢ))
```

Where

| Symbol | Meaning |
|------|------|
| wᵢ | weight of sample i |
| I | indicator function |
| yᵢ | true label |
| h(xᵢ) | predicted label |

Indicator function

```
I(condition) =
1 if condition is true
0 otherwise
```

---

## Learner Weight

The importance of each weak learner is

```
α = 0.5 log((1 − error) / error)
```

Properties

| Error | α |
|------|------|
| small | large positive α |
| 0.5 | α = 0 |
| > 0.5 | negative α |

---

## Sample Weight Update

Weights are updated using

```
wᵢ = wᵢ × exp(-α yᵢ h(xᵢ))
```

Meaning

| Case | Weight Update |
|------|------|
| correct prediction | weight decreases |
| wrong prediction | weight increases |

---

## Weight Normalization

After updating weights they are normalized

```
wᵢ = wᵢ / Σ wᵢ
```

---

# Weighted Gini (Used in Stump Training)

Since samples have weights, impurity must also be weighted.

Class probability

```
p_k = (Σ weights of class k) / (total weight)
```

Weighted Gini impurity

```
Gini = 1 − Σ p_k²
```

Split impurity

```
Gini_split =
W_left * Gini_left +
W_right * Gini_right
```

Where

| Symbol | Meaning |
|------|------|
| W_left | total weight of left node |
| W_right | total weight of right node |

The split with the **lowest weighted impurity** is selected.

---

# AdaBoost Regression

AdaBoost regression follows the **AdaBoost.R2 algorithm**.

Instead of classification error it uses **absolute prediction error**.

---

## Sample Weight Initialization

```
wᵢ = 1 / n
```

---

## Prediction Error

For each sample

```
eᵢ = |yᵢ − h(xᵢ)|
```

Normalized error

```
eᵢ = eᵢ / max(e)
```

---

## Model Error

Weighted error

```
E = Σ wᵢ eᵢ
```

---

## Model Weight Factor

```
β = E / (1 − E)
```

Learner weight

```
α = log(1 / β)
```

---

## Sample Weight Update

Weights are updated using

```
wᵢ = wᵢ × β^(1 − eᵢ)
```

Meaning

| Error | Weight Update |
|------|------|
| small error | weight decreases |
| large error | weight increases |

---

## Weight Normalization

```
wᵢ = wᵢ / Σ wᵢ
```

---

# Weighted SSE (Used in Regression Stumps)

Regression stumps minimize **weighted sum of squared errors**.

Weighted mean

```
μ = ( Σ wᵢ yᵢ ) / ( Σ wᵢ )
```

Weighted SSE

```
SSE = Σ wᵢ (yᵢ − μ)²
```

Split objective

```
SSE_split = SSE_left + SSE_right
```

The split with the **minimum weighted SSE** is chosen.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| n_estimator | number of weak learners |

---

### Derived During Training

| Variable | Meaning |
|------|------|
| roots | decision stump models |
| weights | learner weights α |
| sample_weights | importance of each training sample |
| threshold | split threshold |
| feature_index | selected feature |

---

### Derived During Prediction

| Variable | Meaning |
|------|------|
| scores | accumulated weighted predictions |
| preds | predictions from each stump |

---

# Loop / Vectorized Form

## Classification

Weighted error

```
error = np.sum(sample_weights[misclassified])
```

Learner weight

```
alpha = 0.5 * np.log((1 - error) / error)
```

Weight update

```
sample_weights[misclassified] *= np.exp(alpha)
sample_weights[~misclassified] *= np.exp(-alpha)
```

Normalization

```
sample_weights /= np.sum(sample_weights)
```

---

## Regression

Prediction error

```
error = np.abs(y - preds)
```

Normalize error

```
error = error / np.max(error)
```

Model error

```
E = np.sum(sample_weights * error)
```

Beta

```
beta = E / (1 - E)
```

Weight update

```
sample_weights *= beta ** (1 - error)
```

Normalization

```
sample_weights /= np.sum(sample_weights)
```

---

# Complete Algorithm Steps

## AdaBoost Classification Training

1 Initialize sample weights

```
wᵢ = 1/n
```

2 For each estimator

Train decision stump using weighted dataset.

Compute predictions.

Compute weighted error

```
error = Σ wᵢ I(yᵢ ≠ h(xᵢ))
```

Compute learner weight

```
α = 0.5 log((1 − error)/error)
```

Update sample weights.

Normalize weights.

Store stump and learner weight.

---

## AdaBoost Regression Training

1 Initialize sample weights

```
wᵢ = 1/n
```

2 Train regression stump.

3 Compute prediction errors.

4 Compute model error

```
E = Σ wᵢ eᵢ
```

5 Compute learner weight

```
α = log(1/β)
```

6 Update sample weights

```
wᵢ = wᵢ × β^(1 − eᵢ)
```

7 Normalize weights.

---

## Prediction

For each input sample collect predictions from all weak learners.

Classification

```
sign( Σ αₜ hₜ(x) )
```

Regression

```
( Σ αₜ hₜ(x) ) / ( Σ αₜ )
```

---

# Code Implementation

## AdaBoost Classification

```python
class AdaBoostClassifier:

    def __init__(self,n_estimator):

        self.n_estimator = n_estimator
        self.roots = []
        self.weights = []
        self.feature_sorted = []

    def find_best_split(self,X,y):

        n_samples,n_features = X.shape
        best_ginni = float("inf")
        best_thresold = None
        best_feature = None
        for feature in range(n_features-1):

            sorted_indx = self.feature_sorted[:,feature]
            X_sorted = X[sorted_indx,feature]
            y_sorted = y[sorted_indx]
            weight_sorted = X[sorted_indx,-1]
            n_classes = int(np.max(y))+1
            left_count = np.zeros(n_classes)
            right_count = np.bincount(y_sorted,weights=weight_sorted,minlength=n_classes)
            left_sum = 0
            right_sum = np.sum(weight_sorted)

            for i in range(1,n_samples):

                if(X_sorted[i]==X_sorted[i-1]): continue

                y_val = y_sorted[i-1]
                w = weight_sorted[i-1]
                left_count[y_val]+=w
                right_count[y_val]-=w

                left_sum+=w
                right_sum-=w

                prob_left = left_count/(left_sum+1e-6)
                prob_right = right_count/(right_sum+1e-6)

                left_ginni = 1-np.sum(prob_left**2)
                right_ginni = 1-np.sum(prob_right**2)


                ginni = left_sum * left_ginni + right_sum * right_ginni

                if(best_ginni>ginni):
                    best_ginni = ginni
                    best_feature = feature
                    best_thresold = (X_sorted[i] + X_sorted[i-1])/2
            
        return best_feature,best_thresold
            
    def build_stump(self,X,y):

        feature,thresold = self.find_best_split(X,y)

        left_indx = X[:,feature] <= thresold
        right_indx = ~left_indx

        left_val = np.argmax(np.bincount(y[left_indx]))
        right_val = np.argmax(np.bincount(y[right_indx]))

        # predictions of the stump
        preds = np.where(left_indx, left_val, right_val)

        # misclassified samples
        misclassified = preds != y

        # weighted error
        total_error = np.sum(X[misclassified, -1]) + 1e-6

        model_weight = 0.5 * np.log((1 - total_error) / total_error)

        # weight update
        X[misclassified, -1] *= np.exp(model_weight)
        X[~misclassified, -1] *= np.exp(-model_weight)

        #normalize
        total = np.sum(X[:,-1]) + 1e-6
        X[:,-1] = X[:,-1]/total

        left = Node(value=left_val)
        right = Node(value=right_val)
        root = Node(feature_index=feature,threshold=thresold,left=left,right=right)

        return root,model_weight

    def fit(self,X,y):

        n_samples = X.shape[0]
        self.n_classes = int(np.max(y))+1
        # add bias
        ones = np.ones((n_samples,1))
        bias = ones/n_samples
        X = np.hstack((X,bias))
        self.feature_sorted = np.argsort(X[:,:-1],axis=0)
        for _ in range(self.n_estimator):
            root,weight = self.build_stump(X,y)
            self.roots.append(root)
            self.weights.append(weight)

    def predict(self,X):

        n_samples = X.shape[0]
        scores = np.zeros((n_samples,self.n_classes))

        for root,weight in zip(self.roots,self.weights):

            left_idx = X[:,root.feature_index] <= root.threshold
            preds = np.where(left_idx, root.left.value, root.right.value)

            scores[np.arange(n_samples), preds] += weight

        return np.argmax(scores,axis=1)
   
```

---

## AdaBoost Regression

```python
class AdaBoostRegressor:

    def __init__(self,n_estimator):

        self.n_estimator = n_estimator
        self.roots = []
        self.weights = []
        self.feature_sorted = []

    def find_best_split(self,X,y):

        n_samples,n_features = X.shape
        best_sse = float("inf")
        best_thresold = None
        best_feature = None
        for feature in range(n_features-1):

            sorted_indx = self.feature_sorted[:,feature]
            X_sorted = X[sorted_indx,feature]
            y_sorted = y[sorted_indx]
            weight_sorted = X[sorted_indx,-1]
            left_sum = 0
            left_sq_sum = 0
            left_weight_sum = 0
            right_sum = np.sum( weight_sorted * y_sorted)
            right_sq_sum = np.sum( weight_sorted * y_sorted**2)
            right_weight_sum = np.sum(weight_sorted)

            for i in range(1,n_samples):

                if(X_sorted[i]==X_sorted[i-1]): continue

                y_val = y_sorted[i-1]
                w = weight_sorted[i-1]
                
                left_sum += w*y_val
                right_sum -= w*y_val

                left_sq_sum += w*y_val**2
                right_sq_sum-= w*y_val**2

                left_weight_sum += w
                right_weight_sum -=w

                left = left_sq_sum - (left_sum**2)/(left_weight_sum+1e-6)
                right = right_sq_sum - (right_sum**2)/(right_sq_sum+1e-6)

                sse = left_sum * left + right_sum * right

                if(best_sse>sse):
                    best_sse = sse
                    best_feature = feature
                    best_thresold = (X_sorted[i] + X_sorted[i-1])/2
            
        return best_feature,best_thresold
            
    def build_stump(self,X,y):

        feature,thresold = self.find_best_split(X,y)

        left_indx = X[:,feature] <= thresold
        right_indx = ~left_indx

        left_val = np.mean(y[left_indx])
        right_val = np.mean(y[right_indx])

        # predictions of the stump
        preds = np.where(left_indx, left_val, right_val)

        # weighted error
        total_error = np.abs(y-preds) + 1e-6
        model_weight = 0.5 * np.log((1 - total_error) / total_error)
        beta = np.sum(model_weight * total_error)
        beta = beta/(1-beta)

        # weight update
        X[:, -1] *= beta**(1-total_error)

        #normalize
        total = np.sum(X[:,-1]) + 1e-6
        X[:,-1] = X[:,-1]/total

        left = Node(value=left_val)
        right = Node(value=right_val)
        root = Node(feature_index=feature,threshold=thresold,left=left,right=right)

        return root,model_weight

    def fit(self,X,y):

        n_samples = X.shape[0]
        self.n_classes = int(np.max(y))+1
        # add bias
        ones = np.ones((n_samples,1))
        bias = ones/n_samples
        X = np.hstack((X,bias))
        self.feature_sorted = np.argsort(X[:,:-1],axis=0)
        for _ in range(self.n_estimator):
            root,weight = self.build_stump(X,y)
            self.roots.append(root)
            self.weights.append(weight)

    def predict(self,X):

        n_samples = X.shape[0]
        preds = np.zeros(n_samples)
        weight_sum = np.sum(self.weights)

        for root,alpha in zip(self.roots,self.weights):

            left_idx = X[:,root.feature_index] <= root.threshold
            stump_pred = np.where(left_idx, root.left.value, root.right.value)

            preds += alpha * stump_pred

        return preds / weight_sum

```
