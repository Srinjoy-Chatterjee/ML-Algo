# Algorithm
AdaBoost (Adaptive Boosting)

AdaBoost is an ensemble learning algorithm that combines multiple **weak learners** to produce a strong predictive model.

The core idea of AdaBoost is **sequential learning with adaptive sample weights**.

Instead of training models independently, AdaBoost trains them **one after another**.

After each model:

• Misclassified samples receive **higher weights**  
• Correctly classified samples receive **lower weights**

This forces the next weak learner to focus more on difficult samples.

In this implementation the weak learner is a **decision stump** (a decision tree with depth 1).

Two variants are implemented:

AdaBoost Classification  
AdaBoost Regression

---

# Model

The final model is a **weighted combination of weak learners**.

Assume there are **T weak learners**

h₁(x), h₂(x), … , hₜ(x)

Each learner receives a weight **αₜ** based on its performance.

---

## Classification Prediction

Final classifier

f(x) = sign( Σ αₜ hₜ(x) )

Where

| Symbol | Meaning |
|------|------|
| hₜ(x) | prediction of weak learner t |
| αₜ | weight of weak learner |
| T | number of weak learners |

Learners with **lower error receive higher weights**.

---

## Regression Prediction

Regression uses a weighted average.

f(x) = ( Σ αₜ hₜ(x) ) / ( Σ αₜ )

Where

| Symbol | Meaning |
|------|------|
| αₜ | learner weight |

---

# Math Implementation

## Initialization

Each training sample starts with equal weight.

wᵢ = 1 / n

Where

| Symbol | Meaning |
|------|------|
| wᵢ | weight of sample i |
| n | number of samples |

---

## Weighted Error

For a weak learner

error = Σ wᵢ I(yᵢ ≠ h(xᵢ))

Where

| Symbol | Meaning |
|------|------|
| I | indicator function |
| yᵢ | true label |
| h(xᵢ) | predicted label |

---

## Learner Weight

The importance of each learner is

α = 0.5 log((1 − error) / error)

If error is small → α becomes large.

---

## Sample Weight Update

Weights are updated after each iteration.

Misclassified samples increase weight.

Correct samples decrease weight.

wᵢ = wᵢ × exp(−α yᵢ h(xᵢ))

Then weights are normalized.

wᵢ = wᵢ / Σ wᵢ

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
| sample_weights | importance of each sample |
| threshold | stump split threshold |
| feature_index | selected feature |

---

### Derived During Prediction

| Variable | Meaning |
|------|------|
| scores | accumulated weighted predictions |
| preds | predictions from each stump |

---

# Loop / Vectorized Form

Weighted error

```
total_error = np.sum(sample_weights[misclassified])
```

Learner weight

```
alpha = 0.5 * np.log((1 - error) / error)
```

Weight update

```
w_i = w_i * exp(alpha)     # misclassified
w_i = w_i * exp(-alpha)    # correct
```

Normalization

```
w_i = w_i / sum(w_i)
```

---

# Complete Algorithm Steps

## Training

1 Initialize sample weights

wᵢ = 1/n

2 Repeat **n_estimators times**

Train weak learner using weighted dataset.

Compute predictions of weak learner.

Compute weighted error.

Compute learner weight

α = 0.5 log((1 − error)/error)

Update sample weights

Increase weight for misclassified samples.

Decrease weight for correctly classified samples.

Normalize weights.

Store weak learner and its weight.

---

## Prediction

For each sample

Collect predictions from all weak learners.

Classification

Compute weighted vote.

Regression

Compute weighted average.

---

# Time Complexity

Training complexity

O(n_estimators × stump_training_cost)

Prediction complexity

O(n_estimators)

Where

| Symbol | Meaning |
|------|------|
| n_estimators | number of weak learners |
| n | number of samples |
| d | number of features |

---

# Code Implementation

## AdaBoost Classification

```python
class AdaBoost:

    def __init__(self,n_estimator):

        self.n_estimator = n_estimator      # number of weak learners
        self.roots = []                     # list of trained decision stumps
        self.weights = []                   # model weights alpha
        self.feature_sorted = []            # sorted feature indices for faster splits

    def find_best_split(self,X,y):

        n_samples,n_features = X.shape
        best_ginni = float("inf")
        best_thresold = None
        best_feature = None

        for feature in range(n_features-1):

            sorted_indx = self.feature_sorted[:,feature]
            X_sorted = X[sorted_indx,feature]
            y_sorted = y[sorted_indx]

            weight_sorted = X[sorted_indx,-1]     # sample weights

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
```

```python
    def build_stump(self,X,y):

        feature,thresold = self.find_best_split(X,y)

        left_indx = X[:,feature] <= thresold
        right_indx = ~left_indx

        # majority class in each branch
        left_val = np.argmax(np.bincount(y[left_indx]))
        right_val = np.argmax(np.bincount(y[right_indx]))

        # predictions of stump
        preds = np.where(left_indx, left_val, right_val)

        # misclassified samples
        misclassified = preds != y

        # weighted error
        total_error = np.sum(X[misclassified, -1]) + 1e-6

        # learner weight
        model_weight = 0.5 * np.log((1 - total_error) / total_error)

        # update sample weights
        X[misclassified, -1] *= np.exp(model_weight)
        X[~misclassified, -1] *= np.exp(-model_weight)

        # normalize weights
        total = np.sum(X[:,-1]) + 1e-6
        X[:,-1] = X[:,-1]/total

        left = Node(value=left_val)
        right = Node(value=right_val)

        root = Node(feature_index=feature,threshold=thresold,left=left,right=right)

        return root,model_weight
```

```python
    def fit(self,X,y):

        n_samples = X.shape[0]

        self.n_classes = int(np.max(y))+1

        # initialize sample weights
        ones = np.ones((n_samples,1))
        bias = ones/n_samples

        X = np.hstack((X,bias))

        # pre-sort features for faster splitting
        self.feature_sorted = np.argsort(X[:,:-1],axis=0)

        for _ in range(self.n_estimator):

            root,weight = self.build_stump(X,y)

            self.roots.append(root)
            self.weights.append(weight)
```

```python
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
class AdaBoost:

    def __init__(self,n_estimator):

        self.n_estimator = n_estimator
        self.roots = []
        self.weights = []
        self.feature_sorted = []
```

```python
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
