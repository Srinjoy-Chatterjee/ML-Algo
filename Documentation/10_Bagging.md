# Algorithm
Bagging (Bootstrap Aggregating)

Bagging is an ensemble learning technique used to improve model stability and reduce variance.

The main idea is to train multiple models on different **bootstrap samples** of the dataset and then combine their predictions.

Bootstrap sampling means sampling the training data **with replacement**.

Each model sees a slightly different dataset, which reduces overfitting when predictions are aggregated.

Bagging is commonly used with high variance models such as decision trees.

---

# Model

Bagging does not define a new prediction function.

Instead it combines predictions from multiple models.

If there are **M models**

Prediction is

Classification

ŷ = mode(h₁(x), h₂(x), … , hₘ(x))

Regression

ŷ = (1/M) Σ hᵢ(x)

Where

| Symbol | Meaning |
|------|------|
| hᵢ(x) | prediction from model i |
| M | number of estimators |

---

# Loss Function

Bagging does not introduce a new loss function.

Each individual model optimizes its own loss function.

The ensemble prediction reduces variance by averaging multiple models.

---

# Gradient

Bagging does not use gradient descent.

It works by **resampling the dataset and training independent models**.

---

# Gradient Descent Update

Not applicable.

Bagging is an ensemble strategy rather than an optimization algorithm.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| model | base learning model |
| n_estimators | number of models in the ensemble |

### Derived

| Variable | Meaning |
|------|------|
| models | list of trained models |
| indx | bootstrap sample indices |
| y_pred | predictions from all models |

---

# Vectorized Form

Bootstrap sampling

```
indices = np.random.choice(n, n, replace=True)
```

Prediction aggregation

Classification

```
mode(predictions)
```

Regression

```
mean(predictions)
```

---

# Algorithm Steps

Training

1 Repeat n_estimators times  
2 Sample training data with replacement  
3 Train a base model on the sampled dataset  
4 Store the trained model  

Prediction

1 Collect predictions from all models  
2 Combine predictions  

Classification → majority vote  
Regression → average prediction

---

# Time Complexity

Training

O(n_estimators × training_cost)

Prediction

O(n_estimators × prediction_cost)

Where

| Symbol | Meaning |
|------|------|
| n_estimators | number of models |
| training_cost | cost of training base model |

---

# Implementation

## Bagging for Classification

```python
class Bagging:
    def __init__(self,model,n_estimators):
        self.model = model
        self.n_estimators = n_estimators
        self.models = []

    def fit(self,X,y):
        n = X.shape[0]
        for _ in range(self.n_estimators):

            indx = np.random.choice(n,n,replace=True)

            new_X = X[indx]
            new_Y = y[indx]

            model = deepcopy(self.model)

            model.fit(new_X,new_Y)

            self.models.append(model)

    def predict(self,X):

        y_pred = np.array([model.predict(X) for model in self.models])

        final = []

        for column in y_pred.T:

            final.append(np.bincount(column.astype(int)).argmax())

        return final
```

---

## Bagging for Regression

Regression combines predictions by averaging.

```python
class Bagging:
    def __init__(self,model,n_estimators):
        self.model = model
        self.n_estimators = n_estimators
        self.models = []

    def fit(self,X,y):

        n = X.shape[0]

        for _ in range(self.n_estimators):

            indx = np.random.choice(n,n,replace=True)

            new_X = X[indx]

            new_Y = y[indx]

            model = deepcopy(self.model)

            model.fit(new_X,new_Y)

            self.models.append(model)

    def predict(self,X):

        y_pred = np.array([model.predict(X) for model in self.models])

        return np.mean(y_pred,axis=0)
```