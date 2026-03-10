# Algorithm
Bagging (Bootstrap Aggregating)

Bagging is an ensemble learning technique designed to **reduce variance and improve model stability**.

The key idea is to train multiple models on different **bootstrap samples** of the training dataset and combine their predictions.

Bootstrap sampling means sampling the training dataset **with replacement**.  
As a result, each model is trained on a slightly different dataset.

This diversity between models helps reduce **overfitting**, especially for high-variance models such as **decision trees**.

Bagging works well when:

• base models have high variance  
• datasets contain noise  
• model stability is required  

---

# Model

Bagging does not define a new prediction model.

Instead it combines predictions from **multiple base models**.

Assume there are **M models**:

h₁(x), h₂(x), … , hₘ(x)

---

## Classification Prediction

The final prediction is the **majority vote**.

ŷ = mode(h₁(x), h₂(x), … , hₘ(x))

Where

| Symbol | Meaning |
|------|------|
| hᵢ(x) | prediction from model i |
| M | number of estimators |

---

## Regression Prediction

For regression the predictions are averaged.

ŷ = (1/M) Σ hᵢ(x)

Where

| Symbol | Meaning |
|------|------|
| hᵢ(x) | prediction of model i |

Averaging reduces variance in the predictions.

---

# Math Implementation

## Bootstrap Sampling

Bootstrap sampling creates multiple training datasets.

Given dataset

D = {x₁, x₂, … , xₙ}

A bootstrap sample is created by randomly selecting **n samples with replacement**.

Example

D₁ = sample(D)  
D₂ = sample(D)  
...  
Dₘ = sample(D)

Each dataset may contain **duplicate samples**.

---

## Ensemble Prediction

After training models

h₁, h₂, … , hₘ

Predictions are combined.

Classification

ŷ = argmax_c Σ I(hᵢ(x) = c)

Where

| Symbol | Meaning |
|------|------|
| I | indicator function |
| c | class label |

---

Regression

ŷ = (1/M) Σ hᵢ(x)

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| model | base learning algorithm |
| n_estimators | number of models in ensemble |

---

### Derived During Training

| Variable | Meaning |
|------|------|
| models | list of trained models |
| indx | bootstrap sample indices |
| new_X | sampled training features |
| new_Y | sampled training targets |

---

### Derived During Prediction

| Variable | Meaning |
|------|------|
| y_pred | predictions from each model |
| final | aggregated predictions |

---

# Loop / Vectorized Form

Bootstrap sampling

```
indices = np.random.choice(n, n, replace=True)
```

This creates a new dataset by sampling indices from the original dataset.

---

Prediction aggregation

Classification

```
np.bincount(labels).argmax()
```

This computes the **majority vote**.

Regression

```
np.mean(predictions)
```

This computes the **average prediction** across models.

---

# Algorithm Steps

## Training

1 Initialize empty list of models.

2 Repeat **n_estimators times**

Create bootstrap sample

Sample n indices with replacement.

Create new dataset

new_X = X[indx]  
new_Y = y[indx]

3 Clone the base model.

4 Train model on the bootstrap dataset.

5 Store the trained model.

---

## Prediction

1 Collect predictions from all models.

2 Combine predictions

Classification

Use majority vote.

Regression

Use mean prediction.

---

# Time Complexity

Training complexity

O(n_estimators × training_cost)

Where

| Symbol | Meaning |
|------|------|
| training_cost | training cost of base model |

---

Prediction complexity

O(n_estimators × prediction_cost)

Where

| Symbol | Meaning |
|------|------|
| prediction_cost | prediction cost of base model |

---

# Code Implementation

## Bagging for Classification

```python
class Bagging:
    def __init__(self,model,n_estimators):
        self.model = model               # base learning model
        self.n_estimators = n_estimators # number of models in ensemble
        self.models = []                 # list to store trained models

    def fit(self,X,y):

        n = X.shape[0]

        for _ in range(self.n_estimators):

            # generate bootstrap sample indices
            indx = np.random.choice(n,n,replace=True)

            # create sampled dataset
            new_X = X[indx]
            new_Y = y[indx]

            # clone base model
            model = deepcopy(self.model)

            # train model
            model.fit(new_X,new_Y)

            # store trained model
            self.models.append(model)

    def predict(self,X):

        # collect predictions from all models
        y_pred = np.array([model.predict(X) for model in self.models])

        final = []

        # majority voting
        for column in y_pred.T:

            final.append(np.bincount(column.astype(int)).argmax())

        return final
```

---

## Bagging for Regression

Regression combines predictions using averaging.

```python
class Bagging:
    def __init__(self,model,n_estimators):
        self.model = model               # base regression model
        self.n_estimators = n_estimators # number of estimators
        self.models = []                 # list of trained models

    def fit(self,X,y):

        n = X.shape[0]

        for _ in range(self.n_estimators):

            # bootstrap sampling
            indx = np.random.choice(n,n,replace=True)

            new_X = X[indx]

            new_Y = y[indx]

            # clone base model
            model = deepcopy(self.model)

            # train model
            model.fit(new_X,new_Y)

            self.models.append(model)

    def predict(self,X):

        # collect predictions from all models
        y_pred = np.array([model.predict(X) for model in self.models])

        # average predictions
        return np.mean(y_pred,axis=0)
```