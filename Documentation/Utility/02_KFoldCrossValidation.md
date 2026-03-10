# Algorithm
K-Fold Cross Validation

K-Fold Cross Validation is a model evaluation technique used to estimate the **generalization performance of a machine learning model**.

Instead of splitting the dataset into a single training and testing set, the data is divided into **K equal subsets called folds**.

The model is trained and evaluated **K times**.

Each time:

• **K − 1 folds are used for training**  
• **1 fold is used for validation**

Each sample is used **once as validation data** and **K−1 times as training data**.

The final performance is computed as the **average score across all folds**.

This approach reduces variance in model evaluation compared to a single train-test split.

K-Fold Cross Validation is commonly used in:

• model evaluation  
• hyperparameter tuning  
• bias–variance analysis  

---

# Model

Let the dataset be

D = {(x₁,y₁),(x₂,y₂),...,(xₙ,yₙ)}

The dataset is divided into **K folds**

D₁ , D₂ , ... , Dₖ

For each fold i

Training dataset

D_train = D \ Dᵢ

Validation dataset

D_test = Dᵢ

The model is trained

fᵢ = train(D_train)

Prediction is evaluated

ŷ = fᵢ(x)

Score of fold i

Sᵢ = metric(y_test , ŷ)

Final cross validation score

S = (1/K) Σ Sᵢ

Where

| Symbol | Meaning |
|------|------|
| K | number of folds |
| Sᵢ | score of fold i |
| S | average validation score |

---

# Math Implementation

Let

| Symbol | Meaning |
|------|------|
| X | feature matrix |
| y | target vector |
| n | number of samples |
| K | number of folds |

Step 1: Generate dataset indices

indices = {0,1,2,...,n−1}

Step 2: Shuffle indices (optional)

Step 3: Partition indices into K folds

F₁ , F₂ , ... , Fₖ

For each fold i

Validation set

test_idx = Fᵢ

Training set

train_idx = all_indices − Fᵢ

Training data

X_train = X[train_idx]

y_train = y[train_idx]

Validation data

X_test = X[test_idx]

y_test = y[test_idx]

Train model

model.fit(X_train , y_train)

Predict

ŷ = model.predict(X_test)

Compute score

Sᵢ = metric(y_test , ŷ)

Final score

S = (1/K) Σ Sᵢ

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| model | machine learning model |
| score | evaluation metric |
| n_splits | number of folds |
| shuffle | whether dataset is shuffled |
| random_state | random seed |

### Derived During Execution

| Variable | Meaning |
|------|------|
| indices | dataset indices |
| folds | list of fold index arrays |
| train_idx | training indices |
| test_idx | validation indices |
| scores | validation scores for each fold |

---

# Loop / Vectorized Form

The implementation iterates over folds.

Pseudo structure

```
for i in range(K):

    test_idx = folds[i]

    train_idx = concatenate(all folds except i)

    model.fit(X_train , y_train)

    y_pred = model.predict(X_test)

    score_i = metric(y_test , y_pred)
```

Final score

```
mean_score = mean(score_i)
```

---

# Algorithm Steps

1. Create array of dataset indices

indices = {0,1,2,...,n−1}

2. Shuffle indices if shuffle=True

3. Split indices into K folds

4. For each fold i

Select validation fold

test_idx = folds[i]

Select training folds

train_idx = concatenate(all folds except i)

Train model

model.fit(X_train , y_train)

Predict

y_pred = model.predict(X_test)

Compute score

score_i = metric(y_test , y_pred)

5. Store validation score

6. Repeat for all folds

7. Return mean validation score

---

# Time Complexity

Let

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |
| K | number of folds |

Training complexity

O(K × model_training_cost)

Prediction complexity

O(K × model_prediction_cost)

Total complexity increases **linearly with the number of folds**.

---

# Code Implementation

```python
import numpy as np
from copy import deepcopy


class KFold:

    def __init__(self, model, score, n_splits=5, shuffle=True, random_state=None):

        self.model = model                # base ML model
        self.score = score                # scoring function or Score enum
        self.n_splits = n_splits          # number of folds
        self.shuffle = shuffle            # shuffle dataset before split
        self.random_state = random_state  # random seed


    def split(self, n):

        indices = np.arange(n)

        if self.shuffle:

            if self.random_state is not None:
                np.random.seed(self.random_state)

            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n // self.n_splits)

        fold_sizes[:n % self.n_splits] += 1

        folds = []

        current = 0

        for fold_size in fold_sizes:

            start = current
            stop = current + fold_size

            folds.append(indices[start:stop])

            current = stop

        return folds


    def cross_validate(self, X, y, return_scores=False):

        n = X.shape[0]

        folds = self.split(n)

        scores = []

        for i in range(self.n_splits):

            test_idx = folds[i]

            train_idx = np.hstack(
                [folds[j] for j in range(self.n_splits) if j != i]
            )

            X_train = X[train_idx]
            y_train = y[train_idx]

            X_test = X[test_idx]
            y_test = y[test_idx]

            model = deepcopy(self.model)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if hasattr(self.score, "value"):
                score = self.score.value(y_test, y_pred)
            else:
                score = self.score(y_test, y_pred)

            scores.append(score)

        scores = np.array(scores)

        if return_scores:
            return scores

        return np.mean(scores)
```