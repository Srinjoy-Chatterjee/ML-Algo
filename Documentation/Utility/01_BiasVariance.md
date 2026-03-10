# Algorithm
Bias–Variance Tradeoff

Bias–Variance Tradeoff is used to understand how **model complexity affects prediction error**.

Machine learning models can suffer from two types of error:

• **Bias** – error due to overly simple assumptions in the model  
• **Variance** – error due to high sensitivity to training data

As model complexity increases:

• **Bias decreases**  
• **Variance increases**

This creates a tradeoff between **underfitting** and **overfitting**.

Typical behavior

| Model Complexity | Bias | Variance | Result |
|------|------|------|------|
Low | High | Low | Underfitting |
Medium | Balanced | Balanced | Optimal |
High | Low | High | Overfitting |

Bias–Variance analysis helps identify the **optimal model complexity**.

Examples of complexity parameters:

• polynomial degree  
• tree depth  
• number of estimators  
• regularization strength  

---

# Model

The expected prediction error of a model can be decomposed into three components.

E[(y − f̂(x))²] = Bias² + Variance + Noise

Where

| Symbol | Meaning |
|------|------|
| y | true value |
| f̂(x) | predicted value |
| Bias² | error due to simplified model assumptions |
| Variance | error due to sensitivity to training data |
| Noise | irreducible error |

Interpretation

• **High Bias → underfitting**  
• **High Variance → overfitting**

The optimal model minimizes the **total expected error**.

---

# Math Implementation

Bias–Variance analysis evaluates the model using **different values of a complexity parameter**.

Example: Polynomial Regression

Prediction model

ŷ = X_poly θ

Where

| Symbol | Meaning |
|------|------|
| X_poly | polynomial feature matrix |
| θ | parameter vector |

Model error is measured using a metric such as **Mean Squared Error (MSE)**.

MSE

MSE = (1/n) Σ (y − ŷ)²

Training error approximates **bias**, while validation error reflects **generalization performance**.

To estimate validation error, **K-Fold Cross Validation** is used.

---

# Variables

### User Provided

| Variable | Meaning |
|------|------|
| model_class | machine learning model class |
| param_name | complexity parameter name |
| param_values | list of complexity values |
| X | feature matrix |
| y | target values |
| metric | evaluation metric |
| k_folds | number of cross validation folds |
| shuffle | whether dataset is shuffled |
| random_state | random seed |

### Derived During Execution

| Variable | Meaning |
|------|------|
| train_scores | training error for each complexity value |
| val_scores | validation error for each complexity value |
| folds | K-Fold split indices |
| train_indices | indices used for training |
| val_indices | indices used for validation |

---

# Loop / Vectorized Form

The algorithm evaluates model performance for different complexity levels.

Pseudo structure

```
for value in param_values:

    create model with parameter value

    perform K-Fold cross validation

    compute training error

    compute validation error
```

Errors are averaged across folds.

---

# Algorithm Steps

1. Select a machine learning model

2. Choose a **complexity parameter**

Example

degree (polynomial regression)  
depth (decision tree)

3. Define a range of parameter values

4. For each parameter value

Initialize model with given complexity

model = model_class(param=value)

5. Perform K-Fold Cross Validation

Split dataset into K folds

6. For each fold

Train model on training folds

model.fit(X_train , y_train)

Predict training data

y_train_pred = model.predict(X_train)

Predict validation data

y_val_pred = model.predict(X_val)

Compute errors

train_error = metric(y_train , y_train_pred)

val_error = metric(y_val , y_val_pred)

7. Store average errors

8. Identify parameter with lowest validation error

9. Plot training error and validation error curves

---

# Time Complexity

Let

| Symbol | Meaning |
|------|------|
| n | number of samples |
| d | number of features |
| p | number of parameter values |
| k | number of folds |

Training complexity

O(p × k × model_training_cost)

Prediction complexity

O(p × k × model_prediction_cost)

Total complexity increases with the number of **parameter values** and **folds**.

---

# Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

from validation import KFold
from score import Score


def bias_variance_tradeoff(
        model_class,
        param_name,
        param_values,
        X,
        y,
        metric=Score.MSE,
        k_folds=5,
        shuffle=True,
        random_state=42,
        **model_kwargs
):

    train_scores = []
    val_scores = []

    for value in param_values:

        params = model_kwargs.copy()
        params[param_name] = value

        model = model_class(**params)

        kf = KFold(
            model=model,
            score=metric,
            n_splits=k_folds,
            shuffle=shuffle,
            random_state=random_state
        )

        folds = kf.split(len(X))

        fold_train_scores = []
        fold_val_scores = []

        for i in range(k_folds):

            train_indices = np.hstack(
                [folds[j] for j in range(k_folds) if j != i]
            )

            val_indices = folds[i]

            X_train = X[train_indices]
            y_train = y[train_indices]

            X_val = X[val_indices]
            y_val = y[val_indices]

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            train_score = metric(y_train, y_train_pred)
            val_score = metric(y_val, y_val_pred)

            fold_train_scores.append(train_score)
            fold_val_scores.append(val_score)

        train_scores.append(np.mean(fold_train_scores))
        val_scores.append(np.mean(fold_val_scores))

    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)

    best_idx = np.argmin(val_scores)

    print("Best Parameter:", param_name, "=", param_values[best_idx])
    print("Best Validation Score:", val_scores[best_idx])

    plt.plot(param_values, train_scores, marker='o', label="Training Error")
    plt.plot(param_values, val_scores, marker='o', label="Validation Error")

    plt.xlabel(param_name)
    plt.ylabel(metric.name)

    plt.title("Bias Variance Tradeoff")

    plt.legend()
    plt.grid(True)

    plt.show()

    return train_scores, val_scores
```