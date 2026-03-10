# Hyperparameter Tuning by Machine Learning Model

Hyperparameter tuning is the process of selecting **optimal values for parameters that control the learning process of a model**.

Unlike model parameters (weights), hyperparameters are **set before training** and influence:

• model complexity  
• training behavior  
• regularization strength  

Choosing good hyperparameters improves **generalization and prevents overfitting**.

---

# Hyperparameter Chart

| Model | Important Hyperparameters | Purpose |
|------|------|------|
| Linear Regression | learning_rate | Controls gradient descent updates |
| Polynomial Regression | degree | Controls model complexity |
| Logistic Regression | learning_rate, regularization λ | Controls optimization and weight penalty |
| Softmax Regression | learning_rate, regularization λ | Same as logistic for multi-class |
| SVM | C, kernel | Controls margin and decision boundary |
| Naive Bayes | smoothing α | Prevents zero probabilities |
| KNN | k | Controls neighborhood size |
| Decision Tree | max_depth, min_samples | Limits tree growth |
| Random Forest | n_estimators, max_features | Controls ensemble size and randomness |
| Bagging | n_estimators | Number of bootstrap models |
| AdaBoost | n_estimators, learning_rate | Controls boosting strength |
| Gradient Boosting | n_estimators, learning_rate, depth | Controls boosting complexity |
| XGBoost | learning_rate, depth, λ, α | Controls boosting and regularization |
| Neural Networks | learning_rate, batch_size, layers | Controls training dynamics |

---

# Hyperparameter Mechanisms

## Optimization Control

Hyperparameters that control **training behavior**.

Examples:

| Model | Hyperparameters |
|------|------|
Linear Regression | learning_rate |
Logistic Regression | learning_rate |
Neural Networks | learning_rate, batch_size |

Purpose:

- stabilize training
- control gradient updates

---

## Model Complexity Control

Hyperparameters that control **model flexibility**.

Examples:

| Model | Hyperparameters |
|------|------|
Polynomial Regression | degree |
KNN | k |
Decision Tree | max_depth |

Purpose:

- prevent overfitting
- control bias–variance tradeoff

---

## Ensemble Control

Hyperparameters controlling **ensemble models**.

Examples:

| Model | Hyperparameters |
|------|------|
Random Forest | n_estimators |
Bagging | n_estimators |
Boosting | learning_rate, n_estimators |

Purpose:

- stabilize predictions
- reduce variance

---

## Probability Smoothing

Used in **probabilistic models**.

Example:

| Model | Hyperparameter |
|------|------|
Naive Bayes | α |

Formula

```
P(x|y) = (count + α) / (total + αV)
```

---

# Summary

| Hyperparameter Type | Example Models |
|------|------|
Optimization | Linear, Logistic, Neural Networks |
Complexity control | Polynomial, KNN, Trees |
Ensemble size | Random Forest, Bagging |
Boosting strength | AdaBoost, Gradient Boosting |
Probability smoothing | Naive Bayes |

---

# Relationship Between Hyperparameter Tuning and Regularization

Hyperparameter tuning and regularization are closely related.

Regularization methods usually introduce **hyperparameters that control model complexity**.

Examples:

| Regularization | Hyperparameter |
|------|------|
L1 / L2 regularization | λ |
Tree depth control | max_depth |
KNN neighborhood | k |
Boosting shrinkage | learning_rate |
Naive Bayes smoothing | α |

In practice:

```
Regularization strength = hyperparameter value
```

Therefore hyperparameter tuning is used to **find the optimal level of regularization**.

---

# Key Insight

| Concept | Role |
|------|------|
Hyperparameters | Control model behavior |
Regularization | Prevent overfitting |
Hyperparameter tuning | Finds best regularization strength |

Together they ensure the model achieves **good generalization on unseen data**.