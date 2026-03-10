# Regularization by Machine Learning Model

Regularization helps **prevent overfitting by controlling model complexity**.  
Different algorithms use different regularization techniques depending on how the model works.

---

# Regularization Chart

| Model | Regularization Type | How It Works | Common Parameters |
|------|------|------|------|
| Linear Regression | L1, L2, ElasticNet | Penalizes large weights | λ (alpha) |
| Polynomial Regression | L1, L2 | Controls large polynomial coefficients | λ |
| Logistic Regression | L1, L2, ElasticNet | Penalizes parameters in cross-entropy loss | λ |
| Softmax Regression | L1, L2 | Same as logistic but multi-class | λ |
| SVM | Margin regularization (L2) | Maximizes margin while controlling weight size | C |
| Naive Bayes | Laplace smoothing | Prevents zero probabilities | α |
| KNN | Neighborhood size | Controls model flexibility | k |
| Decision Tree | Structural constraints | Limits tree growth | max_depth, min_samples |
| Random Forest | Bagging + feature randomness | Reduces variance between trees | n_estimators |
| Bagging | Bootstrap aggregation | Reduces variance by averaging models | n_estimators |
| AdaBoost | Sample weighting | Focuses on difficult samples | learning_rate |
| Gradient Boosting | Shrinkage + tree limits | Prevents aggressive boosting | learning_rate, depth |
| XGBoost | L1, L2 + tree penalties | Regularizes leaf weights and tree complexity | α, λ, γ |
| Neural Networks | L1, L2, Dropout | Penalizes weights and reduces co-adaptation | λ, dropout |

---

# Regularization Mechanisms

## Weight Regularization
Used when models learn **explicit parameters (weights)**.

Models:
- Linear Regression
- Logistic Regression
- Softmax Regression
- SVM
- Neural Networks
- XGBoost

Typical form

```
Loss = TaskLoss + λ||w||² + α|w|
```

---

## Structural Regularization
Used for **tree-based models**.

Models:
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Common techniques

- max_depth  
- min_samples_leaf  
- pruning  

---

## Data-Based Regularization
Uses **data resampling to reduce variance**.

Models:
- Bagging
- Random Forest

Methods

- bootstrap sampling
- feature randomness

---

## Instance-Based Regularization
Used in **lazy learning algorithms**.

Models:
- KNN

Methods

- k value
- distance weighting
- feature scaling

---

## Probabilistic Regularization
Used for **probability models**.

Models:
- Naive Bayes

Method

```
P(x|y) = (count + α) / (total + αV)
```

---

# Summary

| Regularization Type | Example Models |
|------|------|
Weight penalties | Linear, Logistic, SVM |
Structural limits | Decision Trees |
Resampling | Random Forest, Bagging |
Neighborhood control | KNN |
Probability smoothing | Naive Bayes |
Boosting shrinkage | AdaBoost, Gradient Boosting |
Leaf penalties | XGBoost |

---

# Key Idea

Regularization depends on **what the model learns**:

| Model learns | Regularization |
|------|------|
weights | L1 / L2 |
tree structure | depth / pruning |
neighbors | k |
probabilities | smoothing |
ensembles | bagging |
boosted trees | shrinkage + penalties |