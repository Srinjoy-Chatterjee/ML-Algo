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

        # KFold object
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

    # best parameter based on validation score
    best_idx = np.argmin(val_scores)

    print("Best Parameter:", param_name, "=", param_values[best_idx])
    print("Best Validation Score:", val_scores[best_idx])

    # Plot Bias-Variance Curve
    plt.plot(param_values, train_scores, marker='o', label="Training Error")
    plt.plot(param_values, val_scores, marker='o', label="Validation Error")

    plt.xlabel(param_name)
    plt.ylabel(metric.name)

    plt.title("Bias Variance Tradeoff")

    plt.legend()
    plt.grid(True)

    plt.show()

    return train_scores, val_scores