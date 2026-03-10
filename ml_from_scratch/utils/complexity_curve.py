import numpy as np
import matplotlib.pyplot as plt

from validation import KFold
from score import Score


# bias variance tradeoff

def model_complexity_curve(
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

    scores = []

    for value in param_values:

        # create model with tested parameter
        params = model_kwargs.copy()
        params[param_name] = value

        model = model_class(**params)

        # run K-Fold validation
        kf = KFold(
            model=model,
            score=metric,
            n_splits=k_folds,
            shuffle=shuffle,
            random_state=random_state
        )

        score = kf.cross_validate(X, y)

        scores.append(score)

    scores = np.array(scores)

    # find best parameter
    best_idx = np.argmin(scores)
    best_param = param_values[best_idx]
    best_score = scores[best_idx]

    print("Best Parameter:", param_name, "=", best_param)
    print("Best Score:", best_score)

    # plot curve
    plt.plot(param_values, scores, marker='o')

    plt.xlabel(param_name)
    plt.ylabel(metric.name)
    plt.title("Model Complexity Curve")

    plt.grid(True)

    plt.show()

    return scores