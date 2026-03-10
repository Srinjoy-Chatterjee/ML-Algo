import numpy as np
from copy import deepcopy


class KFold:

    def __init__(self, model, score, n_splits=5, shuffle=True, random_state=None):

        self.model = model                # base ML model
        self.score = score                # scoring function or Score enum
        self.n_splits = n_splits          # number of folds
        self.shuffle = shuffle            # shuffle dataset before split
        self.random_state = random_state  # random seed


    # ==========================
    # Split indices into folds
    # ==========================
    def split(self, n):

        indices = np.arange(n)

        if self.shuffle:

            if self.random_state is not None:
                np.random.seed(self.random_state)

            np.random.shuffle(indices)

        # compute fold sizes
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


    # ==========================
    # Cross Validation
    # ==========================
    def cross_validate(self, X, y, return_scores=False):

        n = X.shape[0]

        folds = self.split(n)

        scores = []

        for i in range(self.n_splits):

            # validation fold
            test_idx = folds[i]

            # training folds
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

            # support Score enum or normal function
            if hasattr(self.score, "value"):
                score = self.score.value(y_test, y_pred)
            else:
                score = self.score(y_test, y_pred)

            scores.append(score)

        scores = np.array(scores)

        if return_scores:
            return scores

        return np.mean(scores)