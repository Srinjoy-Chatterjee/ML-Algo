import numpy as np
class Kfold:
    def __init__(self,model,score,n_splits=5,shuffle=True,random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.model = model
        self.score = score
        return self.split
    
    def split(self,n):
        indices = np.arange(n)
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
        fold_size = n // self.n_splits
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n
            folds.append(indices[start:end])
        return folds

    def cross_validate(self,X,y):
        n = len(X)
        folds = self.split(n)
        scores = []
        for i in range(self.n_splits):
            train_indices = np.hstack([folds[j] for j in range(self.n_splits) if j != i])
            test_indices = folds[i]
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score = self.score(y_test, y_pred)
            scores.append(score)
        return np.mean(scores)