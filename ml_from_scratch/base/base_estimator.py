class BaseEstimator:
    """
    Base class for all models
    """

    def fit(self, X, y):
        raise NotImplementedError("fit() must be implemented")

    def predict(self, X):
        raise NotImplementedError("predict() must be implemented")