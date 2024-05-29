import numpy as np

class RidgeRegressor:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha  
        self.fit_intercept = fit_intercept
        self.coefficients = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        n_samples, n_features = X.shape
        identity_matrix = np.eye(n_features)
        self.coefficients = np.linalg.solve(X.T.dot(X) + self.alpha * identity_matrix, X.T.dot(y))

    def predict(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        return X.dot(self.coefficients)

    def get_params(self, deep=True):        
        return {'alpha': self.alpha, 'fit_intercept': self.fit_intercept}

    def _add_intercept(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
