import numpy as np
import pandas as pd

class KernelRidgeRegressor:
    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.X = X
        self.y = y
        n_samples = X.shape[0]

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.gaussian_kernel(X[i], X[j])

        K += self.alpha * np.identity(n_samples)

        self.alpha_coef = np.linalg.solve(K, y)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            k_i = np.array([self.gaussian_kernel(self.X[j], X[i]) for j in range(len(self.X))])
            y_pred[i] = np.dot(k_i, self.alpha_coef)

        return y_pred

    def gaussian_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)

    def get_params(self, deep=True):
        return {'alpha': self.alpha, 'gamma': self.gamma}
