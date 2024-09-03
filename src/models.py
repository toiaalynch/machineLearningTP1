import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.theta = None
    
    def fit(self, X, y):
        # Agregar un término de sesgo (bias) a X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Resolver la ecuación normal con regularización L2
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0  # No regularizar el término de sesgo
        self.theta = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * I).dot(X_b.T).dot(y)
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)



class LocallyWeightedRegression:
    def __init__(self, tau=1.0):
        self.tau = tau
    
    def _weight_matrix(self, x, X_train):
        m = X_train.shape[0]
        W = np.exp(-np.sum((X_train - x) ** 2, axis=1) / (2 * self.tau ** 2))
        return np.diag(W)
    
    def predict(self, x, X_train, y_train):
        X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        x_b = np.r_[1, x]
        W = self._weight_matrix(x_b, X_b)
        XTWX = X_b.T.dot(W).dot(X_b)
        theta = np.linalg.pinv(XTWX).dot(X_b.T).dot(W).dot(y_train)
        return x_b.dot(theta)


class NonLinearRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.theta = None
    
    def _polynomial_features(self, X):
        m, n = X.shape
        X_poly = np.ones((m, 1))
        for i in range(1, self.degree + 1):
            for j in range(n):
                X_poly = np.c_[X_poly, X[:, j] ** i]
        return X_poly
    
    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        self.theta = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
    
    def predict(self, X):
        X_poly = self._polynomial_features(X)
        return X_poly.dot(self.theta)
