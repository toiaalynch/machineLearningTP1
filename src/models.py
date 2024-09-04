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
    def __init__(self, tau):
        self.tau = tau
    
    def _weight_matrix(self, x, X_train):
        m = X_train.shape[0]
        # Matriz de pesos (W) calculada en función de las distancias
        W = np.exp(-np.sum((X_train - x) ** 2, axis=1) / (2 * self.tau ** 2))
        return np.diag(W)
    
    def _loss(self, theta, X_b, W, y_train):
        predictions = X_b.dot(theta)
        weighted_error = W.dot(predictions - y_train)
        return np.sum(weighted_error ** 2)

    def predict(self, x, X_train, y_train):
        # Agregar sesgo (bias) a los datos de entrenamiento
        X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        x_b = np.r_[1, x]  # Sesgo para la instancia actual
        W = self._weight_matrix(x_b, X_b)  # Matriz de pesos
        
        # Optimizar los coeficientes (theta) para esta predicción específica
        initial_theta = np.random.randn(X_b.shape[1])
        
        # Usar scipy.optimize para minimizar la pérdida local
        result = minimize(self._loss, initial_theta, args=(X_b, W, y_train), method='BFGS')
        
        # Predicción con los parámetros optimizados
        theta_opt = result.x
        return x_b.dot(theta_opt)



class NonLinearRegression:
    def __init__(self):
        self.theta = None  # Inicializamos los parámetros a ajustar (a y b)
    
    def fit(self, X, y):
        # Tomar el logaritmo natural de X para transformar la variable
        X_log = np.log(X)
        
        # Agregar una columna de 1s para el término de sesgo (bias)
        X_b = np.c_[np.ones((X_log.shape[0], 1)), X_log]
        
        # Resolver usando la ecuación normal para ajustar a y b
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        # Agregar una columna de 1s para el término de sesgo (bias)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calcular las predicciones en el espacio logarítmico
        y_log_pred = X_b.dot(self.theta)
        
        # Volver al espacio original exponenciando los resultados
        return np.exp(y_log_pred)
