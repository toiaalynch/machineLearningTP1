import numpy as np
from numpy import mean, sqrt

def rmse(y_true, y_pred):
    return sqrt(mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    y_mean = mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)
