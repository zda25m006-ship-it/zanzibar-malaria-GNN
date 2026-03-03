import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def calculate_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)


def calculate_poisson_deviance(y_true, y_pred):
    return np.sum(2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred)))


# Example usage:
# y_true = [0, 1, 0, 1]
# y_pred = [0.1, 0.4, 0.35, 0.8]
# y_scores = [0.1, 0.6, 0.4, 0.9]
# print(calculate_rmse(y_true, y_pred))
# print(calculate_mae(y_true, y_pred))
# print(calculate_auc(y_true, y_scores))
# print(calculate_poisson_deviance(y_true, y_pred))