"""
Evaluation metrics for malaria importation prediction.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def r_squared(y_true, y_pred):
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-8)


def poisson_deviance(y_true, y_pred):
    """Poisson deviance statistic."""
    y_true = np.maximum(y_true, 1e-8)
    y_pred = np.maximum(y_pred, 1e-8)
    return 2 * np.sum(y_true * np.log(y_true / y_pred) - (y_true - y_pred))


def classification_auc(y_true, y_pred, threshold=0.0):
    """
    AUC for importation detection (binary: any imported cases vs none).
    """
    y_binary = (y_true > threshold).astype(int)
    if len(np.unique(y_binary)) < 2:
        return 0.5  # Cannot compute AUC with single class
    return roc_auc_score(y_binary, y_pred)


def early_warning_accuracy(y_true, y_pred, threshold_percentile=75):
    """
    Accuracy of early warning: does the model correctly identify
    months with above-average importation?
    """
    threshold = np.percentile(y_true, threshold_percentile)
    y_high = (y_true >= threshold).astype(int)
    y_pred_high = (y_pred >= threshold).astype(int)
    return np.mean(y_high == y_pred_high)


def compute_all_metrics(y_true, y_pred, unguja_mask=None):
    """
    Compute all metrics. Optionally filter to Unguja district nodes only.

    Returns:
        dict of metric_name -> value
    """
    if unguja_mask is not None:
        y_true = y_true[unguja_mask]
        y_pred = y_pred[unguja_mask]

    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'R²': r_squared(y_true, y_pred),
        'Poisson_Dev': poisson_deviance(y_true, y_pred),
        'AUC': classification_auc(y_true, y_pred),
        'Early_Warning_Acc': early_warning_accuracy(y_true, y_pred),
    }
    return metrics


def aggregate_metrics(all_preds, all_targets, num_unguja=7):
    """
    Aggregate metrics across all test time steps.

    Args:
        all_preds: list of [N] prediction arrays
        all_targets: list of [N] target arrays
        num_unguja: number of Unguja district nodes

    Returns:
        dict with 'overall' and 'unguja_only' metric dicts
    """
    # Flatten all predictions and targets
    preds_flat = np.concatenate(all_preds)
    targets_flat = np.concatenate(all_targets)

    # Unguja-only mask (first num_unguja nodes per time step)
    unguja_preds = []
    unguja_targets = []
    for p, t in zip(all_preds, all_targets):
        unguja_preds.append(p[:num_unguja])
        unguja_targets.append(t[:num_unguja])
    unguja_preds = np.concatenate(unguja_preds)
    unguja_targets = np.concatenate(unguja_targets)

    return {
        'overall': compute_all_metrics(targets_flat, preds_flat),
        'unguja_only': compute_all_metrics(unguja_targets, unguja_preds),
    }
