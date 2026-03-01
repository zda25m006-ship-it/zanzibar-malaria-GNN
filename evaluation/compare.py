"""
Model comparison and statistical significance testing.
"""

import numpy as np
from scipy import stats


def compare_models(results: dict):
    """
    Create a comparison table of all models.

    Args:
        results: dict of model_name -> {
            'overall': {metric_name: value},
            'unguja_only': {metric_name: value},
            'predictions': list of arrays,
            'targets': list of arrays,
        }

    Returns:
        comparison_table: formatted string
        best_model: name of best model by RMSE
    """
    print("\n" + "=" * 100)
    print("MODEL COMPARISON - ALL NODES")
    print("=" * 100)
    header = f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Poisson_Dev':>12} {'AUC':>8} {'EW_Acc':>8}"
    print(header)
    print("-" * 100)

    best_rmse = float('inf')
    best_model = None

    for name, res in results.items():
        m = res['overall']
        line = f"{name:<25} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['R²']:>8.4f} {m['Poisson_Dev']:>12.2f} {m['AUC']:>8.4f} {m['Early_Warning_Acc']:>8.4f}"
        print(line)
        if m['RMSE'] < best_rmse:
            best_rmse = m['RMSE']
            best_model = name

    print("\n" + "=" * 100)
    print("MODEL COMPARISON - UNGUJA DISTRICTS ONLY")
    print("=" * 100)
    print(header)
    print("-" * 100)

    for name, res in results.items():
        m = res['unguja_only']
        line = f"{name:<25} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['R²']:>8.4f} {m['Poisson_Dev']:>12.2f} {m['AUC']:>8.4f} {m['Early_Warning_Acc']:>8.4f}"
        print(line)

    print(f"\n>>> Best model (overall RMSE): {best_model}")
    return best_model


def statistical_significance(results: dict, baseline_name: str = 'NB_Regression'):
    """
    Perform statistical significance tests comparing each model against the baseline.
    Uses Wilcoxon signed-rank test on per-time-step errors.
    """
    if baseline_name not in results:
        print(f"Baseline '{baseline_name}' not found in results.")
        return {}

    baseline_preds = results[baseline_name]['predictions']
    baseline_targets = results[baseline_name]['targets']

    # Per-time-step RMSE for baseline
    baseline_errors = []
    for p, t in zip(baseline_preds, baseline_targets):
        baseline_errors.append(np.sqrt(np.mean((p - t) ** 2)))
    baseline_errors = np.array(baseline_errors)

    print("\n" + "=" * 80)
    print(f"STATISTICAL SIGNIFICANCE vs {baseline_name}")
    print("=" * 80)
    print(f"{'Model':<25} {'Mean RMSE':>10} {'p-value':>10} {'Significant':>12} {'Improvement':>12}")
    print("-" * 80)

    sig_results = {}

    for name, res in results.items():
        if name == baseline_name:
            continue

        model_errors = []
        for p, t in zip(res['predictions'], res['targets']):
            model_errors.append(np.sqrt(np.mean((p - t) ** 2)))
        model_errors = np.array(model_errors)

        # Wilcoxon test
        if len(model_errors) >= 3 and not np.all(model_errors == baseline_errors):
            try:
                stat, p_value = stats.wilcoxon(baseline_errors, model_errors,
                                                alternative='greater')
            except Exception:
                p_value = 1.0
        else:
            p_value = 1.0

        improvement = (np.mean(baseline_errors) - np.mean(model_errors)) / max(np.mean(baseline_errors), 1e-8) * 100
        significant = "YES *" if p_value < 0.05 else "No"

        print(f"{name:<25} {np.mean(model_errors):>10.4f} {p_value:>10.4f} {significant:>12} {improvement:>11.1f}%")
        sig_results[name] = {'p_value': p_value, 'improvement_pct': improvement}

    return sig_results
