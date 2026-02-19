"""
Leave-one-month-out cross-validation trainer.
Trains a fresh model for each fold, returns per-fold metrics.
"""

import numpy as np
import torch
import torch.nn as nn
import copy



from training.losses import get_loss_function


def train_one_fold(model_class, model_kwargs: dict,
                   train_graphs: list, test_graph,
                   lr: float = 0.005, epochs: int = 200,
                   patience: int = 20, weight_decay: float = 1e-3,
                   num_unguja: int = 7, loss_name: str = 'combined'):
    """
    Train model_class(**model_kwargs) on train_graphs.
    Use last 20% of train_graphs as in-fold validation for early stopping.
    Evaluate on test_graph.

    Returns: dict with predictions, targets, and per-node metrics.
    """
    # In-fold val split: last 3 months (or 20%) as early-stopping monitor
    n_val_in = max(1, len(train_graphs) // 5)
    fold_train = train_graphs[:-n_val_in]
    fold_val   = train_graphs[-n_val_in:]

    if len(fold_train) < 2:
        fold_train = train_graphs
        fold_val   = train_graphs

    model = model_class(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, factor=0.5
    )
    criterion = get_loss_function(loss_name)

    best_val_loss = float('inf')
    best_state    = copy.deepcopy(model.state_dict())
    no_improve    = 0

    model.train()
    for epoch in range(epochs):
        # Training
        epoch_loss = 0.0
        for g in fold_train:
            optimizer.zero_grad()
            pred = model(g)
            loss = criterion(pred, g.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation (in-fold)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for g in fold_val:
                pred = model(g)
                val_loss += criterion(pred, g.y).item()

        val_loss /= max(len(fold_val), 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break
        model.train()

    # Load best
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred = model(test_graph).numpy()

    target = test_graph.y.numpy()
    return {
        'predictions': pred,
        'targets':     target,
        'test_month':  getattr(test_graph, 'year_month', '?'),
    }


def run_loocv(model_class, model_kwargs: dict, folds: list,
              lr: float = 0.005, epochs: int = 200,
              patience: int = 20, weight_decay: float = 1e-3,
              num_unguja: int = 7, verbose: bool = True, loss: str = 'combined'):
    """
    Run leave-one-month-out CV over all folds.
    Returns list of per-fold result dicts and aggregated stats.
    """
    all_results = []
    unguja_rmses, unguja_maes, unguja_aucs = [], [], []

    for fold in folds:
        if verbose:
            print(f"  Fold: test={fold['test_month']}  "
                  f"(train={len(fold['train_graphs'])} months)", end='  ', flush=True)

        result = train_one_fold(
            model_class, model_kwargs,
            fold['train_graphs'], fold['test_graph'],
            lr=lr, epochs=epochs, patience=patience,
            weight_decay=weight_decay, num_unguja=num_unguja,
            loss_name=loss,
        )
        result['test_month'] = fold['test_month']

        # Per-fold metrics (Unguja nodes only)
        pred_u = result['predictions'][:num_unguja]
        targ_u = result['targets'][:num_unguja]

        rmse = float(np.sqrt(np.mean((pred_u - targ_u) ** 2)))
        mae  = float(np.mean(np.abs(pred_u - targ_u)))

        # AUC: binary importation detection
        from sklearn.metrics import roc_auc_score
        binary_true = (targ_u > 0).astype(int)
        if binary_true.sum() > 0 and binary_true.sum() < len(binary_true):
            try:
                auc = roc_auc_score(binary_true, pred_u)
            except Exception:
                auc = float('nan')
        else:
            auc = float('nan')

        result.update({'unguja_rmse': rmse, 'unguja_mae': mae, 'unguja_auc': auc})
        if verbose:
            print(f"RMSE={rmse:.3f}  MAE={mae:.3f}  AUC={auc:.3f}")

        unguja_rmses.append(rmse)
        unguja_maes.append(mae)
        if not np.isnan(auc):
            unguja_aucs.append(auc)
        all_results.append(result)

    summary = {
        'mean_rmse': float(np.mean(unguja_rmses)),
        'std_rmse':  float(np.std(unguja_rmses)),
        'mean_mae':  float(np.mean(unguja_maes)),
        'std_mae':   float(np.std(unguja_maes)),
        'mean_auc':  float(np.mean(unguja_aucs)) if unguja_aucs else float('nan'),
        'std_auc':   float(np.std(unguja_aucs))  if unguja_aucs else float('nan'),
        'fold_rmses': unguja_rmses,
        'fold_aucs':  unguja_aucs,
    }
    return all_results, summary
