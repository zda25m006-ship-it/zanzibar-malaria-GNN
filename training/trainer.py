"""
Unified training loop for all GNN models.
Supports both single-graph and temporal sequence models.
"""

import torch
import torch.nn as nn
import numpy as np
from training.losses import get_loss_function


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_single_graph_model(model, train_graphs, val_graphs,
                              lr=0.001, epochs=200, loss_name='combined',
                              patience=15, weight_decay=1e-4, verbose=True):
    """
    Train a single-graph model (GCN or GAT) using per-snapshot training.

    Args:
        model: GCN or GAT model
        train_graphs: list of PyG Data objects for training
        val_graphs: list of PyG Data objects for validation
        lr: learning rate
        epochs: max epochs
        loss_name: name of loss function
        patience: early stopping patience
        weight_decay: L2 regularization
        verbose: print progress

    Returns:
        dict with train_losses, val_losses, best_epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    criterion = get_loss_function(loss_name)
    early_stopping = EarlyStopping(patience=patience)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for data in train_graphs:
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_graphs)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_graphs:
                pred = model(data)
                loss = criterion(pred, data.y)
                val_loss += loss.item()
        avg_val_loss = val_loss / max(len(val_graphs), 1)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if early_stopping.should_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.restore_best(model)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': epoch + 1 - early_stopping.counter,
    }


def train_temporal_model(model, train_sequences, val_sequences,
                          lr=0.001, epochs=200, loss_name='combined',
                          patience=15, weight_decay=1e-4, verbose=True):
    """
    Train a temporal model (ST-GNN or Graph Transformer) using sequences.

    Args:
        model: ST-GNN or GraphTransformer model
        train_sequences: list of (history, target) tuples
        val_sequences: list of (history, target) tuples
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    criterion = get_loss_function(loss_name)
    early_stopping = EarlyStopping(patience=patience)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0

        for history, target in train_sequences:
            optimizer.zero_grad()
            pred = model(history, target)
            loss = criterion(pred, target.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for history, target in val_sequences:
                pred = model(history, target)
                loss = criterion(pred, target.y)
                val_loss += loss.item()
                n_val += 1
        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if early_stopping.should_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.restore_best(model)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': epoch + 1 - early_stopping.counter,
    }


def evaluate_model(model, test_data, model_type='single'):
    """
    Evaluate a model on test data and return predictions.

    Args:
        model: trained model
        test_data: list of Data objects (single) or (history, target) tuples (temporal)
        model_type: 'single' or 'temporal'

    Returns:
        all_preds, all_targets: lists of numpy arrays
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        if model_type == 'single':
            for data in test_data:
                pred = model(data)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(data.y.cpu().numpy())
        else:
            for history, target in test_data:
                pred = model(history, target)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.y.cpu().numpy())

    return all_preds, all_targets
