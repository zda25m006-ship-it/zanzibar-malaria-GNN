"""
Custom loss functions for malaria count prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PoissonNLLLoss(nn.Module):
    """Poisson negative log-likelihood loss for count data."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: predicted rates (already passed through softplus)
            target: actual counts
        """
        pred = torch.clamp(pred, min=1e-6)
        loss = pred - target * torch.log(pred)
        # Add log(target!) approximation via Stirling for normalization
        return loss.mean()


class NegativeBinomialLoss(nn.Module):
    """
    Negative binomial loss for overdispersed count data.
    Parameterized by mean (mu) and dispersion (alpha).
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-6)
        alpha = self.alpha

        r = 1.0 / alpha
        loss = (
            torch.lgamma(target + r) - torch.lgamma(torch.tensor(r)) - torch.lgamma(target + 1)
            + r * torch.log(torch.tensor(r) / (r + pred))
            + target * torch.log(pred / (r + pred))
        )
        return -loss.mean()


class WeightedImportationLoss(nn.Module):
    """
    Custom loss that gives more weight to imported cases (non-zero targets)
    and Unguja district nodes.
    """

    def __init__(self, num_unguja_nodes: int = 7, unguja_weight: float = 3.0,
                 nonzero_weight: float = 2.0):
        super().__init__()
        self.num_unguja = num_unguja_nodes
        self.unguja_weight = unguja_weight
        self.nonzero_weight = nonzero_weight

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-6)

        # Base Poisson loss per node
        loss = pred - target * torch.log(pred)

        # Weight Unguja nodes more
        weights = torch.ones_like(target)
        weights[:self.num_unguja] = self.unguja_weight

        # Weight non-zero targets more
        nonzero_mask = (target > 0).float()
        weights = weights * (1.0 + nonzero_mask * (self.nonzero_weight - 1.0))

        return (loss * weights).mean()


class CombinedLoss(nn.Module):
    """Combination of Poisson and MSE losses."""

    def __init__(self, poisson_weight: float = 0.7, mse_weight: float = 0.3):
        super().__init__()
        self.poisson = PoissonNLLLoss()
        self.mse = nn.MSELoss()
        self.pw = poisson_weight
        self.mw = mse_weight

    def forward(self, pred, target):
        return self.pw * self.poisson(pred, target) + self.mw * self.mse(pred, target)


def get_loss_function(name: str, **kwargs):
    """Factory function for loss functions."""
    losses = {
        'poisson': PoissonNLLLoss,
        'negative_binomial': NegativeBinomialLoss,
        'mse': nn.MSELoss,
        'weighted': WeightedImportationLoss,
        'combined': CombinedLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    return losses[name](**kwargs)
