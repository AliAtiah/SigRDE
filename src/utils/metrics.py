"""
Metrics and evaluation helpers.
"""

import torch
import numpy as np
from typing import Dict


def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict[str, float]:
    """
    Compute common pricing metrics.
    """
    pred = pred.detach().cpu().flatten()
    true = true.detach().cpu().flatten()
    errors = (pred - true).abs().numpy()
    rel_errors = errors / (np.abs(true.numpy()) + 1e-6)
    return {
        'mae': float(np.mean(errors)),
        'rpe': float(np.mean(rel_errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'cvar_90': float(np.mean(errors[errors >= np.quantile(errors, 0.90)])),
        'cvar_95': float(np.mean(errors[errors >= np.quantile(errors, 0.95)])),
        'cvar_99': float(np.mean(errors[errors >= np.quantile(errors, 0.99)])),
    }


