"""
Loss functions including CVaR-tilted losses.
"""

import torch
import torch.nn as nn
from typing import Optional


class CVaRLoss(nn.Module):
    """
    Conditional Value-at-Risk tilted loss.
    
    L = E[(1 + η * 1{ε² ≥ Q_q(ε²)}) * ε²]
    """

    def __init__(self, quantile: float = 0.95, weight: float = 1.5):
        super().__init__()
        self.quantile = quantile
        self.weight = weight

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        squared_errors = errors ** 2
        quantile_val = torch.quantile(squared_errors.flatten(), self.quantile)
        weights = 1 + self.weight * (squared_errors >= quantile_val).float()
        return torch.mean(weights * squared_errors)


class BSDELoss(nn.Module):
    """
    Combined BSDE loss with terminal, drift, and optional HJB components.
    """

    def __init__(
        self,
        lambda_term: float = 1.0,
        lambda_drift: float = 1.0,
        lambda_hjb: float = 0.0,
        cvar_quantile: float = 0.95,
        cvar_weight: float = 1.5
    ):
        super().__init__()
        self.lambda_term = lambda_term
        self.lambda_drift = lambda_drift
        self.lambda_hjb = lambda_hjb
        self.cvar_loss = CVaRLoss(cvar_quantile, cvar_weight)

    def forward(
        self,
        Y_pred: torch.Tensor,
        Y_true: torch.Tensor,
        drift_residual: torch.Tensor,
        hjb_residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        terminal_error = Y_pred[:, -1] - Y_true
        term_loss = self.cvar_loss(terminal_error)
        drift_loss = torch.mean(drift_residual ** 2)
        total_loss = self.lambda_term * term_loss + self.lambda_drift * drift_loss
        if hjb_residual is not None and self.lambda_hjb > 0:
            hjb_loss = torch.mean(torch.relu(hjb_residual) ** 2)
            total_loss += self.lambda_hjb * hjb_loss
        return total_loss


