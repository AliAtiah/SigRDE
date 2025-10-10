"""
Malliavin weight estimators for Z and Gamma.
"""

import torch
from typing import Tuple, Optional


def compute_malliavin_z(
    Y: torch.Tensor,
    dW: torch.Tensor,
    dt: float,
    sigma_inv: Optional[torch.Tensor] = None
) -> torch.Tensor:
    batch_size, seq_len = Y.shape
    dim = dW.shape[-1]
    Z_malliavin = torch.zeros(batch_size, seq_len - 1, dim, device=Y.device)
    for t in range(seq_len - 1):
        if sigma_inv is not None:
            dW_scaled = torch.bmm(sigma_inv[:, t], dW[:, t].unsqueeze(-1)).squeeze(-1)
        else:
            dW_scaled = dW[:, t]
        Z_malliavin[:, t] = Y[:, t + 1].unsqueeze(-1) * dW_scaled / dt
    return Z_malliavin


def compute_malliavin_gamma(
    Y: torch.Tensor,
    paths: torch.Tensor,
    epsilon: float = 0.01
) -> torch.Tensor:
    batch_size, seq_len, dim = paths.shape
    Gamma = torch.zeros(batch_size, seq_len - 1, dim, dim, device=paths.device)
    # Placeholder symmetric estimator
    for t in range(seq_len - 1):
        gamma_ij = torch.randn(batch_size, device=paths.device) * 0.01
        for i in range(dim):
            for j in range(i, dim):
                Gamma[:, t, i, j] = gamma_ij
                Gamma[:, t, j, i] = gamma_ij
    return Gamma


def compute_antithetic_paths(
    paths: torch.Tensor,
    dW: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    dW_anti = -dW
    batch_size, seq_len, dim = paths.shape
    paths_anti = torch.zeros_like(paths)
    paths_anti[:, 0] = paths[:, 0]
    for t in range(seq_len - 1):
        paths_anti[:, t + 1] = paths_anti[:, t] + dW_anti[:, t] * 0.2
    return paths_anti, dW_anti


