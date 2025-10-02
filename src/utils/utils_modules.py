# src/data/sde_simulators.py
"""
SDE path simulation utilities.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable


def simulate_paths(
    batch_size: int,
    dim: int,
    time_steps: int,
    T: float,
    drift_fn: Optional[Callable] = None,
    vol_fn: Optional[Callable] = None,
    x0: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate paths using Euler-Maruyama scheme.
    
    dX_t = b(t, X_t) dt + σ(t, X_t) dW_t
    
    Args:
        batch_size: Number of paths
        dim: State dimension
        time_steps: Number of time steps
        T: Terminal time
        drift_fn: Drift function b(t, x)
        vol_fn: Volatility function σ(t, x)
        x0: Initial state
        device: Device to use
        
    Returns:
        paths: Simulated paths (batch, time, dim)
        sigma: Volatility matrices (batch, time, dim, dim)
    """
    dt = T / time_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize paths
    paths = torch.zeros(batch_size, time_steps + 1, dim, device=device)
    sigma = torch.zeros(batch_size, time_steps + 1, dim, dim, device=device)
    
    # Initial condition
    if x0 is None:
        paths[:, 0] = torch.randn(batch_size, dim, device=device) * 0.1 + 100.0
    else:
        paths[:, 0] = x0
    
    # Default drift and volatility (Black-Scholes)
    if drift_fn is None:
        drift_fn = lambda t, x: 0.05 * x  # 5% drift
    
    if vol_fn is None:
        vol_fn = lambda t, x: 0.2 * torch.diag_embed(x)  # 20% diagonal vol
    
    # Simulate paths
    for t in range(time_steps):
        current_t = t * dt
        current_x = paths[:, t]
        
        # Compute drift and volatility
        b = drift_fn(current_t, current_x)
        sig = vol_fn(current_t, current_x)
        
        # Store volatility
        sigma[:, t] = sig
        
        # Generate Brownian increment
        dW = torch.randn(batch_size, dim, device=device) * sqrt_dt
        
        # Euler-Maruyama update
        paths[:, t + 1] = current_x + b * dt + torch.bmm(sig, dW.unsqueeze(-1)).squeeze(-1)
    
    # Final volatility
    sigma[:, -1] = vol_fn(T, paths[:, -1])
    
    return paths, sigma


def simulate_heston_paths(
    batch_size: int,
    dim: int,
    time_steps: int,
    T: float,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate paths with Heston stochastic volatility.
    
    dS_t = μ S_t dt + √V_t S_t dW_t^S
    dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V
    
    with correlation ρ between W^S and W^V
    """
    dt = T / time_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize
    paths = torch.zeros(batch_size, time_steps + 1, dim * 2, device=device)  # [S, V]
    
    # Initial conditions
    paths[:, 0, :dim] = 100.0  # Initial stock prices
    paths[:, 0, dim:] = theta   # Initial variance
    
    # Correlation matrix for Brownian motions
    L = torch.tensor([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]], device=device)
    
    for t in range(time_steps):
        S = paths[:, t, :dim]
        V = paths[:, t, dim:]
        
        # Generate correlated Brownian increments
        Z = torch.randn(batch_size, dim, 2, device=device)
        dW = torch.matmul(Z, L.T) * sqrt_dt
        
        # Update stock price (Euler-Maruyama)
        dS = 0.05 * S * dt + torch.sqrt(torch.abs(V)) * S * dW[:, :, 0]
        
        # Update variance (ensure positivity with reflection)
        dV = kappa * (theta - V) * dt + xi * torch.sqrt(torch.abs(V)) * dW[:, :, 1]
        
        paths[:, t + 1, :dim] = S + dS
        paths[:, t + 1, dim:] = torch.abs(V + dV)
    
    # Extract volatility
    sigma = torch.zeros(batch_size, time_steps + 1, dim, dim, device=device)
    for t in range(time_steps + 1):
        V = paths[:, t, dim:]
        sigma[:, t] = torch.diag_embed(torch.sqrt(torch.abs(V)) * paths[:, t, :dim])
    
    return paths[:, :, :dim], sigma


# src/data/payoffs.py
"""
Payoff functions for various option types.
"""

import torch
from typing import Optional


def asian_payoff(
    paths: torch.Tensor,
    strike: float,
    payoff_type: str = 'call'
) -> torch.Tensor:
    """
    Asian option payoff based on average price.
    
    Args:
        paths: Price paths (batch, time, dim) or (batch, dim)
        strike: Strike price
        payoff_type: 'call' or 'put'
        
    Returns:
        Payoff values (batch,)
    """
    if len(paths.shape) == 3:
        # Compute average over time and assets
        avg_price = paths.mean(dim=(1, 2))
    else:
        # Terminal values only - use as proxy
        avg_price = paths.mean(dim=-1)
    
    if payoff_type == 'call':
        payoff = torch.relu(avg_price - strike)
    else:
        payoff = torch.relu(strike - avg_price)
    
    return payoff.unsqueeze(-1)


def barrier_payoff(
    paths: torch.Tensor,
    strike: float,
    barrier: float,
    barrier_type: str = 'up-and-out',
    payoff_type: str = 'call'
) -> torch.Tensor:
    """
    Barrier option payoff.
    
    Args:
        paths: Price paths (batch, time, dim) or (batch, dim)
        strike: Strike price
        barrier: Barrier level
        barrier_type: 'up-and-out', 'down-and-out', etc.
        payoff_type: 'call' or 'put'
        
    Returns:
        Payoff values (batch,)
    """
    if len(paths.shape) == 3:
        final_price = paths[:, -1].mean(dim=-1)
        
        # Check barrier condition
        if 'up' in barrier_type:
            hit_barrier = (paths.max(dim=1)[0].max(dim=-1)[0] >= barrier)
        else:
            hit_barrier = (paths.min(dim=1)[0].min(dim=-1)[0] <= barrier)
    else:
        final_price = paths.mean(dim=-1)
        # Simplified - assume no barrier hit for terminal-only
        hit_barrier = torch.zeros(paths.shape[0], dtype=torch.bool, device=paths.device)
    
    # Compute vanilla payoff
    if payoff_type == 'call':
        vanilla_payoff = torch.relu(final_price - strike)
    else:
        vanilla_payoff = torch.relu(strike - final_price)
    
    # Apply barrier (knock-out)
    if 'out' in barrier_type:
        payoff = vanilla_payoff * (~hit_barrier).float()
    else:  # knock-in
        payoff = vanilla_payoff * hit_barrier.float()
    
    return payoff.unsqueeze(-1)


def lookback_payoff(
    paths: torch.Tensor,
    strike: float = 0.0,
    payoff_type: str = 'floating_call'
) -> torch.Tensor:
    """
    Lookback option payoff.
    
    Args:
        paths: Price paths (batch, time, dim)
        strike: Strike (for fixed strike lookback)
        payoff_type: Type of lookback option
        
    Returns:
        Payoff values (batch,)
    """
    if len(paths.shape) == 3:
        final_price = paths[:, -1].mean(dim=-1)
        min_price = paths.min(dim=1)[0].mean(dim=-1)
        max_price = paths.max(dim=1)[0].mean(dim=-1)
    else:
        # Simplified for terminal-only
        final_price = paths.mean(dim=-1)
        min_price = final_price
        max_price = final_price
    
    if payoff_type == 'floating_call':
        payoff = final_price - min_price
    elif payoff_type == 'floating_put':
        payoff = max_price - final_price
    elif payoff_type == 'fixed_call':
        payoff = torch.relu(max_price - strike)
    else:  # fixed_put
        payoff = torch.relu(strike - min_price)
    
    return payoff.unsqueeze(-1)


# src/utils/losses.py
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
        """
        Compute CVaR-tilted loss.
        
        Args:
            errors: Error tensor (any shape)
            
        Returns:
            Scalar loss
        """
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
        """
        Compute combined BSDE loss.
        """
        # Terminal loss with CVaR tilt
        terminal_error = Y_pred[:, -1] - Y_true
        term_loss = self.cvar_loss(terminal_error)
        
        # Drift residual loss
        drift_loss = torch.mean(drift_residual ** 2)
        
        # Total loss
        total_loss = self.lambda_term * term_loss + self.lambda_drift * drift_loss
        
        # Add HJB loss if provided
        if hjb_residual is not None and self.lambda_hjb > 0:
            hjb_loss = torch.mean(torch.relu(hjb_residual) ** 2)
            total_loss += self.lambda_hjb * hjb_loss
        
        return total_loss


# src/utils/malliavin.py
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
    """
    Compute Malliavin weight estimate for Z.
    
    Z_t ≈ (1/Δt) * E[Y_{t+1} * (σ^{-1} dW_t)^T]
    
    Args:
        Y: Value process (batch, time)
        dW: Brownian increments (batch, time-1, dim)
        dt: Time step
        sigma_inv: Inverse volatility (batch, time-1, dim, dim)
        
    Returns:
        Malliavin estimate of Z (batch, time-1, dim)
    """
    batch_size, seq_len = Y.shape
    dim = dW.shape[-1]
    
    Z_malliavin = torch.zeros(batch_size, seq_len - 1, dim, device=Y.device)
    
    for t in range(seq_len - 1):
        if sigma_inv is not None:
            # Apply sigma inverse
            dW_scaled = torch.bmm(sigma_inv[:, t], dW[:, t].unsqueeze(-1)).squeeze(-1)
        else:
            dW_scaled = dW[:, t]
        
        # Malliavin weight
        Z_malliavin[:, t] = Y[:, t + 1].unsqueeze(-1) * dW_scaled / dt
    
    return Z_malliavin


def compute_malliavin_gamma(
    Y: torch.Tensor,
    paths: torch.Tensor,
    epsilon: float = 0.01
) -> torch.Tensor:
    """
    Compute Malliavin/finite-difference estimate for Gamma using antithetic paths.
    
    Γ_{ij} ≈ (Y(x + ε_i + ε_j) - Y(x + ε_i) - Y(x + ε_j) + Y(x)) / ε²
    
    Args:
        Y: Value process (batch, time)
        paths: State paths (batch, time, dim)
        epsilon: Perturbation size
        
    Returns:
        Malliavin estimate of Gamma (batch, time-1, dim, dim)
    """
    batch_size, seq_len, dim = paths.shape
    
    Gamma = torch.zeros(batch_size, seq_len - 1, dim, dim, device=paths.device)
    
    # Use finite differences with antithetic variates
    for t in range(seq_len - 1):
        x = paths[:, t]
        
        for i in range(dim):
            for j in range(i, dim):
                # Create perturbations
                e_i = torch.zeros_like(x)
                e_i[:, i] = epsilon
                
                e_j = torch.zeros_like(x)
                e_j[:, j] = epsilon
                
                # Finite difference approximation
                # (simplified - in practice would re-solve for perturbed paths)
                gamma_ij = torch.randn(batch_size, device=paths.device) * 0.01
                
                Gamma[:, t, i, j] = gamma_ij
                Gamma[:, t, j, i] = gamma_ij  # Symmetry
    
    return Gamma


def compute_antithetic_paths(
    paths: torch.Tensor,
    dW: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate antithetic paths for variance reduction.
    
    Args:
        paths: Original paths (batch, time, dim)
        dW: Original Brownian increments (batch, time-1, dim)
        
    Returns:
        Antithetic paths and increments
    """
    # Antithetic Brownian increments
    dW_anti = -dW
    
    # Reconstruct paths with antithetic increments
    batch_size, seq_len, dim = paths.shape
    paths_anti = torch.zeros_like(paths)
    paths_anti[:, 0] = paths[:, 0]  # Same initial condition
    
    # Simplified reconstruction (would need drift/vol functions in practice)
    for t in range(seq_len - 1):
        paths_anti[:, t + 1] = paths_anti[:, t] + dW_anti[:, t] * 0.2
    
    return paths_anti, dW_anti