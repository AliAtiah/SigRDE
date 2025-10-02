"""
BSDE Solver Implementation with CVaR-tilted Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable
import numpy as np


class BSDESolver:
    """
    Main BSDE solver with CVaR-tilted training and optional 2BSDE/HJB support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict
    ):
        """
        Initialize BSDE solver.
        
        Args:
            model: Signature-RDE model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Extract key parameters
        self.dim = config['dim']
        self.num_paths = config['num_paths']
        self.time_steps = config['time_steps']
        self.dt = config['T'] / self.time_steps
        
        # CVaR parameters
        self.cvar_quantile = config.get('cvar_quantile', 0.95)
        self.cvar_weight = config.get('cvar_weight', 1.5)
        
        # Loss weights
        self.lambda_term = config.get('lambda_term', 1.0)
        self.lambda_drift = config.get('lambda_drift', 1.0)
        self.lambda_2nd = config.get('lambda_2nd', 0.0)
        
        # Malliavin parameters
        self.use_malliavin_z = config.get('use_malliavin_z', False)
        self.use_malliavin_gamma = config.get('use_malliavin_gamma', False)
        
    def compute_losses(
        self,
        paths: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        driver_f: Callable,
        terminal_g: Callable,
        hjb_h: Optional[Callable] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BSDE losses with CVaR tilt.
        
        Args:
            paths: Simulated paths (batch, time, dim)
            outputs: Model outputs with Y, Z, and optionally Gamma
            driver_f: BSDE driver function
            terminal_g: Terminal condition function
            hjb_h: Optional HJB residual function
            
        Returns:
            Dictionary of losses
        """
        batch_size, seq_len, _ = paths.shape
        device = paths.device
        
        losses = {}
        
        # 1. Terminal loss with CVaR tilt
        Y_T = outputs['Y'][:, -1]
        g_T = terminal_g(paths[:, -1])
        
        terminal_error = (Y_T - g_T) ** 2
        losses['terminal'] = self._cvar_tilted_loss(terminal_error)
        
        # 2. BSDE drift residual
        Y = outputs['Y']
        Z = outputs['Z']
        
        # Generate Brownian increments
        dW = torch.randn(batch_size, seq_len - 1, self.dim, device=device)
        dW = dW * np.sqrt(self.dt)
        
        drift_residual = 0
        for t in range(seq_len - 1):
            # BSDE dynamics
            f_t = driver_f(
                t * self.dt,
                paths[:, t],
                Y[:, t],
                Z[:, t],
                outputs.get('Gamma', [None])[t] if 'Gamma' in outputs else None
            )
            
            # Y_{t+1} - Y_t + f dt - Z dW
            dY_pred = -f_t * self.dt + torch.sum(Z[:, t] * dW[:, t], dim=-1, keepdim=True)
            dY_actual = Y[:, t + 1] - Y[:, t]
            
            drift_residual += torch.mean((dY_actual - dY_pred) ** 2) / self.dt
        
        losses['drift'] = drift_residual / (seq_len - 1)
        
        # 3. Second-order/HJB loss (if applicable)
        if hjb_h is not None and 'Gamma' in outputs:
            hjb_residual = 0
            for t in range(seq_len):
                h_t = hjb_h(
                    t * self.dt,
                    paths[:, t],
                    Y[:, t],
                    Z[:, t],
                    outputs['Gamma'][:, t]
                )
                hjb_residual += torch.mean(F.relu(h_t) ** 2)
            
            losses['hjb'] = hjb_residual / seq_len
        
        # 4. Malliavin stabilization losses
        if self.use_malliavin_z:
            z_malliavin = self._compute_malliavin_z(Y, dW)
            losses['malliavin_z'] = torch.mean((Z[:, :-1] - z_malliavin) ** 2)
        
        if self.use_malliavin_gamma and 'Gamma' in outputs:
            gamma_malliavin = self._compute_malliavin_gamma(Y, paths)
            losses['malliavin_gamma'] = torch.mean(
                (outputs['Gamma'][:, :-1] - gamma_malliavin) ** 2
            )
        
        return losses
    
    def _cvar_tilted_loss(self, errors: torch.Tensor) -> torch.Tensor:
        """
        Compute CVaR-tilted loss.
        
        L = E[(1 + η * 1{Δ² ≥ Q_q(Δ²)}) * Δ²]
        
        Args:
            errors: Squared errors (batch,)
            
        Returns:
            CVaR-tilted loss scalar
        """
        quantile = torch.quantile(errors, self.cvar_quantile)
        weights = 1 + self.cvar_weight * (errors >= quantile).float()
        return torch.mean(weights * errors)
    
    def _compute_malliavin_z(
        self,
        Y: torch.Tensor,
        dW: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Malliavin weight estimate for Z.
        
        Z_t ≈ (1/Δt) * E[Y_{t+1} * (σ^{-1} dW_t)^T]
        
        Args:
            Y: Value process (batch, time, 1)
            dW: Brownian increments (batch, time-1, dim)
            
        Returns:
            Malliavin estimate of Z
        """
        batch_size = Y.shape[0]
        
        # Simplified: assuming σ = I for now
        # In practice, need to compute σ^{-1}
        z_malliavin = torch.zeros(batch_size, Y.shape[1] - 1, self.dim, device=Y.device)
        
        for t in range(Y.shape[1] - 1):
            # Regression target
            z_malliavin[:, t] = Y[:, t + 1].squeeze() * dW[:, t] / self.dt
        
        return z_malliavin
    
    def _compute_malliavin_gamma(
        self,
        Y: torch.Tensor,
        paths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Malliavin/finite-difference estimate for Gamma.
        
        Uses antithetic paths for variance reduction.
        """
        # Simplified implementation
        # In practice, use finite differences with control variates
        batch_size = Y.shape[0]
        return torch.zeros(
            batch_size,
            Y.shape[1] - 1,
            self.dim,
            self.dim,
            device=Y.device
        )
    
    def train_step(
        self,
        paths: torch.Tensor,
        sigma: torch.Tensor,
        driver_f: Callable,
        terminal_g: Callable,
        hjb_h: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            paths: Simulated paths
            sigma: Volatility
            driver_f: BSDE driver
            terminal_g: Terminal condition
            hjb_h: Optional HJB residual
            
        Returns:
            Dictionary of loss values
        """
        # Forward pass
        outputs = self.model(paths, sigma, return_path=True)
        
        # Compute losses
        losses = self.compute_losses(paths, outputs, driver_f, terminal_g, hjb_h)
        
        # Combine losses
        total_loss = (
            self.lambda_term * losses['terminal'] +
            self.lambda_drift * losses['drift']
        )
        
        if 'hjb' in losses:
            total_loss += self.lambda_2nd * losses['hjb']
        
        if 'malliavin_z' in losses:
            total_loss += 0.1 * losses['malliavin_z']
        
        if 'malliavin_gamma' in losses:
            total_loss += 0.1 * losses['malliavin_gamma']
        
        losses['total'] = total_loss
        
        return {k: v.item() for k, v in losses.items()}