"""
Main Signature-RDE BSDE Architecture Implementation
Paper: Deep Signature and Neural RDE Methods for Path-Dependent Portfolio Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import torchcde
import signatory

class SignatureRDEBSDE(nn.Module):
    """
    Main Signature-RDE architecture for solving path-dependent BSDEs.
    
    Combines truncated log-signature encoders with Neural RDE backbone
    for high-dimensional path-dependent problems in quantitative finance.
    """
    
    def __init__(
        self,
        dim: int,
        signature_depth: int = 3,
        rde_hidden_dim: int = 128,
        rde_width_multiplier: int = 4,
        use_2bsde: bool = False,
        dropout: float = 0.0,
        layer_norm: bool = True
    ):
        """
        Args:
            dim: State dimension
            signature_depth: Truncation depth for log-signature
            rde_hidden_dim: Hidden dimension for RDE
            rde_width_multiplier: Width multiplier for RDE vector field
            use_2bsde: Whether to use 2BSDE head for second-order
            dropout: Dropout rate
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.dim = dim
        self.signature_depth = signature_depth
        self.rde_hidden_dim = rde_hidden_dim
        self.use_2bsde = use_2bsde
        
        # Calculate signature dimension
        self.sig_dim = signatory.logsignature_channels(
            in_channels=dim + 1,  # +1 for time
            depth=signature_depth
        )
        
        # Initial embedding network
        self.h_init = nn.Sequential(
            nn.Linear(dim, rde_hidden_dim),
            nn.ReLU(),
            nn.Linear(rde_hidden_dim, rde_hidden_dim)
        )
        
        # Neural RDE vector field
        self.rde_func = RDEFunc(
            input_dim=self.sig_dim,
            hidden_dim=rde_hidden_dim,
            width_multiplier=rde_width_multiplier,
            dropout=dropout,
            layer_norm=layer_norm
        )
        
        # Decoder heads
        self.y_head = nn.Sequential(
            nn.Linear(rde_hidden_dim, rde_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(rde_hidden_dim // 2, 1)
        )
        
        self.z_head = nn.Sequential(
            nn.Linear(rde_hidden_dim, rde_hidden_dim),
            nn.ReLU(),
            nn.Linear(rde_hidden_dim, dim)
        )
        
        if use_2bsde:
            # Gamma head for second-order (HJB)
            self.gamma_head = nn.Sequential(
                nn.Linear(rde_hidden_dim, rde_hidden_dim),
                nn.ReLU(),
                nn.Linear(rde_hidden_dim, dim * dim)
            )
        
        # Layer norm for signatures if enabled
        if layer_norm:
            self.sig_norm = nn.LayerNorm(self.sig_dim)

        # Temporal attention readout (learnable query)
        self.attn_query = nn.Parameter(torch.randn(rde_hidden_dim))
        self.attn_proj = nn.Linear(rde_hidden_dim, rde_hidden_dim, bias=False)
    
    def compute_signature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute truncated log-signature of time-augmented paths.
        
        Args:
            x: Path tensor of shape (batch, time, dim)
            
        Returns:
            Log-signature features of shape (batch, time, sig_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Time augmentation
        t = torch.linspace(0, 1, seq_len, device=x.device)
        t = t.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        x_augmented = torch.cat([t, x], dim=-1)
        
        # Compute log-signature
        logsig = signatory.logsignature(
            x_augmented,
            self.signature_depth,
            basepoint=True
        )
        
        if hasattr(self, 'sig_norm'):
            logsig = self.sig_norm(logsig)
            
        return logsig
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        return_path: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Signature-RDE model.
        
        Args:
            x: Path tensor of shape (batch, time, dim)
            sigma: Volatility for Z scaling (batch, time, dim, dim)
            return_path: Whether to return full path or just terminal
            
        Returns:
            Dictionary with Y, Z, and optionally Gamma
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        
        # Compute log-signature features
        logsig = self.compute_signature(x)
        
        # Initial hidden state
        h0 = self.h_init(x[:, 0, :])
        
        # Create controlled path for RDE
        # Using cubic spline interpolation
        coeffs = torchcde.natural_cubic_coeffs(logsig)
        X = torchcde.CubicSpline(coeffs)
        
        # Solve RDE
        t = torch.linspace(0, 1, seq_len, device=device)
        
        # Use adjoint for memory efficiency
        h = torchcde.cdeint(
            X=X,
            func=self.rde_func,
            z0=h0,
            t=t,
            adjoint=True,
            method='midpoint'
        )
        # torchcde returns (time, batch, hidden). Make batch-first.
        h = h.transpose(0, 1)  # (batch, time, hidden)
        
        # Decode to BSDE quantities
        # Temporal attention context
        q = self.attn_query.unsqueeze(0).unsqueeze(0)  # (1,1,hidden)
        keys = self.attn_proj(h)  # (batch, time, hidden)
        attn_logits = torch.sum(keys * q, dim=-1)  # (batch, time)
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (batch, time, 1)
        context = torch.sum(attn_weights * h, dim=1)  # (batch, hidden)

        Y_path = self.y_head(h)
        Z_path = self.z_head(h)
        
        # Scale Z by sigma^{1/2} if provided
        if sigma is not None:
            # Compute Sigma^{1/2} using Cholesky
            jitter = 1e-6
            eye = torch.eye(dim, device=device)
            sigma_sqrt = torch.linalg.cholesky(sigma + jitter * eye)
            Z_path = torch.matmul(Z_path.unsqueeze(-2), sigma_sqrt).squeeze(-2)
        
        # Expose path-wise outputs and attention context-based terminal Y_attn
        outputs = {'Y': Y_path, 'Z': Z_path, 'hidden': h, 'Y_attn': self.y_head(context.unsqueeze(1))}
        
        if self.use_2bsde:
            Gamma = self.gamma_head(h).view(batch_size, seq_len, dim, dim)
            # Symmetrize Gamma
            Gamma = 0.5 * (Gamma + Gamma.transpose(-1, -2))
            outputs['Gamma'] = Gamma
        
        if not return_path:
            # Return only terminal values (keep attention-based Y_attn as scalar)
            term = {k: v[:, -1] for k, v in outputs.items() if k != 'Y_attn'}
            term['Y_attn'] = outputs['Y_attn'].squeeze(1)
            outputs = term
            
        return outputs


class RDEFunc(nn.Module):
    """
    Neural RDE vector field function.
    
    Implements the learned vector field F_θ for the RDE:
    dh_t = F_θ(h_t) dU_t
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        width_multiplier: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        layers = []
        in_dim = hidden_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim * width_multiplier if i == 0 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm and i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            if dropout > 0 and i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim
        
        # Final layer maps to hidden_dim x input_dim
        layers.append(nn.Linear(in_dim, hidden_dim * (input_dim + 1)))
        
        self.net = nn.Sequential(*layers)
        # Time FiLM conditioning
        self.time_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
    
    def forward(self, t, z):
        """
        Compute vector field at time t and state z.
        
        Args:
            t: Time (scalar)
            z: Hidden state of shape (batch, hidden_dim)
            
        Returns:
            Vector field output
        """
        # Time features (sin, cos)
        t_feat = torch.stack([torch.sin(t), torch.cos(t)], dim=0).unsqueeze(0).expand(z.shape[0], -1)
        gamma_beta = self.time_proj(t_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        z_cond = (1 + torch.tanh(gamma)) * z + beta
        out = self.net(z_cond)
        return out.view(z.shape[0], self.hidden_dim, self.input_dim + 1)


class DecoderHeads(nn.Module):
    """
    Decoder heads for Y, Z, and Gamma with optional Malliavin stabilization.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        use_2bsde: bool = False,
        malliavin_z: bool = False,
        malliavin_gamma: bool = False
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.use_2bsde = use_2bsde
        self.malliavin_z = malliavin_z
        self.malliavin_gamma = malliavin_gamma
        
        # Y head (value function)
        self.y_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Z head (gradient)
        if not malliavin_z:
            self.z_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            )
        
        # Gamma head (Hessian)
        if use_2bsde and not malliavin_gamma:
            self.gamma_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim * state_dim)
            )
    
    def forward(
        self,
        h: torch.Tensor,
        z_malliavin: Optional[torch.Tensor] = None,
        gamma_malliavin: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decode hidden states to BSDE quantities.
        """
        batch_size = h.shape[0]
        
        outputs = {}
        
        # Y output
        outputs['Y'] = self.y_net(h)
        
        # Z output
        if self.malliavin_z and z_malliavin is not None:
            outputs['Z'] = z_malliavin
        else:
            outputs['Z'] = self.z_net(h)
        
        # Gamma output (if 2BSDE)
        if self.use_2bsde:
            if self.malliavin_gamma and gamma_malliavin is not None:
                outputs['Gamma'] = gamma_malliavin
            else:
                gamma = self.gamma_net(h)
                gamma = gamma.view(batch_size, self.state_dim, self.state_dim)
                # Symmetrize
                gamma = 0.5 * (gamma + gamma.transpose(-1, -2))
                outputs['Gamma'] = gamma
        
        return outputs


# Multi-window DBDP implementation
class MultiWindowSignatureRDE(nn.Module):
    """
    Multi-window implementation following DBDP for variance reduction.
    """
    
    def __init__(
        self,
        base_model: SignatureRDEBSDE,
        window_size: int = 12,
        window_stride: int = 6
    ):
        super().__init__()
        self.base_model = base_model
        self.window_size = window_size
        self.window_stride = window_stride
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process path using overlapping windows.
        """
        batch_size, seq_len, dim = x.shape
        
        # Collect outputs from all windows
        all_outputs = []
        
        for start in range(0, seq_len - self.window_size + 1, self.window_stride):
            end = min(start + self.window_size, seq_len)
            
            # Extract window
            x_window = x[:, start:end, :]
            sigma_window = sigma[:, start:end] if sigma is not None else None
            
            # Process window
            outputs = self.base_model(x_window, sigma_window, return_path=True)
            all_outputs.append(outputs)
        
        # Aggregate outputs (averaging overlapping regions)
        return self._aggregate_windows(all_outputs, seq_len)
    
    def _aggregate_windows(
        self,
        outputs: list,
        seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate outputs from overlapping windows.
        """
        if len(outputs) == 0:
            raise ValueError("No window outputs to aggregate")

        sample = outputs[0]
        device = sample['Y'].device
        batch_size = sample['Y'].shape[0]

        agg: Dict[str, torch.Tensor] = {}
        count = torch.zeros(batch_size, seq_len, 1, device=device)

        keys = ['Y', 'Z'] + ([ 'Gamma' ] if 'Gamma' in sample else [])
        for k in keys:
            shape_tail = sample[k].shape[2:]
            agg[k] = torch.zeros(batch_size, seq_len, *shape_tail, device=device)

        # We need the same iteration as in forward; recompute positions
        # Since we don't store starts/ends, infer from concatenation lengths by tracking
        # Assume constant window_size and stride
        # Build start indices from cumulative placement
        # Safer: pass starts via storing in dicts; here we infer by stepping
        # We'll recompute using self.window_size and self.window_stride
        pos = 0
        for idx, out in enumerate(outputs):
            start = idx * self.window_stride
            end = min(start + out['Y'].shape[1], seq_len)
            length = end - start
            sl = slice(start, end)
            for k in keys:
                agg[k][:, sl] += out[k][:, :length]
            count[:, sl] += 1

        count = torch.clamp(count, min=1.0)
        for k in keys:
            agg[k] = agg[k] / count

        return agg