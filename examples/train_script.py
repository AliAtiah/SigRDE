"""
Main training script for Signature-RDE BSDE solver.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any

import sys
sys.path.append('..')

from src.models.signature_rde import SignatureRDEBSDE, MultiWindowSignatureRDE
from src.solvers.bsde_solver import BSDESolver
from src.data.sde_simulators import simulate_paths
from src.data.payoffs import asian_payoff, barrier_payoff
from src.utils.losses import CVaRLoss
from src.utils.metrics import compute_metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: Dict[str, Any]) -> nn.Module:
    """Initialize model based on configuration."""
    model = SignatureRDEBSDE(
        dim=config['dim'],
        signature_depth=config['model']['signature_depth'],
        rde_hidden_dim=config['model']['rde_width'],
        use_2bsde=config['model'].get('use_2bsde', False),
        dropout=config['model'].get('dropout', 0.0),
        layer_norm=config['model'].get('layer_norm', True)
    )
    
    # Wrap in multi-window if specified
    if config['solver'].get('use_dbdp', False):
        model = MultiWindowSignatureRDE(
            base_model=model,
            window_size=config['solver']['window_size'],
            window_stride=config['solver'].get('window_stride', 
                                              config['solver']['window_size'] // 2)
        )
    
    return model


def setup_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Setup optimizer with scheduler."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    return optimizer


def setup_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]):
    """Setup learning rate scheduler."""
    total_steps = config['training']['epochs']
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps // 4,
        T_mult=2
    )
    
    return scheduler


def train_epoch(
    model: nn.Module,
    solver: BSDESolver,
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
    epoch: int,
    writer: SummaryWriter
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    batch_size = config['training']['batch_size']
    num_batches = config['training']['batches_per_epoch']
    
    epoch_losses = {
        'terminal': 0.0,
        'drift': 0.0,
        'hjb': 0.0,
        'total': 0.0
    }
    
    for batch_idx in range(num_batches):
        # Simulate paths
        paths, sigma = simulate_paths(
            batch_size=batch_size,
            dim=config['dim'],
            time_steps=config['solver']['time_steps'],
            T=config['T'],
            drift_fn=config.get('drift_fn', None),
            vol_fn=config.get('vol_fn', None)
        )
        
        # Get terminal condition function
        if config['problem_type'] == 'asian':
            terminal_g = lambda x: asian_payoff(x, config['strike'])
        elif config['problem_type'] == 'barrier':
            terminal_g = lambda x: barrier_payoff(x, config['strike'], config['barrier'])
        else:
            raise ValueError(f"Unknown problem type: {config['problem_type']}")
        
        # Driver function (simplified for demo)
        def driver_f(t, x, y, z, gamma=None):
            # Simple linear driver for testing
            return -0.5 * torch.sum(z ** 2, dim=-1, keepdim=True)
        
        # HJB function (if applicable)
        hjb_h = None
        if config['model'].get('use_2bsde', False):
            def hjb_h(t, x, y, z, gamma):
                # Simplified HJB residual
                return torch.sum(gamma.diagonal(dim1=-2, dim2=-1), dim=-1, keepdim=True)
        
        # Training step
        optimizer.zero_grad()
        
        losses = solver.train_step(
            paths=paths,
            sigma=sigma,
            driver_f=driver_f,
            terminal_g=terminal_g,
            hjb_h=hjb_h
        )
        
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training'].get('grad_clip', 1.0)
        )
        
        optimizer.step()
        
        # Accumulate losses
        for key, val in losses.items():
            epoch_losses[key] += val
        
        # Log to tensorboard
        if batch_idx % 10 == 0:
            step = epoch * num_batches + batch_idx
            for key, val in losses.items():
                writer.add_scalar(f'train/{key}', val, step)
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate(
    model: nn.Module,
    config: Dict[str, Any],
    num_samples: int = 1000
) -> Dict[str, float]:
    """Validate model performance."""
    model.eval()
    
    metrics = {
        'rpe': [],  # Relative pricing error
        'cvar_90': [],
        'cvar_95': [],
        'cvar_99': []
    }
    
    with torch.no_grad():
        for _ in range(num_samples // config['validation']['batch_size']):
            # Simulate validation paths
            paths, sigma = simulate_paths(
                batch_size=config['validation']['batch_size'],
                dim=config['dim'],
                time_steps=config['solver']['time_steps'],
                T=config['T']
            )
            
            # Forward pass
            outputs = model(paths, sigma, return_path=False)
            
            # Compute true value (simplified - would use reference solution)
            if config['problem_type'] == 'asian':
                true_value = asian_payoff(paths[:, -1], config['strike'])
            else:
                true_value = barrier_payoff(paths[:, -1], config['strike'], 
                                           config.get('barrier', 100))
            
            pred_value = outputs['Y'].squeeze()
            
            # Compute metrics
            errors = (pred_value - true_value).abs()
            rel_errors = errors / (true_value.abs() + 1e-6)
            
            metrics['rpe'].append(rel_errors.mean().item())
            metrics['cvar_90'].append(torch.quantile(errors, 0.90).item())
            metrics['cvar_95'].append(torch.quantile(errors, 0.95).item())
            metrics['cvar_99'].append(torch.quantile(errors, 0.99).item())
    
    # Average metrics
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Signature-RDE BSDE Solver')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log_dir', type=str, default='runs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for model checkpoints')
    parser.add_argument('--use_2bsde', action='store_true',
                       help='Use 2BSDE head for HJB')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    if args.use_2bsde:
        config['model']['use_2bsde'] = True
    
    # Setup device
    device = torch.device(args.device)
    config['device'] = device
    
    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = setup_model(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize solver
    solver = BSDESolver(model, config)
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        # Adjust CVaR parameters (curriculum)
        if epoch > config['training']['epochs'] * 0.2:
            solver.cvar_quantile = min(0.99, solver.cvar_quantile + 0.01)
            solver.cvar_weight = min(3.0, solver.cvar_weight + 0.1)
        
        # Enable 2BSDE loss after warm-up
        if epoch > config['training']['epochs'] * 0.2 and config['model'].get('use_2bsde', False):
            solver.lambda_2nd = config.get('lambda_2nd', 0.2)
        
        # Training
        train_losses = train_epoch(
            model, solver, optimizer, config, epoch, writer
        )
        
        # Validation
        if epoch % config['validation']['interval'] == 0:
            val_metrics = validate(model, config)
            
            # Log validation metrics
            for key, val in val_metrics.items():
                writer.add_scalar(f'val/{key}', val, epoch)
            
            # Save best model
            if val_metrics['rpe'] < best_val_loss:
                best_val_loss = val_metrics['rpe']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'val_metrics': val_metrics
                }, f"{args.checkpoint_dir}/best_model.pt")
            
            print(f"Epoch {epoch}: Train Loss={train_losses['total']:.4f}, "
                  f"Val RPE={val_metrics['rpe']:.4f}, "
                  f"CVaR95={val_metrics['cvar_95']:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint periodically
        if epoch % config['training'].get('checkpoint_interval', 50) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, f"{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()