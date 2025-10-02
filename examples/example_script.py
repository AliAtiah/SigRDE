"""
Complete workflow example for Signature-RDE BSDE solver.
This script demonstrates the full pipeline from data generation to evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time

# Import our modules
from src.models.signature_rde import SignatureRDEBSDE
from src.solvers.bsde_solver import BSDESolver
from src.data.sde_simulators import simulate_paths, simulate_heston_paths
from src.data.payoffs import asian_payoff, barrier_payoff, lookback_payoff
from src.utils.losses import CVaRLoss


def main():
    """
    Complete workflow demonstration.
    """
    print("=" * 80)
    print("Deep Signature-RDE BSDE Solver - Complete Workflow")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # =========================================================================
    # 4. TRAINING
    # =========================================================================
    print("\n" + "="*40)
    print("4. TRAINING")
    print("="*40)
    
    # Initialize solver and optimizer
    solver = BSDESolver(model, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # Training history
    train_history = {
        'epoch': [],
        'train_loss': [],
        'terminal_loss': [],
        'drift_loss': [],
        'val_error': [],
        'val_cvar_95': [],
        'learning_rate': [],
        'time': []
    }
    
    print("Starting training...")
    print("-" * 60)
    
    best_val_error = float('inf')
    
    for epoch in range(config['training']['epochs']):
        epoch_start = time.time()
        model.train()
        
        # Training step
        optimizer.zero_grad()
        
        # Generate new batch
        batch_paths, batch_sigma = simulate_paths(
            batch_size=config['training']['batch_size'],
            dim=config['dim'],
            time_steps=config['solver']['time_steps'],
            T=config['T'],
            device=device
        )
        
        # Compute losses
        losses = solver.train_step(
            paths=batch_paths,
            sigma=batch_sigma,
            driver_f=driver_f,
            terminal_g=terminal_g
        )
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            config['training']['grad_clip']
        )
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_paths, val_sigma, return_path=False)
                val_pred = val_outputs['Y'].squeeze()
                val_true = terminal_g(val_paths[:, -1]).squeeze()
                
                val_errors = (val_pred - val_true).abs().cpu().numpy()
                val_error = np.mean(val_errors)
                val_cvar_95 = np.mean(val_errors[val_errors >= np.quantile(val_errors, 0.95)])
                
                if val_error < best_val_error:
                    best_val_error = val_error
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_error': val_error,
                        'config': config
                    }, 'best_model.pt')
        else:
            val_error = None
            val_cvar_95 = None
        
        # Record history
        epoch_time = time.time() - epoch_start
        train_history['epoch'].append(epoch)
        train_history['train_loss'].append(losses['total'])
        train_history['terminal_loss'].append(losses['terminal'])
        train_history['drift_loss'].append(losses['drift'])
        train_history['val_error'].append(val_error)
        train_history['val_cvar_95'].append(val_cvar_95)
        train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        train_history['time'].append(epoch_time)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Loss: {losses['total']:.4f} | "
                  f"Term: {losses['terminal']:.4f} | "
                  f"Drift: {losses['drift']:.4f} | "
                  f"Val Error: {val_error:.4f if val_error else 'N/A'} | "
                  f"Time: {epoch_time:.2f}s")
    
    print("-" * 60)
    print(f"Training completed! Best validation error: {best_val_error:.4f}")
    
    # =========================================================================
    # 5. EVALUATION
    # =========================================================================
    print("\n" + "="*40)
    print("5. EVALUATION")
    print("="*40)
    
    # Load best model
    checkpoint = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Evaluating on test set...")
    
    # Generate large test set
    test_paths, test_sigma = simulate_paths(
        batch_size=5000,
        dim=config['dim'],
        time_steps=config['solver']['time_steps'],
        T=config['T'],
        device=device
    )
    
    with torch.no_grad():
        test_outputs = model(test_paths, test_sigma, return_path=False)
        test_pred = test_outputs['Y'].squeeze()
        test_true = terminal_g(test_paths[:, -1]).squeeze()
    
    # Compute metrics
    test_errors = (test_pred - test_true).abs().cpu().numpy()
    rel_errors = test_errors / (np.abs(test_true.cpu().numpy()) + 1e-6)
    
    metrics = {
        'Mean Absolute Error': np.mean(test_errors),
        'Relative Pricing Error (%)': np.mean(rel_errors) * 100,
        'RMSE': np.sqrt(np.mean(test_errors**2)),
        'CVaR_0.90': np.mean(test_errors[test_errors >= np.quantile(test_errors, 0.90)]),
        'CVaR_0.95': np.mean(test_errors[test_errors >= np.quantile(test_errors, 0.95)]),
        'CVaR_0.975': np.mean(test_errors[test_errors >= np.quantile(test_errors, 0.975)]),
        'CVaR_0.99': np.mean(test_errors[test_errors >= np.quantile(test_errors, 0.99)])
    }
    
    print("\nTest Set Metrics:")
    print("-" * 40)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:.<30} {metric_value:.4f}")
    
    # =========================================================================
    # 6. VISUALIZATION
    # =========================================================================
    print("\n" + "="*40)
    print("6. VISUALIZATION")
    print("="*40)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Training curves
    ax = axes[0, 0]
    ax.plot(train_history['epoch'], train_history['train_loss'], label='Total Loss')
    ax.plot(train_history['epoch'], train_history['terminal_loss'], label='Terminal Loss')
    ax.plot(train_history['epoch'], train_history['drift_loss'], label='Drift Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Validation error
    ax = axes[0, 1]
    val_epochs = [e for e, v in zip(train_history['epoch'], train_history['val_error']) if v is not None]
    val_errors = [v for v in train_history['val_error'] if v is not None]
    val_cvars = [v for v in train_history['val_cvar_95'] if v is not None]
    ax.plot(val_epochs, val_errors, 'o-', label='Mean Error')
    ax.plot(val_epochs, val_cvars, 's-', label='CVaR_0.95')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Error')
    ax.set_title('Validation Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax = axes[0, 2]
    ax.hist(test_errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(test_errors), color='red', linestyle='--', label='Mean')
    ax.axvline(np.quantile(test_errors, 0.95), color='orange', linestyle='--', label='95% Quantile')
    ax.axvline(np.quantile(test_errors, 0.99), color='darkred', linestyle='--', label='99% Quantile')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Predictions vs True Values
    ax = axes[1, 0]
    sample_indices = np.random.choice(len(test_pred), 500, replace=False)
    ax.scatter(test_true[sample_indices].cpu(), test_pred[sample_indices].cpu(), 
               alpha=0.5, s=10)
    ax.plot([test_true.min().cpu(), test_true.max().cpu()], 
            [test_true.min().cpu(), test_true.max().cpu()], 
            'r--', label='Perfect Prediction')
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Predictions vs True Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. CVaR curve
    ax = axes[1, 1]
    quantiles = np.linspace(0.9, 0.99, 20)
    cvars = [np.mean(test_errors[test_errors >= np.quantile(test_errors, q)]) for q in quantiles]
    ax.plot(quantiles, cvars, 'o-', linewidth=2)
    ax.set_xlabel('Quantile')
    ax.set_ylabel('CVaR')
    ax.set_title('Conditional Value-at-Risk Curve')
    ax.grid(True, alpha=0.3)
    
    # 6. Learning rate schedule
    ax = axes[1, 2]
    ax.plot(train_history['epoch'], train_history['learning_rate'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Signature-RDE BSDE Solver Results', fontsize=16)
    plt.tight_layout()
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to 'results.png'")
    
    # =========================================================================
    # 7. SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n" + "="*40)
    print("7. SENSITIVITY ANALYSIS")
    print("="*40)
    
    print("Computing Greeks (sensitivities)...")
    
    # Sample a smaller batch for sensitivity analysis
    sens_paths, sens_sigma = simulate_paths(
        batch_size=100,
        dim=config['dim'],
        time_steps=config['solver']['time_steps'],
        T=config['T'],
        device=device
    )
    
    with torch.no_grad():
        sens_outputs = model(sens_paths, sens_sigma, return_path=True)
    
    # Extract Z (gradient) and analyze
    Z = sens_outputs['Z']
    Z_norms = torch.norm(Z, dim=-1)
    
    print(f"\nZ (Gradient) Statistics:")
    print(f"  Mean norm: {Z_norms.mean().item():.4f}")
    print(f"  Std norm: {Z_norms.std().item():.4f}")
    print(f"  Max norm: {Z_norms.max().item():.4f}")
    print(f"  Min norm: {Z_norms.min().item():.4f}")
    
    # If 2BSDE, analyze Gamma
    if 'Gamma' in sens_outputs:
        Gamma = sens_outputs['Gamma']
        Gamma_norms = torch.norm(Gamma.view(100, -1, config['dim']**2), dim=-1)
        print(f"\nGamma (Hessian) Statistics:")
        print(f"  Mean norm: {Gamma_norms.mean().item():.4f}")
        print(f"  Max eigenvalue: {torch.linalg.eigvalsh(Gamma).max().item():.4f}")
    
    # =========================================================================
    # 8. SUMMARY
    # =========================================================================
    print("\n" + "="*40)
    print("8. SUMMARY")
    print("="*40)
    
    # Create summary dataframe
    summary_data = {
        'Metric': [
            'Model Parameters',
            'Training Epochs',
            'Best Validation Error',
            'Test MAE',
            'Test RPE (%)',
            'Test CVaR_0.95',
            'Test CVaR_0.99',
            'Training Time (s)',
            'Avg Time/Epoch (s)'
        ],
        'Value': [
            f"{total_params:,}",
            config['training']['epochs'],
            f"{best_val_error:.4f}",
            f"{metrics['Mean Absolute Error']:.4f}",
            f"{metrics['Relative Pricing Error (%)']:.2f}",
            f"{metrics['CVaR_0.95']:.4f}",
            f"{metrics['CVaR_0.99']:.4f}",
            f"{sum(train_history['time']):.1f}",
            f"{np.mean(train_history['time']):.2f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\nExperiment Summary:")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('experiment_summary.csv', index=False)
    print("\nSummary saved to 'experiment_summary.csv'")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return model, train_history, metrics


if __name__ == "__main__":
    model, history, metrics = main()1. CONFIGURATION
    # =========================================================================
    print("\n" + "="*40)
    print("1. CONFIGURATION")
    print("="*40)
    
    config = {
        # Problem setup
        'problem_type': 'asian',
        'dim': 100,
        'T': 1.0,
        'strike': 100.0,
        
        # Model architecture
        'model': {
            'signature_depth': 3,
            'rde_width': 128,
            'use_2bsde': False,
            'dropout': 0.0,
            'layer_norm': True
        },
        
        # Solver settings
        'solver': {
            'time_steps': 50,
            'num_paths': 1000,
            'use_dbdp': True,
            'window_size': 12,
            'use_malliavin_z': True
        },
        
        # Training settings
        'training': {
            'epochs': 200,
            'batch_size': 256,
            'learning_rate': 0.001,
            'grad_clip': 1.0
        },
        
        # Loss configuration
        'cvar_quantile': 0.95,
        'cvar_weight': 1.5,
        'lambda_term': 1.0,
        'lambda_drift': 1.0,
        'lambda_2nd': 0.0
    }
    
    print(f"Problem: {config['problem_type'].upper()} option pricing")
    print(f"Dimension: {config['dim']}")
    print(f"Time horizon: {config['T']}")
    print(f"Signature depth: {config['model']['signature_depth']}")
    print(f"RDE width: {config['model']['rde_width']}")
    
    # =========================================================================
    # 2. MODEL INITIALIZATION
    # =========================================================================
    print("\n" + "="*40)
    print("2. MODEL INITIALIZATION")
    print("="*40)
    
    model = SignatureRDEBSDE(
        dim=config['dim'],
        signature_depth=config['model']['signature_depth'],
        rde_hidden_dim=config['model']['rde_width'],
        use_2bsde=config['model']['use_2bsde'],
        dropout=config['model']['dropout'],
        layer_norm=config['model']['layer_norm']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    # =========================================================================
    # 3. DATA GENERATION
    # =========================================================================
    print("\n" + "="*40)
    print("3. DATA GENERATION")
    print("="*40)
    
    print("Generating sample paths...")
    
    # Generate training data
    train_paths, train_sigma = simulate_paths(
        batch_size=config['training']['batch_size'],
        dim=config['dim'],
        time_steps=config['solver']['time_steps'],
        T=config['T'],
        device=device
    )
    
    print(f"Training paths shape: {train_paths.shape}")
    print(f"Volatility shape: {train_sigma.shape}")
    
    # Generate validation data
    val_paths, val_sigma = simulate_paths(
        batch_size=1000,
        dim=config['dim'],
        time_steps=config['solver']['time_steps'],
        T=config['T'],
        device=device
    )
    
    # Define payoff functions
    if config['problem_type'] == 'asian':
        terminal_g = lambda x: asian_payoff(x, config['strike'])
    elif config['problem_type'] == 'barrier':
        terminal_g = lambda x: barrier_payoff(x, config['strike'], barrier=120.0)
    else:
        terminal_g = lambda x: lookback_payoff(x)
    
    # Define BSDE driver
    def driver_f(t, x, y, z, gamma=None):
        # Linear driver for testing
        return -0.5 * torch.sum(z ** 2, dim=-1, keepdim=True)
    
    # =========================================================================
    # 