"""
Ablation studies for Signature-RDE BSDE solver.
Reproduces ablation results from the paper.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import time
from typing import Dict, List, Any
import sys
sys.path.append('..')

from src.models.signature_rde import SignatureRDEBSDE
from src.solvers.bsde_solver import BSDESolver
from src.data.sde_simulators import simulate_paths
from src.data.payoffs import asian_payoff, barrier_payoff


def run_single_ablation(
    config: Dict[str, Any],
    param_name: str,
    param_value: Any,
    num_epochs: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Run a single ablation experiment.
    
    Args:
        config: Base configuration
        param_name: Parameter to ablate
        param_value: Value for the parameter
        num_epochs: Number of training epochs
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    # Update config with ablation parameter
    if '.' in param_name:
        # Nested parameter (e.g., 'model.signature_depth')
        keys = param_name.split('.')
        temp = config
        for key in keys[:-1]:
            temp = temp[key]
        temp[keys[-1]] = param_value
    else:
        config[param_name] = param_value
    
    # Initialize model
    model = SignatureRDEBSDE(
        dim=config['dim'],
        signature_depth=config['model']['signature_depth'],
        rde_hidden_dim=config['model']['rde_width'],
        use_2bsde=config['model'].get('use_2bsde', False)
    ).to(device)
    
    # Initialize solver and optimizer
    solver = BSDESolver(model, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Terminal condition
    if config['problem_type'] == 'asian':
        terminal_g = lambda x: asian_payoff(x, config['strike'])
    else:
        terminal_g = lambda x: barrier_payoff(x, config['strike'], config.get('barrier', 100))
    
    # Driver function
    def driver_f(t, x, y, z, gamma=None):
        return -0.5 * torch.sum(z ** 2, dim=-1, keepdim=True)
    
    # Training metrics
    train_losses = []
    train_times = []
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Simulate paths
        paths, sigma = simulate_paths(
            batch_size=256,
            dim=config['dim'],
            time_steps=config['solver']['time_steps'],
            T=config['T'],
            device=device
        )
        
        # Training step
        optimizer.zero_grad()
        losses = solver.train_step(paths, sigma, driver_f, terminal_g)
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(losses)
        train_times.append(time.time() - epoch_start)
    
    total_time = time.time() - start_time
    
    # Validation
    model.eval()
    val_errors = []
    
    with torch.no_grad():
        for _ in range(10):
            val_paths, val_sigma = simulate_paths(
                batch_size=500,
                dim=config['dim'],
                time_steps=config['solver']['time_steps'],
                T=config['T'],
                device=device
            )
            
            outputs = model(val_paths, val_sigma, return_path=False)
            pred_values = outputs['Y'].squeeze()
            true_values = terminal_g(val_paths[:, -1]).squeeze()
            
            errors = (pred_values - true_values).abs()
            val_errors.extend(errors.cpu().numpy())
    
    val_errors = np.array(val_errors)
    
    # Compute metrics
    metrics = {
        'param_value': param_value,
        'final_train_loss': train_losses[-1]['total'],
        'mean_error': np.mean(val_errors),
        'rpe': np.mean(val_errors / (np.abs(val_errors) + 1e-6)) * 100,
        'cvar_90': np.mean(val_errors[val_errors >= np.quantile(val_errors, 0.90)]),
        'cvar_95': np.mean(val_errors[val_errors >= np.quantile(val_errors, 0.95)]),
        'cvar_975': np.mean(val_errors[val_errors >= np.quantile(val_errors, 0.975)]),
        'cvar_99': np.mean(val_errors[val_errors >= np.quantile(val_errors, 0.99)]),
        'training_time': total_time,
        'time_per_epoch': np.mean(train_times),
        'model_params': sum(p.numel() for p in model.parameters()),
        'nan_rate': sum(1 for l in train_losses if np.isnan(l['total'])) / len(train_losses) * 100
    }
    
    return metrics


def ablation_signature_depth(base_config: Dict, device: str = 'cpu') -> pd.DataFrame:
    """Run ablation on signature depth."""
    print("\n=== Ablation: Signature Depth ===")
    
    depths = [2, 3, 4, 5]
    results = []
    
    for depth in depths:
        print(f"Testing signature depth = {depth}")
        metrics = run_single_ablation(
            base_config.copy(),
            'model.signature_depth',
            depth,
            device=device
        )
        metrics['signature_depth'] = depth
        results.append(metrics)
        
        print(f"  RPE: {metrics['rpe']:.2f}%, CVaR_0.95: {metrics['cvar_95']:.4f}, "
              f"Time: {metrics['training_time']:.1f}s")
    
    return pd.DataFrame(results)


def ablation_rde_width(base_config: Dict, device: str = 'cpu') -> pd.DataFrame:
    """Run ablation on RDE hidden dimension."""
    print("\n=== Ablation: RDE Width ===")
    
    widths = [64, 128, 192, 256]
    results = []
    
    for width in widths:
        print(f"Testing RDE width = {width}")
        metrics = run_single_ablation(
            base_config.copy(),
            'model.rde_width',
            width,
            device=device
        )
        metrics['rde_width'] = width
        results.append(metrics)
        
        print(f"  RPE: {metrics['rpe']:.2f}%, CVaR_0.95: {metrics['cvar_95']:.4f}, "
              f"Params: {metrics['model_params']:,}")
    
    return pd.DataFrame(results)


def ablation_window_size(base_config: Dict, device: str = 'cpu') -> pd.DataFrame:
    """Run ablation on DBDP window size."""
    print("\n=== Ablation: Window Size ===")
    
    windows = [8, 12, 16, 20]
    results = []
    
    for window in windows:
        print(f"Testing window size = {window}")
        metrics = run_single_ablation(
            base_config.copy(),
            'solver.window_size',
            window,
            device=device
        )
        metrics['window_size'] = window
        results.append(metrics)
        
        print(f"  RPE: {metrics['rpe']:.2f}%, CVaR_0.95: {metrics['cvar_95']:.4f}, "
              f"NaN rate: {metrics['nan_rate']:.1f}%")
    
    return pd.DataFrame(results)


def ablation_malliavin(base_config: Dict, device: str = 'cpu') -> pd.DataFrame:
    """Run ablation on Malliavin stabilization."""
    print("\n=== Ablation: Malliavin Stabilization ===")
    
    settings = [
        {'use_malliavin_z': False, 'use_malliavin_gamma': False, 'label': 'None'},
        {'use_malliavin_z': True, 'use_malliavin_gamma': False, 'label': 'Z only'},
        {'use_malliavin_z': False, 'use_malliavin_gamma': True, 'label': 'Gamma only'},
        {'use_malliavin_z': True, 'use_malliavin_gamma': True, 'label': 'Both'}
    ]
    
    results = []
    
    for setting in settings:
        print(f"Testing Malliavin: {setting['label']}")
        config = base_config.copy()
        config['solver']['use_malliavin_z'] = setting['use_malliavin_z']
        config['solver']['use_malliavin_gamma'] = setting['use_malliavin_gamma']
        
        metrics = run_single_ablation(
            config,
            'dummy',  # No specific parameter change
            None,
            device=device
        )
        metrics['malliavin'] = setting['label']
        results.append(metrics)
        
        print(f"  RPE: {metrics['rpe']:.2f}%, CVaR_0.95: {metrics['cvar_95']:.4f}, "
              f"NaN rate: {metrics['nan_rate']:.1f}%")
    
    return pd.DataFrame(results)


def ablation_cvar_params(base_config: Dict, device: str = 'cpu') -> pd.DataFrame:
    """Run ablation on CVaR parameters."""
    print("\n=== Ablation: CVaR Parameters ===")
    
    settings = [
        {'quantile': 0.90, 'weight': 1.0},
        {'quantile': 0.95, 'weight': 1.5},
        {'quantile': 0.975, 'weight': 2.0},
        {'quantile': 0.99, 'weight': 3.0}
    ]
    
    results = []
    
    for setting in settings:
        print(f"Testing CVaR: q={setting['quantile']}, η={setting['weight']}")
        config = base_config.copy()
        config['cvar_quantile'] = setting['quantile']
        config['cvar_weight'] = setting['weight']
        
        metrics = run_single_ablation(
            config,
            'dummy',
            None,
            device=device
        )
        metrics['cvar_quantile'] = setting['quantile']
        metrics['cvar_weight'] = setting['weight']
        results.append(metrics)
        
        print(f"  RPE: {metrics['rpe']:.2f}%, CVaR_0.95: {metrics['cvar_95']:.4f}, "
              f"CVaR_0.99: {metrics['cvar_99']:.4f}")
    
    return pd.DataFrame(results)


def plot_ablation_results(results_dict: Dict[str, pd.DataFrame], save_dir: str = 'figures'):
    """Create visualization of ablation results."""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create subplots for each ablation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1. Signature depth
    if 'signature_depth' in results_dict:
        df = results_dict['signature_depth']
        ax = axes[0]
        ax.plot(df['signature_depth'], df['rpe'], 'o-', label='RPE', linewidth=2)
        ax.set_xlabel('Signature Depth')
        ax.set_ylabel('RPE (%)')
        ax.set_title('Signature Depth Ablation')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(df['signature_depth'], df['training_time'], 's-', color='red', 
                label='Time', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Training Time (s)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    # 2. RDE width
    if 'rde_width' in results_dict:
        df = results_dict['rde_width']
        ax = axes[1]
        ax.plot(df['rde_width'], df['rpe'], 'o-', label='RPE', linewidth=2)
        ax.plot(df['rde_width'], df['cvar_95'], 's-', label='CVaR_0.95', linewidth=2)
        ax.set_xlabel('RDE Width')
        ax.set_ylabel('Error')
        ax.set_title('RDE Width Ablation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Window size
    if 'window_size' in results_dict:
        df = results_dict['window_size']
        ax = axes[2]
        ax.plot(df['window_size'], df['rpe'], 'o-', label='RPE', linewidth=2)
        ax.plot(df['window_size'], df['nan_rate'], 's-', label='NaN Rate', linewidth=2)
        ax.set_xlabel('Window Size')
        ax.set_ylabel('Value')
        ax.set_title('Window Size Ablation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Malliavin
    if 'malliavin' in results_dict:
        df = results_dict['malliavin']
        ax = axes[3]
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width/2, df['rpe'], width, label='RPE')
        ax.bar(x + width/2, df['cvar_95'], width, label='CVaR_0.95')
        ax.set_xlabel('Malliavin Setting')
        ax.set_ylabel('Error')
        ax.set_title('Malliavin Stabilization')
        ax.set_xticks(x)
        ax.set_xticklabels(df['malliavin'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. CVaR parameters - Heatmap
    if 'cvar_params' in results_dict:
        df = results_dict['cvar_params']
        ax = axes[4]
        pivot = df.pivot_table(values='cvar_99', 
                               index='cvar_weight', 
                               columns='cvar_quantile')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title('CVaR_0.99 vs CVaR Parameters')
    
    # 6. Combined metrics
    ax = axes[5]
    for name, df in results_dict.items():
        if name == 'signature_depth':
            ax.plot(df['training_time'], df['rpe'], 'o-', label=name)
    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('RPE (%)')
    ax.set_title('Efficiency Frontier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Ablation Study Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ablation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--config', type=str, default='configs/asian_d100.yaml',
                       help='Base configuration file')
    parser.add_argument('--ablation', type=str, default='all',
                       choices=['all', 'signature_depth', 'rde_width', 
                               'window_size', 'malliavin', 'cvar_params'],
                       help='Which ablation to run')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='Directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Dictionary to store results
    all_results = {}
    
    # Run ablations
    if args.ablation == 'all' or args.ablation == 'signature_depth':
        all_results['signature_depth'] = ablation_signature_depth(base_config, args.device)
    
    if args.ablation == 'all' or args.ablation == 'rde_width':
        all_results['rde_width'] = ablation_rde_width(base_config, args.device)
    
    if args.ablation == 'all' or args.ablation == 'window_size':
        all_results['window_size'] = ablation_window_size(base_config, args.device)
    
    if args.ablation == 'all' or args.ablation == 'malliavin':
        all_results['malliavin'] = ablation_malliavin(base_config, args.device)
    
    if args.ablation == 'all' or args.ablation == 'cvar_params':
        all_results['cvar_params'] = ablation_cvar_params(base_config, args.device)
    
    # Save results
    for name, df in all_results.items():
        df.to_csv(f'{args.output_dir}/ablation_{name}.csv', index=False)
        print(f"\nSaved {name} results to {args.output_dir}/ablation_{name}.csv")
    
    # Create visualizations
    if all_results:
        fig = plot_ablation_results(all_results, args.output_dir)
        print(f"\nSaved visualizations to {args.output_dir}/ablation_results.png")
    
    # Print summary
    print("\n=== Ablation Summary ===")
    for name, df in all_results.items():
        print(f"\n{name}:")
        print(df[['param_value', 'rpe', 'cvar_95', 'training_time']].to_string())


if __name__ == "__main__":
    main()