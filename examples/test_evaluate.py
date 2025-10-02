# tests/test_models.py
"""
Unit tests for model components.
"""

import pytest
import torch
import numpy as np
import sys
sys.path.append('..')

from src.models.signature_rde import SignatureRDEBSDE, RDEFunc
from src.data.sde_simulators import simulate_paths


class TestSignatureRDE:
    """Test Signature-RDE model."""
    
    def test_model_initialization(self):
        """Test model can be initialized with various configs."""
        model = SignatureRDEBSDE(
            dim=10,
            signature_depth=3,
            rde_hidden_dim=64
        )
        assert model is not None
        
    def test_forward_pass(self):
        """Test forward pass produces correct output shapes."""
        batch_size, seq_len, dim = 32, 20, 10
        model = SignatureRDEBSDE(dim=dim, signature_depth=2)
        
        # Create dummy input
        x = torch.randn(batch_size, seq_len, dim)
        
        # Forward pass
        outputs = model(x, return_path=True)
        
        assert outputs['Y'].shape == (batch_size, seq_len, 1)
        assert outputs['Z'].shape == (batch_size, seq_len, dim)
        assert outputs['hidden'].shape == (batch_size, seq_len, model.rde_hidden_dim)
        
    def test_2bsde_head(self):
        """Test 2BSDE head produces Gamma output."""
        model = SignatureRDEBSDE(
            dim=5,
            signature_depth=2,
            use_2bsde=True
        )
        
        x = torch.randn(16, 10, 5)
        outputs = model(x, return_path=True)
        
        assert 'Gamma' in outputs
        assert outputs['Gamma'].shape == (16, 10, 5, 5)
        # Check symmetry
        gamma = outputs['Gamma']
        assert torch.allclose(gamma, gamma.transpose(-1, -2))


class TestRDEFunc:
    """Test RDE vector field."""
    
    def test_vector_field_output(self):
        """Test vector field produces correct shapes."""
        input_dim, hidden_dim = 20, 64
        func = RDEFunc(input_dim, hidden_dim)
        
        batch_size = 32
        t = torch.tensor(0.5)
        z = torch.randn(batch_size, hidden_dim)
        
        output = func(t, z)
        assert output.shape == (batch_size, hidden_dim, input_dim + 1)


# scripts/evaluate.py
"""
Evaluation script for trained models.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import sys
sys.path.append('..')

from src.models.signature_rde import SignatureRDEBSDE
from src.data.sde_simulators import simulate_paths
from src.data.payoffs import asian_payoff, barrier_payoff


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize model
    model = SignatureRDEBSDE(
        dim=config['dim'],
        signature_depth=config['model']['signature_depth'],
        rde_hidden_dim=config['model']['rde_width'],
        use_2bsde=config['model'].get('use_2bsde', False)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def evaluate_pricing(model, config, num_samples=10000, device='cpu'):
    """Evaluate pricing accuracy."""
    results = {
        'rpe': [],
        'absolute_error': [],
        'cvar_90': [],
        'cvar_95': [],
        'cvar_975': [],
        'cvar_99': []
    }
    
    batch_size = 1000
    num_batches = num_samples // batch_size
    
    all_errors = []
    all_rel_errors = []
    
    for _ in range(num_batches):
        # Simulate paths
        paths, sigma = simulate_paths(
            batch_size=batch_size,
            dim=config['dim'],
            time_steps=config['solver']['time_steps'],
            T=config['T'],
            device=device
        )
        
        # Model prediction
        with torch.no_grad():
            outputs = model(paths, sigma, return_path=False)
            pred_values = outputs['Y'].squeeze()
        
        # True values
        if config['problem_type'] == 'asian':
            true_values = asian_payoff(paths[:, -1], config['strike']).squeeze()
        else:
            true_values = barrier_payoff(paths[:, -1], config['strike'], 
                                        config.get('barrier', 100)).squeeze()
        
        # Compute errors
        errors = (pred_values - true_values).abs()
        rel_errors = errors / (true_values.abs() + 1e-6)
        
        all_errors.append(errors.cpu().numpy())
        all_rel_errors.append(rel_errors.cpu().numpy())
    
    # Concatenate all errors
    all_errors = np.concatenate(all_errors)
    all_rel_errors = np.concatenate(all_rel_errors)
    
    # Compute metrics
    results['absolute_error'] = np.mean(all_errors)
    results['rpe'] = np.mean(all_rel_errors) * 100  # Percentage
    
    # CVaR metrics
    for q in [0.90, 0.95, 0.975, 0.99]:
        threshold = np.quantile(all_errors, q)
        cvar = np.mean(all_errors[all_errors >= threshold])
        results[f'cvar_{int(q*100)}'] = cvar
    
    return results


def evaluate_greeks(model, config, device='cpu'):
    """Evaluate Greeks (Z and Gamma) accuracy."""
    # Simplified - would compare against finite difference or analytical solutions
    batch_size = 100
    
    paths, sigma = simulate_paths(
        batch_size=batch_size,
        dim=config['dim'],
        time_steps=config['solver']['time_steps'],
        T=config['T'],
        device=device
    )
    
    with torch.no_grad():
        outputs = model(paths, sigma, return_path=True)
    
    Z = outputs['Z']
    
    # Compute statistics
    z_mean = Z.mean().item()
    z_std = Z.std().item()
    z_norm = torch.norm(Z, dim=-1).mean().item()
    
    results = {
        'z_mean': z_mean,
        'z_std': z_std,
        'z_norm': z_norm
    }
    
    if 'Gamma' in outputs:
        Gamma = outputs['Gamma']
        gamma_norm = torch.norm(Gamma.view(batch_size, -1, config['dim']**2), 
                                dim=-1).mean().item()
        results['gamma_norm'] = gamma_norm
    
    return results


def run_scaling_test(model, config, dimensions=[20, 50, 100, 200], device='cpu'):
    """Test model scaling with dimension."""
    results = []
    
    for dim in dimensions:
        # Override dimension
        test_config = config.copy()
        test_config['dim'] = dim
        
        # Create test model (simplified - would need to retrain)
        test_model = SignatureRDEBSDE(
            dim=dim,
            signature_depth=config['model']['signature_depth'],
            rde_hidden_dim=config['model']['rde_width']
        ).to(device)
        test_model.eval()
        
        # Time forward pass
        import time
        paths, sigma = simulate_paths(
            batch_size=100,
            dim=dim,
            time_steps=config['solver']['time_steps'],
            T=config['T'],
            device=device
        )
        
        start = time.time()
        with torch.no_grad():
            _ = test_model(paths, sigma)
        elapsed = time.time() - start
        
        # Memory usage
        if device == 'cuda':
            memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        else:
            memory = 0
        
        results.append({
            'dimension': dim,
            'time': elapsed,
            'memory_gb': memory,
            'params': sum(p.numel() for p in test_model.parameters())
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                       help='Output file for results')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, args.device)
    
    # Pricing evaluation
    print("Evaluating pricing accuracy...")
    pricing_results = evaluate_pricing(model, config, args.num_samples, args.device)
    
    print("\nPricing Results:")
    print(f"  RPE: {pricing_results['rpe']:.2f}%")
    print(f"  Absolute Error: {pricing_results['absolute_error']:.4f}")
    print(f"  CVaR_0.95: {pricing_results['cvar_95']:.4f}")
    print(f"  CVaR_0.99: {pricing_results['cvar_99']:.4f}")
    
    # Greeks evaluation
    print("\nEvaluating Greeks...")
    greeks_results = evaluate_greeks(model, config, args.device)
    
    print("\nGreeks Results:")
    for key, value in greeks_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Scaling test
    print("\nRunning scaling test...")
    scaling_df = run_scaling_test(model, config, device=args.device)
    print(scaling_df)
    
    # Save results
    all_results = {
        'pricing': pricing_results,
        'greeks': greeks_results,
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    # Save to CSV
    results_df = pd.DataFrame([all_results['pricing']])
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Save detailed results
    with open(args.output.replace('.csv', '_detailed.yaml'), 'w') as f:
        yaml.dump(all_results, f)


if __name__ == "__main__":
    main()