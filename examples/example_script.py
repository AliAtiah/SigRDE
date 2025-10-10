"""
Minimal, working example using the training API.
"""

import torch
import numpy as np

from src.models.signature_rde import SignatureRDEBSDE
from src.solvers.bsde_solver import BSDESolver
from src.data.sde_simulators import simulate_paths
from src.data.payoffs import asian_payoff


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        'problem_type': 'asian',
        'dim': 10,
        'T': 1.0,
        'strike': 100.0,
        'model': {
            'signature_depth': 2,
            'rde_width': 64,
            'use_2bsde': False,
            'dropout': 0.0,
            'layer_norm': True,
        },
        'solver': {
            'time_steps': 20,
        },
        'training': {
            'epochs': 3,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'grad_clip': 1.0,
        },
    }

    model = SignatureRDEBSDE(
        dim=config['dim'],
        signature_depth=config['model']['signature_depth'],
        rde_hidden_dim=config['model']['rde_width'],
        use_2bsde=config['model']['use_2bsde'],
        dropout=config['model']['dropout'],
        layer_norm=config['model']['layer_norm'],
    ).to(device)

    solver = BSDESolver(model, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    def driver_f(t, x, y, z, gamma=None):
        return -0.5 * torch.sum(z ** 2, dim=-1, keepdim=True)

    for epoch in range(config['training']['epochs']):
        paths, sigma = simulate_paths(
            batch_size=config['training']['batch_size'],
            dim=config['dim'],
            time_steps=config['solver']['time_steps'],
            T=config['T'],
            device=str(device),
        )

        terminal_g = lambda x: asian_payoff(x, config['strike'])

        optimizer.zero_grad()
        losses = solver.train_step(paths, sigma, driver_f, terminal_g)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        optimizer.step()
        print(f"Epoch {epoch+1}: loss={losses['total']:.4f}")

    with torch.no_grad():
        out = model(paths, sigma, return_path=False)
        print(f"Final Y mean: {out['Y'].mean().item():.4f}")


if __name__ == "__main__":
    main()
