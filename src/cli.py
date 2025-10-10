"""
Command-line interface for SigRDE.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from src.models.signature_rde import SignatureRDEBSDE, MultiWindowSignatureRDE
from src.solvers.bsde_solver import BSDESolver
from src.data.sde_simulators import simulate_paths
from src.data.payoffs import asian_payoff, barrier_payoff


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = SignatureRDEBSDE(
        dim=config['dim'],
        signature_depth=config['model']['signature_depth'],
        rde_hidden_dim=config['model']['rde_width'],
        use_2bsde=config['model'].get('use_2bsde', False),
        dropout=config['model'].get('dropout', 0.0),
        layer_norm=config['model'].get('layer_norm', True)
    ).to(device)

    solver = BSDESolver(model, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Minimal training loop
    for epoch in range(args.epochs or config['training']['epochs']):
        paths, sigma = simulate_paths(
            batch_size=config['training']['batch_size'],
            dim=config['dim'],
            time_steps=config['solver']['time_steps'],
            T=config['T'],
            device=str(device)
        )

        if config['problem_type'] == 'asian':
            terminal_g = lambda x: asian_payoff(x, config['strike'])
        else:
            terminal_g = lambda x: barrier_payoff(x, config['strike'], config.get('barrier', 120.0))

        def driver_f(t, x, y, z, gamma=None):
            return -0.5 * torch.sum(z ** 2, dim=-1, keepdim=True)

        optimizer.zero_grad()
        losses = solver.train_step(paths, sigma, driver_f, terminal_g)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training'].get('grad_clip', 1.0))
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={losses['total']:.4f} term={losses['terminal']:.4f} drift={losses['drift']:.4f}")

    ckpt_dir = Path(args.output or 'checkpoints')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, ckpt_dir / 'last.pt')
    print(f"Saved checkpoint to {ckpt_dir / 'last.pt'}")


def cmd_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = SignatureRDEBSDE(
        dim=config['dim'],
        signature_depth=config['model']['signature_depth'],
        rde_hidden_dim=config['model']['rde_width'],
        use_2bsde=config['model'].get('use_2bsde', False),
        dropout=config['model'].get('dropout', 0.0),
        layer_norm=config['model'].get('layer_norm', True)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        paths, sigma = simulate_paths(
            batch_size=args.batch_size,
            dim=config['dim'],
            time_steps=config['solver']['time_steps'],
            T=config['T'],
            device=str(device)
        )
        outputs = model(paths, sigma, return_path=False)
        print(f"Eval Y mean: {outputs['Y'].mean().item():.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='sigrde', description='SigRDE command-line interface')
    sub = parser.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train', help='Train model')
    p_train.add_argument('--config', type=str, default='configs/default.yaml')
    p_train.add_argument('--device', type=str, default='auto')
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--epochs', type=int, default=None)
    p_train.add_argument('--output', type=str, default='checkpoints')
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser('eval', help='Evaluate checkpoint')
    p_eval.add_argument('--config', type=str, default='configs/default.yaml')
    p_eval.add_argument('--checkpoint', type=str, required=True)
    p_eval.add_argument('--device', type=str, default='auto')
    p_eval.add_argument('--batch_size', type=int, default=1024)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()


