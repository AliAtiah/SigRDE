# SigRDE: Deep Signature and Neural RDE Methods for Path-Dependent Portfolio Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of signature-based neural RDE methods for solving path-dependent BSDEs in quantitative finance. This repository contains the code and experiments for the paper "Deep Signature and Neural RDE Methods for Path-Dependent Portfolio Optimization".

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Examples](#examples)
- [Paper](#paper)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements novel neural architectures combining truncated log-signature encoders with Neural RDE backbones for solving high-dimensional path-dependent problems in quantitative finance. The method is particularly effective for:

- **Asian basket options** (up to 100 dimensions)
- **Barrier options** with path-dependent payoffs
- **Portfolio optimization** with HJB equations
- **Path-dependent BSDEs** in general

### Key Innovation

The approach leverages the mathematical structure of signatures to efficiently encode path information while maintaining computational tractability for high-dimensional problems.

## ✨ Key Features

- **Signature-RDE Architecture**: Combines truncated log-signatures with neural RDEs
- **High-Dimensional Support**: Tested on problems up to 100 dimensions
- **Multiple Problem Types**: Asian options, barrier options, portfolio optimization
- **Robust Training**: CVaR-based loss functions for tail risk calibration
- **Malliavin Calculus**: Optional Malliavin derivative estimation
- **Comprehensive Evaluation**: Extensive ablation studies and benchmarks

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/SigRDE.git
cd SigRDE

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

- `torch>=2.0.0` - Deep learning framework
- `torchdiffeq>=0.2.3` - Neural ODE solvers
- `signatory>=1.2.6` - Signature computations
- `torchcde>=0.2.5` - Neural CDEs
- `numpy`, `scipy`, `pandas` - Scientific computing
- `matplotlib`, `seaborn` - Visualization

## 🏃‍♂️ Quick Start

### Basic Usage

```python
import torch
from src.models.signature_rde import SignatureRDEBSDE
from src.solvers.bsde_solver import BSDESolver

# Initialize model
model = SignatureRDEBSDE(
    dim=100,
    signature_depth=3,
    rde_hidden_dim=128
)

# Set up solver
solver = BSDESolver(
    model=model,
    time_steps=100,
    num_paths=10000
)

# Train on Asian option problem
solver.train(config_path="configs/asian_d100.yaml")
```

### Running Examples

```bash
# Train Asian basket option model
python examples/train_script.py --config configs/asian_d100.yaml

# Run complete workflow example
python examples/example_script.py

# Evaluate trained model
python examples/test_evaluate.py --model_path checkpoints/asian_model.pth
```

## 📁 Project Structure

```
SigRDE/
├── src/                          # Source code
│   ├── models/                   # Neural network models
│   │   └── signature_rde.py     # Main Signature-RDE architecture
│   ├── solvers/                  # BSDE solvers
│   │   └── bsde_solver.py       # Main solver implementation
│   ├── data/                     # Data generation utilities
│   └── utils/                    # Helper functions
│       └── utils_modules.py     # Utility modules
├── examples/                     # Example scripts
│   ├── train_script.py          # Training script
│   ├── test_evaluate.py         # Evaluation script
│   ├── ablation_script.py       # Ablation studies
│   └── example_script.py        # Complete workflow example
├── configs/                      # Configuration files
│   ├── default.yaml             # Default configuration
│   ├── asian_d100.yaml          # Asian option config
│   └── portfolio_hjb.yaml       # Portfolio optimization config
├── docs/                         # Documentation
│   ├── QF_PAPER___Submission.pdf # Research paper
│   └── license_setup.txt        # License information
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## ⚙️ Configuration

The project uses YAML configuration files for easy experimentation. Key configuration sections:

### Model Configuration
```yaml
model:
  signature_depth: 3          # Truncation depth for signatures
  rde_width: 128             # Hidden dimension for RDE
  rde_width_multiplier: 4    # Width multiplier for vector field
  use_2bsde: false           # Enable for HJB problems
  dropout: 0.0               # Dropout rate
  layer_norm: true           # Layer normalization
```

### Solver Configuration
```yaml
solver:
  time_steps: 100            # Number of time discretization steps
  num_paths: 10000          # Number of Monte Carlo paths
  use_dbdp: true            # Use deep backward dynamic programming
  window_size: 12           # Window size for path processing
  use_malliavin_z: true     # Estimate Malliavin derivatives
```

### Training Configuration
```yaml
training:
  epochs: 500               # Number of training epochs
  batch_size: 512           # Batch size
  learning_rate: 0.001      # Learning rate
  grad_clip: 1.0           # Gradient clipping
```

## 📊 Examples

### 1. Asian Basket Options

Train a model for 100-dimensional Asian basket options:

```bash
python examples/train_script.py --config configs/asian_d100.yaml
```

### 2. Portfolio Optimization

Solve portfolio optimization with HJB equations:

```bash
python examples/train_script.py --config configs/portfolio_hjb.yaml
```

### 3. Ablation Studies

Run comprehensive ablation studies:

```bash
python examples/ablation_script.py
```

## 📄 Paper

The research paper "Deep Signature and Neural RDE Methods for Path-Dependent Portfolio Optimization" is available in `docs/QF_PAPER___Submission.pdf`.

### Citation

If you use this code in your research, please cite:

```bibtex
@article{alzahrani2024sigrde,
  title={Deep Signature and Neural RDE Methods for Path-Dependent Portfolio Optimization},
  author={Alzahrani, Ali},
  journal={Quantitative Finance},
  year={2024}
}
```

## 🔬 Key Results

- **Scalability**: Successfully handles problems up to 100 dimensions
- **Accuracy**: Competitive performance on standard benchmarks
- **Efficiency**: Significant speedup over traditional Monte Carlo methods
- **Robustness**: CVaR-based training improves tail risk calibration

## 🛠️ Development

### Code Style

The project follows PEP 8 style guidelines. Use the provided tools:

```bash
# Format code
black src/ examples/

# Lint code
flake8 src/ examples/

# Type checking
mypy src/
```

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The PyTorch team for the excellent deep learning framework
- The authors of `signatory` for signature computation utilities
- The `torchdiffeq` team for neural ODE solvers
- The quantitative finance community for valuable feedback

## 📞 Contact

Ali Alzahrani - [@yourusername](https://github.com/yourusername) - your.email@example.com

Project Link: [https://github.com/yourusername/SigRDE](https://github.com/yourusername/SigRDE)

---

**Note**: This is research code. For production use, additional testing and validation are recommended.
