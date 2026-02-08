# LDP-MIC: Correlation-Aware Local Differential Privacy for Federated Learning

[![Paper](https://img.shields.io/badge/USENIX%20Security-2026-blue)](https://anonymous.4open.science/r/LDP-MIC)

## Overview

LDP-MIC is a correlation-adaptive local differential privacy framework for federated learning under untrusted aggregation. By adapting client-side noise to feature–target dependence via the Maximum Information Coefficient (MIC), LDP-MIC mitigates the utility degradation typical of LDP in heterogeneous, non-IID settings while enforcing all privacy guarantees locally.

**Key Innovation**: Unlike standard LDP that applies uniform noise, LDP-MIC allocates privacy budget asymmetrically across features based on their correlation with the target variable—applying less noise to informative features and more to less salient ones—while maintaining rigorous (ε,δ)-LDP guarantees.

## Key Features

- **Correlation-aware noise allocation** using Maximum Information Coefficient (MIC)
- **(ε,δ)-LDP guarantees** without requiring a trusted aggregator
- **MIC-based input normalization** for improved utility under differential privacy
- **Support for both CDP and LDP modes**
- **Compatible with NVIDIA CUDA and AMD ROCm GPUs**

## Computational Infrastructure

> **Note**: The experiments reported in this paper were conducted on an **AMD-based supercomputer** equipped with AMD Instinct MI250X GPUs. The infrastructure leveraged ROCm (Radeon Open Compute) for high-performance distributed training, enabling large-scale federated learning simulations with hundreds of clients.

### Why AMD GPUs?

Our research infrastructure utilizes AMD Instinct accelerators for:
- **Scalability**: Parallelizing large-scale client simulations across multiple GPUs
- **Performance**: High memory bandwidth (3.2 TB/s on MI250X) for gradient computations
- **ROCm Ecosystem**: Native PyTorch support via `torch.cuda` API compatibility

### Reproducibility on Other Hardware

All experiments rely only on **standard PyTorch operations** and are fully compatible with:
- **NVIDIA GPUs**: CUDA 11.8+ (tested on A100, V100, RTX 3090)
- **AMD GPUs**: ROCm 5.6+ (tested on MI250X, MI210)
- **CPU**: For debugging and small-scale testing

Equivalent results can be reproduced on commodity hardware with appropriate runtime scaling.

## Repository Structure

```
LDP-MIC/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── src/                      # Core implementation
│   ├── FedAverage.py        # Main federated averaging entry point
│   ├── FedUser.py           # Client-side LDP/CDP implementation
│   ├── FedServer.py         # Server-side aggregation
│   ├── modelUtil.py         # Model architectures (MIC, InputNorm)
│   ├── mic_utils.py         # MIC computation utilities
│   ├── datasets.py          # Dataset loading and partitioning
│   ├── dataloader.py        # Data loader utilities
│   ├── compare_methods.py   # Method comparison script
│   └── quick_test.py        # Quick validation test
├── scripts/                  # Execution scripts
│   ├── hpc/                 # HPC cluster scripts
│   └── E1-E4_*.sh           # Experiment scripts
├── notebooks/                # Jupyter notebooks for analysis
│   ├── 03_label_flipping_robustness.ipynb
│   ├── 04_gradient_leakage.ipynb
│   └── 05_backdoor_resilience.ipynb
├── configs/                  # Experiment configurations
├── data/                     # Dataset directory
└── results/                  # Output directory
```

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- GPU with CUDA 11.8+ (NVIDIA) or ROCm 5.6+ (AMD)

### Setup

1. Clone the repository:
```bash
git clone https://anonymous.4open.science/r/LDP-MIC
cd LDP-MIC
```

2. Create conda environment:
```bash
conda create -n ldpmic python=3.10
conda activate ldpmic
```

3. Install PyTorch:
```bash
# For NVIDIA GPUs (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For AMD GPUs (ROCm):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

4. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run a quick test:
```bash
python src/quick_test.py
```

### Run LDP-MIC on MNIST:
```bash
python src/FedAverage.py \
    --data mnist \
    --nclient 100 \
    --nclass 10 \
    --ncpc 2 \
    --model mnist_fully_connected_MIC \
    --mode LDP \
    --round 150 \
    --epsilon 8
```

### Compare methods:
```bash
python src/compare_methods.py
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Dataset name | mnist |
| `--nclient` | Number of clients | 100 |
| `--nclass` | Number of classes | 10 |
| `--ncpc` | Classes per client (non-IID) | 2 |
| `--model` | Model architecture | mnist_fully_connected_IN |
| `--mode` | Privacy mode: LDP or CDP | LDP |
| `--round` | Number of FL rounds | 150 |
| `--epsilon` | Privacy budget ε | 8 |
| `--lr` | Learning rate | 0.1 |
| `--sr` | Client sample rate | 1.0 |
| `--physical_bs` | Max physical batch size (Opacus) | 3 |

## Datasets

All datasets used in this paper are publicly available:

| Dataset | Source | Samples | Classes | Task |
|---------|--------|---------|---------|------|
| Adult Census Income (ACI) | UCI ML Repository | 48,842 | 2 | Income prediction (>$50K) |
| MNIST | torchvision | 70,000 | 10 | Digit recognition |
| Fashion-MNIST | torchvision | 70,000 | 10 | Clothing classification |
| EMNIST | torchvision | 814,255 | 62 | Character recognition |
| CH-MNIST | CASIA | 10,000 | 10 | Chinese handwriting |

### Non-IID Partitioning

- **Adult Census Income**: Partitioned by occupation attribute across clients
- **MNIST variants**: Dirichlet-based label-skew partitioning (α=0.5)

Datasets are automatically downloaded on first run (except CH-MNIST which requires manual setup).

## Available Models

| Model | Description |
|-------|-------------|
| `mnist_fully_connected_MIC` | MNIST MLP with MIC-based normalization |
| `mnist_fully_connected_IN` | MNIST MLP with standard InputNorm |
| `mnist_fully_connected` | MNIST MLP baseline (no normalization) |
| `resnet18_MIC` | ResNet-18 with MIC normalization |
| `resnet18_IN` | ResNet-18 with InputNorm |
| `alexnet_MIC` | AlexNet with MIC normalization |
| `purchase_fully_connected_MIC` | Purchase dataset MLP with MIC |

## Experiment Scripts

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `E1_*.sh` | Privacy-utility tradeoff | Figure 3-4 |
| `E2_*.sh` | Varying privacy budgets | Figure 4 |
| `E3_*.sh` | Convergence analysis | Figure 5-6 |
| `E4_*.sh` | Additional experiments | Table 2 |

## Hardware Requirements

### Minimum
- CPU: 8+ cores
- RAM: 16GB
- GPU: 8GB VRAM (NVIDIA or AMD)
- Storage: 10GB

### Estimated Runtime

| Hardware | Full Reproduction |
|----------|-------------------|
| Single GPU (8GB) | ~24 hours |
| Single GPU (24GB) | ~12 hours |
| Multi-GPU cluster | ~3-6 hours |

## MIC Computation

The MIC (Maximum Information Coefficient) computation supports multiple backends:
1. **minepy** library (recommended) - original MINE algorithm
2. **scikit-learn** mutual information (fallback)
3. **Correlation-based** approximation (last resort)

To install minepy:
```bash
pip install minepy
```

## Expected Results

Results should match paper within ±2% due to randomness:

| Dataset | ε | LDP-MIC | Standard LDP | Gap |
|---------|---|---------|--------------|-----|
| Adult (ACI) | 10 | ~69.5% | ~59.5% | +10% |
| MNIST | 8 | ~89% | ~82% | +7% |
| Fashion-MNIST | 8 | ~75% | ~65% | +10% |
| EMNIST | 8 | ~75% | ~65% | +10% |

## Troubleshooting

### CUDA/ROCm Out of Memory
```bash
python src/FedAverage.py --physical_bs 1 ...
```

### AMD GPU Issues
Ensure ROCm is properly installed:
```bash
rocm-smi  # Check GPU status
python -c "import torch; print(torch.cuda.is_available())"
```

### Opacus Compatibility
The code includes a fix for PyTorch 2.6+ compatibility:
```python
torch.load = functools.partial(torch.load, weights_only=False)
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
[Citation will be added after de-anonymization]
```
