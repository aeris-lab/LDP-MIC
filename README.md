# LDP-MIC: Correlation-Aware Local Differential Privacy for Federated Learning

## Overview

LDP-MIC is a correlation-adaptive local differential privacy framework for federated learning under untrusted aggregation. By adapting client-side noise to feature–target dependence via the Maximum Information Coefficient (MIC), LDP-MIC mitigates the utility degradation typical of LDP in heterogeneous, non-IID settings while enforcing all privacy guarantees locally.

## Key Features

- **Correlation-aware noise allocation** using Maximum Information Coefficient (MIC)
- **(ε,δ)-LDP guarantees** without requiring a trusted aggregator
- **MIC-based input normalization** for improved utility under differential privacy
- **Support for both CDP and LDP modes**
- **Compatible with NVIDIA CUDA and AMD ROCm GPUs**

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
│   ├── E1_*.sh              # Experiment 1 scripts (various datasets)
│   └── setup.sh             # Environment setup
├── configs/                  # Experiment configurations
├── notebooks/                # Jupyter notebooks for analysis
├── data/                     # Dataset directory
├── docs/                     # Additional documentation
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
# For NVIDIA GPUs:
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

### Run baseline LDP (without MIC):
```bash
python src/FedAverage.py \
    --data mnist \
    --nclient 100 \
    --model mnist_fully_connected_IN \
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
| `--data` | Dataset: mnist, cifar10, cifar100, fashionmnist, emnist, purchase, chmnist | mnist |
| `--nclient` | Number of clients | 100 |
| `--nclass` | Number of classes | 10 |
| `--ncpc` | Classes per client (non-IID) | 2 |
| `--model` | Model architecture (see below) | mnist_fully_connected_IN |
| `--mode` | Privacy mode: LDP or CDP | LDP |
| `--round` | Number of FL rounds | 150 |
| `--epsilon` | Privacy budget ε | 8 |
| `--lr` | Learning rate | 0.1 |
| `--sr` | Client sample rate | 1.0 |
| `--physical_bs` | Max physical batch size (Opacus) | 3 |

### Available Models

| Model | Description |
|-------|-------------|
| `mnist_fully_connected_MIC` | MNIST MLP with MIC-based normalization |
| `mnist_fully_connected_IN` | MNIST MLP with standard InputNorm |
| `mnist_fully_connected` | MNIST MLP baseline |
| `resnet18_MIC` | ResNet-18 with MIC normalization |
| `resnet18_IN` | ResNet-18 with InputNorm |
| `alexnet_MIC` | AlexNet with MIC normalization |
| `purchase_fully_connected_MIC` | Purchase dataset MLP with MIC |

## Reproducing Paper Results

### Main Experiments

| Figure/Table | Description | Command |
|--------------|-------------|---------|
| Figure 3 | Privacy-utility (Adult) | `bash scripts/E1_purchase.sh` |
| Figure 4 | Privacy-utility (MNIST variants) | `bash scripts/E1_mnist.sh` |
| Figure 5 | Convergence (Adult) | See scripts/E3_*.sh |
| Figure 7 | Label-flipping robustness | `notebooks/03_label_flipping.ipynb` |
| Table 2 | Backdoor attack | `notebooks/05_backdoor.ipynb` |

### Run all experiments for a dataset:
```bash
# MNIST experiments
bash scripts/E1_mnist.sh

# CIFAR-10 experiments  
bash scripts/E1_cifar10.sh

# Fashion-MNIST experiments
bash scripts/E1_fashionmnist.sh
```

## Hardware Requirements

### Minimum
- CPU: 8+ cores
- RAM: 16GB
- GPU: 8GB VRAM (NVIDIA or AMD)
- Storage: 10GB

### Paper Experiments
Experiments were conducted on a **multi-GPU HPC cluster with AMD GPUs**. However, all code uses standard PyTorch operations and works on any CUDA or ROCm compatible GPU.

| Hardware | Estimated Time (Full Reproduction) |
|----------|-----------------------------------|
| Single GPU (8GB) | ~24 hours |
| Single GPU (24GB) | ~12 hours |
| Multi-GPU (4x) | ~6 hours |

## Datasets

All datasets are publicly available:

| Dataset | Source | Samples | Classes |
|---------|--------|---------|---------|
| MNIST | torchvision | 70,000 | 10 |
| Fashion-MNIST | torchvision | 70,000 | 10 |
| CIFAR-10 | torchvision | 60,000 | 10 |
| CIFAR-100 | torchvision | 60,000 | 100 |
| EMNIST | torchvision | 814,255 | 62 |
| Purchase | UCI | 197,324 | 100 |
| CH-MNIST | Custom | 10,000 | 10 |

Datasets are automatically downloaded on first run.

## MIC Computation

The MIC (Maximum Information Coefficient) computation uses:
1. **minepy** library (if installed) - original MINE algorithm
2. **scikit-learn** mutual information (fallback)
3. **Correlation-based** approximation (last resort)

To install minepy (optional):
```bash
pip install minepy
```

## Expected Results

Results should match paper within ±2% due to randomness:

| Dataset | ε | LDP-MIC | Standard LDP |
|---------|---|---------|--------------|
| MNIST | 8 | ~89% | ~82% |
| CIFAR-10 | 8 | ~58% | ~51% |
| Fashion-MNIST | 8 | ~78% | ~71% |

## Troubleshooting

### CUDA Out of Memory
Reduce physical batch size:
```bash
python src/FedAverage.py --physical_bs 1
```

### Opacus Compatibility
The code includes a fix for PyTorch 2.6+ compatibility:
```python
torch.load = functools.partial(torch.load, weights_only=False)
```

### AMD GPU Issues
Ensure ROCm is properly installed:
```bash
rocm-smi  # Check GPU status
python -c "import torch; print(torch.cuda.is_available())"
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
[Citation will be added after de-anonymization]
```
