# LDP-MIC: Correlation-Aware Local Differential Privacy for Federated Learning

## Overview

LDP-MIC is a correlation-adaptive local differential privacy framework for federated learning under untrusted aggregation. By adapting client-side noise to feature–target dependence via the Maximum Information Coefficient (MIC), LDP-MIC mitigates the utility degradation typical of LDP in heterogeneous, non-IID settings while enforcing all privacy guarantees locally.

## Key Features

- **Correlation-aware noise allocation** using Maximum Information Coefficient (MIC)
- **(ε,δ)-LDP guarantees** without requiring a trusted aggregator
- **Trust-based aggregation** for robustness against poisoning attacks
- **Linear scalability** with number of clients and feature dimensions
- **No additional communication overhead** compared to standard FedAvg

## Repository Structure

```
LDP-MIC/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── src/                      # Core implementation
│   ├── ldp_mic.py           # Main LDP-MIC algorithm
│   ├── mic_computation.py   # MIC calculation utilities
│   ├── noise_calibration.py # Adaptive sensitivity calibration
│   ├── trust_aggregation.py # Trust-based server aggregation
│   ├── attacks/             # Attack implementations
│   └── baselines/           # Baseline methods
├── configs/                  # Experiment configurations
├── notebooks/                # Jupyter notebooks for experiments
├── scripts/                  # Execution scripts
│   └── hpc/                 # HPC cluster scripts (optional)
├── data/                     # Dataset utilities
└── docs/                     # Additional documentation
```

**Note**: Some large-scale experiments (privacy-utility tradeoffs, convergence analysis, scalability) were conducted on an HPC cluster and are provided as Python scripts rather than notebooks for efficiency.

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

1. Clone the repository:

```bash
git clone https://anonymous.4open.science/r/LDP-MIC
cd LDP-MIC
```

2. Create a virtual environment:

```bash
conda create -n ldpmic python=3.10
conda activate ldpmic
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download datasets:

```bash
python data/download_datasets.py --all
```

## Quick Start

Run a basic privacy-utility experiment on the Adult Census Income dataset:

```bash
python scripts/run_privacy_utility.py --dataset adult --epsilon 1.0
```

Run with multiple privacy budgets:

```bash
python scripts/run_privacy_utility.py \
    --dataset adult \
    --epsilon 0.5 1.0 2.0 5.0 10.0 \
    --seeds 5
```

## Reproducing Paper Results

### Main Experiments

| Figure/Table | Description | Script/Notebook | Est. Time (1 GPU) |
|--------------|-------------|-----------------|-------------------|
| Figure 3 | Privacy-utility tradeoff (Adult) | `scripts/run_privacy_utility.py` | ~2 hours |
| Figure 4 | Privacy-utility tradeoff (MNIST variants) | `scripts/run_privacy_utility.py` | ~4 hours |
| Figure 5 | Convergence analysis (Adult) | `scripts/run_convergence.py` | ~2 hours |
| Figure 6 | Convergence analysis (MNIST variants) | `scripts/run_convergence.py` | ~3 hours |
| Figure 7 | Label-flipping robustness | `notebooks/03_label_flipping_robustness.ipynb` | ~1 hour |
| Figure 8 | Gradient leakage resilience | `notebooks/04_gradient_leakage.ipynb` | ~30 min |
| Figure 9 | Scalability analysis | `scripts/run_scalability.py` | ~2 hours |
| Table 2 | Backdoor attack success rate | `notebooks/05_backdoor_resilience.ipynb` | ~1 hour |
| Table 3 | Trust-based detection performance | `notebooks/03_label_flipping_robustness.ipynb` | ~1 hour |

### Running All Experiments

To reproduce all main results:

```bash
bash scripts/run_all_experiments.sh
```

Results will be saved to `results/figures/` and `results/tables/`.

## Hardware Requirements

### Minimum (for verification)

- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070)
- Storage: 10GB free space
- Time: ~15 hours for full reproduction

### Recommended (for faster execution)

- CPU: 16+ cores
- RAM: 32GB
- GPU: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090/4090)
- Time: ~6 hours for full reproduction

### Paper Experiments

Experiments in the paper were conducted on a **multi-GPU HPC cluster** to parallelize large-scale client simulations and accelerate repeated experimental runs. However, all experiments rely only on **standard PyTorch operations** and do not require specialized hardware or interconnect features. Equivalent results can be reproduced on commodity multi-GPU systems.

| Hardware | Estimated Time |
|----------|----------------|
| Single RTX 3090/4090 | ~24 hours |
| Multi-GPU workstation (4x GPUs) | ~6 hours |
| Cloud instance (8x V100) | ~3 hours |

## Datasets

All datasets used are publicly available benchmarks:

| Dataset | Source | Size | Task |
|---------|--------|------|------|
| Adult Census Income | UCI ML Repository | 48,842 samples | Binary classification |
| MNIST | Yann LeCun | 70,000 samples | 10-class classification |
| Fashion-MNIST | Zalando Research | 70,000 samples | 10-class classification |
| EMNIST | NIST | 814,255 samples | 62-class classification |
| CH-MNIST | CASIA | 10,000 samples | 10-class classification |

Datasets are automatically downloaded when running:

```bash
python data/download_datasets.py --all
```

## Configuration

Experiment configurations are stored in YAML files under `configs/`. Example configuration for Adult dataset:

```yaml
dataset:
  name: adult
  num_clients: 50
  partition: non_iid_occupation

model:
  architecture: mlp
  hidden_layers: [128, 64, 32]

training:
  rounds: 400
  local_epochs: 1
  batch_size: 128
  learning_rate: 0.01
  clip_bound: 1.0

privacy:
  epsilon: [0.5, 1.0, 2.0, 5.0, 10.0]
  delta: 1e-5
  mic_budget_ratio: 0.2

aggregation:
  sample_rate: 0.2
  trust_weights: [0.4, 0.3, 0.3]
  trust_threshold: 0.4
```

## Algorithm Overview

LDP-MIC operates in three phases on each client:

1. **Phase 1 - Correlation Analysis**: Compute privatized MIC scores between features and target
2. **Phase 2 - Noise Calibration**: Allocate privacy budget inversely proportional to MIC scores
3. **Phase 3 - Training & Privatization**: Train on locally privatized data and send updates

The server performs trust-aware aggregation using only privatized updates, requiring no trust assumptions.

## Expected Results

Results should match the paper within ±1-2% due to random seed variation. Key expected outcomes:

- **Privacy-Utility (Adult, ε=1.0)**: LDP-MIC ~61% vs Standard LDP ~54%
- **Privacy-Utility (MNIST, ε=10.0)**: LDP-MIC ~90% vs Standard LDP ~82%
- **Label-Flipping Defense**: 95% TPR, 2% FPR by round 50
- **Backdoor ASR Reduction**: 30.6 percentage points vs No-DP

## Troubleshooting

### Common Issues

**CUDA out of memory**: Reduce batch size in config or use gradient accumulation:
```bash
python scripts/run_privacy_utility.py --batch_size 32 --grad_accumulation 4
```

**MIC computation slow**: Enable parallel computation:
```bash
python scripts/run_privacy_utility.py --mic_parallel --mic_workers 4
```

**Dataset download fails**: Manually download from source URLs listed in `data/README.md`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
[Citation will be added after de-anonymization]
```

## Acknowledgments

We thank the anonymous reviewers for their valuable feedback.
