# Reproducing Paper Results

This document provides step-by-step instructions to reproduce all experiments.

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# Create environment
conda create -n ldpmic python=3.10
conda activate ldpmic

# Install PyTorch
# For NVIDIA GPUs (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For AMD GPUs (ROCm):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, GPU: {torch.cuda.is_available()}')"
python -c "import opacus; print(f'Opacus: {opacus.__version__}')"
python src/quick_test.py
```

---

## Dataset Preparation

Datasets are automatically downloaded on first run. Supported datasets:

| Dataset | Auto-download | Size |
|---------|---------------|------|
| MNIST | Yes | ~50MB |
| Fashion-MNIST | Yes | ~50MB |
| CIFAR-10 | Yes | ~170MB |
| CIFAR-100 | Yes | ~170MB |
| EMNIST | Yes | ~500MB |
| Purchase | Manual | ~100MB |
| CH-MNIST | Manual | ~20MB |

### Manual Dataset Setup (if needed)

**Purchase Dataset:**
```bash
mkdir -p data/purchase
# Download from UCI ML Repository
# Place dataset_purchase file in data/purchase/
```

---

## Main Experiments

### Experiment 1: Privacy-Utility Tradeoff

#### MNIST (Figure 4a)

```bash
# LDP-MIC
python src/FedAverage.py --data mnist --nclient 100 --ncpc 2 \
    --model mnist_fully_connected_MIC --mode LDP --round 150 --epsilon 8 --E 1

# Baseline LDP (InputNorm)
python src/FedAverage.py --data mnist --nclient 100 --ncpc 2 \
    --model mnist_fully_connected_IN --mode LDP --round 150 --epsilon 8 --E 1

# No privacy baseline
python src/FedAverage.py --data mnist --nclient 100 --ncpc 2 \
    --model mnist_fully_connected --mode CDP --round 150 --epsilon 1000 --E 1
```

**Expected Results (ε=8):**
| Method | Accuracy |
|--------|----------|
| LDP-MIC | ~89% |
| LDP-IN | ~82% |
| No-DP | ~95% |

#### Fashion-MNIST (Figure 4b)

```bash
python src/FedAverage.py --data fashionmnist --nclient 100 --ncpc 2 \
    --model mnist_fully_connected_MIC --mode LDP --round 150 --epsilon 8 --E 1
```

#### CIFAR-10 (Figure 4c)

```bash
python src/FedAverage.py --data cifar10 --nclient 100 --ncpc 2 \
    --model resnet18_MIC --mode LDP --round 150 --epsilon 8 --E 1
```

#### Run All E1 Experiments

```bash
# Using provided scripts
bash scripts/E1_mnist.sh
bash scripts/E1_fashionmnist.sh
bash scripts/E1_cifar10.sh
bash scripts/E1_emnist.sh
```

---

### Experiment 2: Varying Privacy Budgets

Test multiple epsilon values:

```bash
for eps in 2 4 6 8 10; do
    python src/FedAverage.py --data mnist --nclient 100 --ncpc 2 \
        --model mnist_fully_connected_MIC --mode LDP --round 150 \
        --epsilon $eps --E 2
done
```

---

### Experiment 3: Convergence Analysis

Track accuracy over rounds:

```bash
# Modify FedAverage.py to save per-round accuracy, or use:
python src/FedAverage.py --data mnist --nclient 100 --ncpc 2 \
    --model mnist_fully_connected_MIC --mode LDP --round 150 --epsilon 8 --E 3
```

---

### Experiment 4: Attack Evaluation

See notebooks:
- `notebooks/03_label_flipping_robustness.ipynb`
- `notebooks/04_gradient_leakage.ipynb`
- `notebooks/05_backdoor_resilience.ipynb`

---

## Method Comparison

Run automated comparison:

```bash
python src/compare_methods.py
```

This compares:
- No-DP baseline
- Standard LDP (PrivateFL)
- LDP-MIC (proposed)

---

## HPC Cluster Execution (Optional)

For large-scale experiments on SLURM clusters:

```bash
# Edit scripts/hpc/run_experiment.slurm for your cluster
sbatch scripts/hpc/run_experiment.slurm
```

---

## Expected Runtime

| Experiment | Single GPU (8GB) | Single GPU (24GB) |
|------------|------------------|-------------------|
| MNIST (100 clients, 150 rounds) | ~2 hours | ~1 hour |
| CIFAR-10 (100 clients, 150 rounds) | ~4 hours | ~2 hours |
| Full reproduction | ~24 hours | ~12 hours |

---

## Verifying Results

Results are saved to `log/E{experiment_number}/`:

```bash
ls log/E1/
# mnist_100_2_LDP_mnist_fully_connected_MIC_8.csv
```

CSV format:
```
data,num_client,ncpc,mode,model,epsilon,accuracy
mnist,100,2,LDP,mnist_fully_connected_MIC,8,0.89
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce physical batch size
python src/FedAverage.py --physical_bs 1 ...
```

### Slow Training

```bash
# Reduce number of clients for testing
python src/FedAverage.py --nclient 10 --round 50 ...
```

### Opacus Errors

Ensure PyTorch and Opacus versions are compatible:
```bash
pip install opacus>=1.4.0
```
