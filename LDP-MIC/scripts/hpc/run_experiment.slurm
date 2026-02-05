#!/bin/bash
#SBATCH --job-name=ldp-mic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu          # Adjust to your cluster
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ============================================================
# LDP-MIC Experiment Script
# Adjust module loads for your cluster environment
# ============================================================

# Create logs directory if not exists
mkdir -p logs

# Load modules (adjust for your cluster)
# For NVIDIA GPUs:
# module load python/3.10
# module load cuda/11.8

# For AMD GPUs (ROCm):
# module load python/3.10
# module load rocm/5.6

# Activate conda environment
source activate ldpmic

# Set environment variables for better GPU utilization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================
# Run LDP-MIC Experiment
# ============================================================

# Example: Run MIC-based federated learning on MNIST
python src/FedAverage.py \
    --data mnist \
    --nclient 100 \
    --nclass 10 \
    --ncpc 2 \
    --model mnist_fully_connected_MIC \
    --mode LDP \
    --round 150 \
    --epsilon 8 \
    --lr 0.1 \
    --E 1

echo "Experiment completed at $(date)"
