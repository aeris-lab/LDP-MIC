#!/bin/bash
#SBATCH --job-name=ldpmic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ==============================================================================
# LDP-MIC Experiment Runner for HPC Clusters
# ==============================================================================
# This script is designed to work with both NVIDIA and AMD GPU clusters.
# Adjust the module loads below for your specific cluster configuration.
# ==============================================================================

# Load modules (uncomment appropriate lines for your cluster)
# 
# For NVIDIA CUDA clusters:
# module load python/3.10
# module load cuda/11.8
#
# For AMD ROCm clusters:
# module load python/3.10
# module load rocm/5.6

# Activate conda environment
source activate ldpmic

# Create logs directory if needed
mkdir -p logs

# Experiment parameters
DATA=${DATA:-mnist}
NCLIENT=${NCLIENT:-100}
NCLASS=${NCLASS:-10}
NCPC=${NCPC:-2}
MODEL=${MODEL:-mnist_fully_connected_MIC}
MODE=${MODE:-LDP}
ROUND=${ROUND:-150}
EPSILON=${EPSILON:-8}
EPOCHS=${EPOCHS:-1}

echo "============================================"
echo "LDP-MIC Experiment"
echo "============================================"
echo "Data: $DATA"
echo "Clients: $NCLIENT"
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "Epsilon: $EPSILON"
echo "Rounds: $ROUND"
echo "============================================"

# Run experiment
python src/FedAverage.py \
    --data $DATA \
    --nclient $NCLIENT \
    --nclass $NCLASS \
    --ncpc $NCPC \
    --model $MODEL \
    --mode $MODE \
    --round $ROUND \
    --epsilon $EPSILON \
    --E $EPOCHS

echo "Experiment completed!"
