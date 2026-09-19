#!/bin/bash
#SBATCH --job-name=hagai-task
#SBATCH --partition=gpu-wolfson
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

cd /home/iscb/wolfson/hagairavid/converge
set -a; source .env; set +a
export SKEMPI_USE_SMALL_CHECKPOINTS=0
export SKEMPI_USE_SAPROT_STRUCTURE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="$1"

echo "=== overfitting sanity check (regression head) ==="
uv run python -m dl.training.sanity_check --active-heads regression
SANITY_STATUS=$?
if [ $SANITY_STATUS -ne 0 ]; then
    echo "Sanity check failed (exit $SANITY_STATUS) -- aborting before real training."
    exit $SANITY_STATUS
fi

echo "=== real training: $CONFIG ==="
uv run python train.py --config "$CONFIG" --mode train
