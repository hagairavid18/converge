#!/bin/bash
#SBATCH --job-name=hagai-task
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

cd /home/iscb/wolfson/hagairavid/converge
set -a; source .env; set +a
export SKEMPI_USE_SMALL_CHECKPOINTS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== cache warm: records[${START_INDEX}:${END_INDEX}] ==="
uv run python -c "
from data.embedding_pipeline.pipeline import run_embedding_pipeline
from data.processed_io import load_mutation_records, load_sample_chain_maps

records = load_mutation_records()[${START_INDEX}:${END_INDEX}]
sample_chain_maps = load_sample_chain_maps()
processed = run_embedding_pipeline(records, sample_chain_maps)
print(f'cached {len(processed)} samples')
"
