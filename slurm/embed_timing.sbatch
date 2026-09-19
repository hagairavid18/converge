#!/bin/bash
#SBATCH --job-name=hagai-task
#SBATCH --partition=gpu-wolfson
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

cd /home/iscb/wolfson/hagairavid/converge
set -a; source .env; set +a
export SKEMPI_USE_SMALL_CHECKPOINTS=0

uv run python -c "
import time
from data.embedding_pipeline.pipeline import run_embedding_pipeline
from data.processed_io import load_mutation_records, load_sample_chain_maps

records = load_mutation_records()
sample_chain_maps = load_sample_chain_maps()

N = 5
t0 = time.time()
processed = run_embedding_pipeline(records, sample_chain_maps, sample_id_limit=N)
elapsed = time.time() - t0
print(f'processed {len(processed)} samples in {elapsed:.1f}s ({elapsed/len(processed):.1f}s/sample)')
"
