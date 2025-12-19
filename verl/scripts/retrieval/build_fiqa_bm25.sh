#!/bin/bash
#SBATCH --job-name=build_msmarco_bm25
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_msmarco_bm25_%j.out
#SBATCH --error=logs/build_msmarco_bm25_%j.err

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv

cd /gpfs/home4/scur1900/darling_lukas

python3 /gpfs/home4/scur1900/darling_lukas/verl/scripts/build_beir_bm25_index.py \
  --beir-dataset /home/scur1900/darling_lukas/verl/scripts/datasets/msmarco \
  --output-dir /home/scur1900/darling_lukas/verl/scripts/datasets/msmarco/bm25 \
  --threads 64 \
  --max-docs 600000

echo "msmarco bm25 index build complete!"
