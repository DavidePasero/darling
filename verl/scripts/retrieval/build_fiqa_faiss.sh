#!/bin/bash
#SBATCH --job-name=build_msmarco_faiss
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_msmarco_faiss_%j.out
#SBATCH --error=logs/build_msmarco_faiss_%j.err

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv

cd /gpfs/home4/scur1900/darling_lukas

python3 /gpfs/home4/scur1900/darling_lukas/verl/scripts/build_beir_faiss_index.py \
  --beir-dataset /home/scur1900/scratch_shared/msmarco \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --output-dir /home/scur1900/scratch_shared/msmarco/faiss \
  --nlist 256 \
  --m 16 \
  --batch-size 64 \
  --device cuda \
  --max-docs 1000

echo "FAISS index build complete!"
