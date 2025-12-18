#!/bin/bash
#SBATCH --job-name=build_fiqa_faiss
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_fiqa_faiss_%j.out
#SBATCH --error=logs/build_fiqa_faiss_%j.err

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv

cd /gpfs/home4/scur1900/darling_davide

python3 verl/scripts/build_beir_faiss_index.py \
  --beir-dataset /home/scur1900/scratch_shared/fiqa \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --output-dir /home/scur1900/scratch_shared/fiqa/faiss_index \
  --nlist 256 \
  --m 32 \
  --batch-size 256 \
  --device cuda

echo "FAISS index build complete!"
