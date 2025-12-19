#!/bin/bash
#SBATCH --job-name=build_msmarco_faiss
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
<<<<<<< HEAD:verl/scripts/build_fiqa_faiss.sh
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_fiqa_faiss_%j.out
#SBATCH --error=logs/build_fiqa_faiss_%j.err
=======
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_msmarco_faiss_%j.out
#SBATCH --error=logs/build_msmarco_faiss_%j.err
>>>>>>> 6b374a5c2cc7b0f85e18901768aed138fa61c3b8:verl/scripts/retrieval/build_fiqa_faiss.sh

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv

cd /gpfs/home4/scur1900/darling_davide

<<<<<<< HEAD:verl/scripts/build_fiqa_faiss.sh
python3 verl/scripts/build_beir_faiss_index.py \
  --beir-dataset /home/scur1900/scratch_shared/fiqa \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --output-dir /home/scur1900/scratch_shared/fiqa/faiss_index \
  --nlist 256 \
  --m 32 \
  --batch-size 256 \
  --device cuda
=======
python3 /gpfs/home4/scur1900/darling_lukas/verl/scripts/build_beir_faiss_index.py \
  --beir-dataset /home/scur1900/scratch_shared/msmarco \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --output-dir /home/scur1900/scratch_shared/msmarco/faiss \
  --nlist 256 \
  --m 16 \
  --batch-size 64 \
  --device cuda \
  --max-docs 1000
>>>>>>> 6b374a5c2cc7b0f85e18901768aed138fa61c3b8:verl/scripts/retrieval/build_fiqa_faiss.sh

echo "FAISS index build complete!"
