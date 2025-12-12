#!/bin/bash
#SBATCH --job-name=try_retriever
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --time=00:01:00
#SBATCH --output=logs/try_retriever_%j.out
#SBATCH --error=logs/try_retriever_%j.err

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv
conda clean --all

# pip install "numpy<2.0" \
#             "tensordict>=0.8.0,<=0.9.1" \
#             "sentence-transformers==2.2.2" \
#             "transformers==4.35.2" \
#             "huggingface-hub==0.22.2" \
#             faiss-cpu \
#             verl



python /home/scur1900/darling_davide/verl/scripts/test_retrieval_system.py
