#!/bin/bash
#SBATCH --job-name=beir_faiss
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=logs/faiss_%j.out
#SBATCH --error=logs/faiss_%j.err

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv

conda clean --all

pip install pyserini
#pip install -U sentence-transformers

python /home/scur1900/darling_lukas/verl/scripts/build_beir_faiss_index.py
