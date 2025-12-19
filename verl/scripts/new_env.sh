#!/bin/bash
#SBATCH --job-name=new_env
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:01:00
#SBATCH --output=logs/new_env_%j.out
#SBATCH --error=logs/new_env_%j.err

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
#conda create -n dataset_env python=3.10
conda activate dataset_env
conda clean --all

pip install "numpy<2.0" #\
#             "tensordict>=0.8.0,<=0.9.1" \
#             "sentence-transformers==2.2.2" \
#             "transformers==4.35.2" \
#             "huggingface-hub==0.22.2" \
#             faiss-cpu \
#             verl



python /home/scur1900/darling_lukas/verl/scripts/build_beir_bm25_index.py