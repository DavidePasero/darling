#!/bin/bash
#SBATCH --job-name=pyserini
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=logs/pyserini_%j.out
#SBATCH --error=logs/pyserini_%j.err

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv
conda clean --all

pip install pyserini faiss-cpu

python -c "from pyserini.search.faiss import FaissSearcher; FaissSearcher.from_prebuilt_index('msmarco-v1-passage.bge-base-en-v1.5', 'BAAI/bge-base-en-v1.5')"
