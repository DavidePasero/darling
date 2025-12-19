#!/bin/bash
#SBATCH --job-name=try_retriever
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=logs_retr/try_retriever_%j.out
#SBATCH --error=logs_retr/try_retriever_%j.err

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv
conda clean --all

export DEBUG_LOG=1


 # pip install "numpy<2.0" \
#             "tensordict>=0.8.0,<=0.9.1" \
#             "sentence-transformers==2.2.2" \
#             "transformers==4.35.2" \
#             "huggingface-hub==0.22.2" \
#             faiss-cpu \
#             verl



python /home/scur1900/darling_lukas/verl/scripts/test_retrieval_system.py  --beir-dataset /home/scur1900/scratch_shared/msmarco \
 --retriever-type faiss --device cuda --embedding-model "Qwen/Qwen3-Embedding-0.6B"  --faiss-index ~/scratch_shared/msmarco/faiss/faiss_index.faiss \
 --faiss-id-mapping ~/scratch_shared/msmarco/faiss/id_mapping.pkl --k 20 --search_batch_size=512
