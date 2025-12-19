#!/bin/bash
#SBATCH --job-name=try_retriever
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=logs/try_retriever_%j.out
#SBATCH --error=logs/try_retriever_%j.err

module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv
conda clean --all

export DEBUG_LOG=1

echo "========================================="
echo "Checking NumPy and FAISS installation"
echo "========================================="

# Check NumPy version and downgrade if needed
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
echo "Current NumPy version: $NUMPY_VERSION"

if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo "WARNING: NumPy 2.x detected. Downgrading to NumPy 1.x for FAISS compatibility..."
    pip install "numpy<2.0" --force-reinstall
    echo "NumPy downgraded successfully"
else
    echo "NumPy version is compatible with FAISS"
fi

# Remove both faiss-cpu and faiss-gpu to ensure clean installation
echo "Removing any existing FAISS installations..."
pip uninstall faiss-cpu faiss-gpu -y 2>/dev/null || true

# Install faiss-gpu with force reinstall
echo "Installing faiss-gpu (force reinstall)..."
pip install --force-reinstall --no-cache-dir faiss-gpu
echo "faiss-gpu installed successfully"
pip show faiss-gpu

echo "========================================="
echo "Verifying FAISS installation"
echo "========================================="

# Test FAISS import and show attributes
python3 -c "
import faiss
import sys
print(f'FAISS module location: {faiss.__file__}')
print(f'FAISS version: {faiss.__version__ if hasattr(faiss, \"__version__\") else \"Unknown\"}')
print(f'Has read_index: {hasattr(faiss, \"read_index\")}')
print(f'Has StandardGpuResources: {hasattr(faiss, \"StandardGpuResources\")}')
print(f'Available FAISS attributes: {[attr for attr in dir(faiss) if not attr.startswith(\"_\")][:20]}')
"

echo "========================================="
echo "FAISS installation complete"
echo "========================================="

 # pip install "numpy<2.0" \
#             "tensordict>=0.8.0,<=0.9.1" \
#             "sentence-transformers==2.2.2" \
#             "transformers==4.35.2" \
#             "huggingface-hub==0.22.2" \
#             faiss-cpu \
#             verl



python /home/scur1900/darling_lukas/verl/scripts/test_retrieval_system.py  --beir-dataset /home/scur1900/scratch_shared/msmarco \
 --retriever-type bm25 --device cuda --embedding-model "Qwen/Qwen3-Embedding-0.6B"  --faiss-index ~/scratch_shared/fiqa/faiss_index/embeddings_cache.npy \
 --faiss-id-mapping ~/scratch_shared/msmarco/id_mapping.pkl
