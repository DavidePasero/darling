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

export DEBUG_LOG=1

CHECKPOINT_DIR=/home/scur1900/scratch_shared/merged_checkpoints
LOGDIR="/home/scur1900/logs/vllm_eval"

ACTOR_PATH=/home/scur1900/scratch_shared/scratch-shared-alisia/merged/global_step_250_llama_instruct/actor
ACTOR_NAME=darling_trained_bm25

VLLM_LOG="$LOGDIR/vllm_eval_actor_${SLURM_JOB_ID}.log"
VLLM_PORT=8000
CONTAINER=/projects/2/managed_datasets/containers/vllm/vllm_25.09.sif

echo "Starting vLLM server for actor: $ACTOR_NAME..."
CUDA_VISIBLE_DEVICES=0 \
apptainer exec --nv -B $PWD $CONTAINER bash -lc "
  python3 -m vllm.entrypoints.openai.api_server \
    --model $ACTOR_PATH \
    --served-model-name $ACTOR_NAME \
    --dtype float16 \
    --max-model-len 8192 \
    --port $VLLM_PORT \
    --host 0.0.0.0
" > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
for i in {1..120}; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" || true)
  if [[ "$STATUS" == "200" ]]; then break; fi
  sleep 2
done


python /home/scur1900/darling_lukas/verl/scripts/test_retrieval_system.py \
 --beir-dataset /home/scur1900/scratch_shared/msmarco \
 --retriever-type bm25 --device cuda --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
 --model-name $ACTOR_NAME \
 --use-vllm \
 --vllm-port $VLLM_PORT \
 --num-rewrites 5 \
 --faiss-index ~/scratch_shared/fiqa/faiss_index/embeddings_cache.npy \
 --faiss-id-mapping ~/scratch_shared/msmarco/id_mapping.pkl \
