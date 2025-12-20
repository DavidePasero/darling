#!/bin/bash
#SBATCH --job-name=RER_BEIR_RETR_DIV
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/beir_retrieval_rerank_%j.out
#SBATCH --error=logs/beir_retrieval_rerank_%j.err

##############################
# CONFIGURATION
##############################
CHECKPOINT_DIR=/home/scur1900/scratch-shared/checkpoints
USER=davide_copy 

# BEIR Dataset Configuration
BEIR_DATASET="msmarco"
BEIR_DIR="/home/scur1900/scratch_shared/${BEIR_DATASET}"
QUERIES_FILE="${BEIR_DIR}/queries.jsonl"
TRAIN_QRELS="${BEIR_DIR}/qrels/train.tsv"
DEV_QRELS="${BEIR_DIR}/qrels/dev.tsv"
QRELS_FILE="qrels/train.tsv"

# Retriever Configuration
RETRIEVER_TYPE="faiss"  # Options: "faiss" or "bm25"
FAISS_INDEX="${BEIR_DIR}/faiss/faiss_index.faiss"
FAISS_ID_MAPPING="${BEIR_DIR}/faiss/id_mapping.pkl"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
BM25_INDEX="${BEIR_DIR}/bm25_index/index"
BM25_ID_MAPPING="${BEIR_DIR}/bm25_index/id_mapping.pkl"
BM25_K1=0.9
BM25_B=0.4
BM25_NUM_THREADS=128

# Reranker Configuration
RETRIEVAL_REWARD_METHOD="reranker+ndcg" # Options: "ndcg" or "reranker+ndcg"
RERANKER_PORT=8003

# Model Configuration
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE=256
N_REWRITES=10
MAX_PROMPT_LEN=256
MAX_RESPONSE_LEN=512
LEARNING_RATE=1e-6
EPOCHS=500
PROMPT_EXTENDER=rewrite

# Reward Configuration
K=20

# Diversity Reward Configuration (DARLING-style)
DIVERSITY_ENABLED=true
DIVERSITY_METHOD="ngram"  # Options: "ngram", "partition_vllm", "unlikelihood"
NGRAM_N=4
CLASSIFIER_MODEL="/home/scur1900/models/dogtooth"
PARTITION_REWARD_PATH="verl/verl/utils/reward_score/partition_reward_vllm_serve.py"

# Reward Combination Configuration
MULTIPLICATIVE=false
LAMBDA_RM=1.0
LAMBDA_RULE=1.0

# FAISS Performance Configuration
NPROBE=8
ENCODING_BATCH_SIZE=256
SEARCH_BATCH_SIZE=256

# Logging Configuration
LOG_REWRITE_FREQ=100
LOG_NUM_QUERIES=10
LOG_NUM_REWRITES=10

# Resource Configuration
GPU_MEMORY_UTIL=0.6
OFFLOAD_PARAMS=true
OFFLOAD_OPTIMIZER=false

##############################
# ENVIRONMENT SETUP
##############################
module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
# Fix for openjdk activation bug
export JAVA_HOME=""
export JAVA_LD_LIBRARY_PATH=""

conda activate verlenv

# Fix MKL threading layer incompatibility with vLLM
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
export HYDRA_FULL_ERROR=1
export DEBUG_LOG=0
export DIV_DEBUG_LOG=0
export RERANKER_PORT=${RERANKER_PORT} # Export for python script

# Enable strict error checking AFTER conda activation
set -euo pipefail
export WANDB_API_KEY=9dbe3f48ba14149757b7b5b73b02ef3e9cab94b5

# Set working directory and PYTHONPATH
WORK_DIR="/gpfs/home4/scur1900/darling_${USER}"
cd ${WORK_DIR}
export PYTHONPATH="${WORK_DIR}/verl:${PYTHONPATH:-}"

echo "========== ENV INFO =========="
echo "Node: $(hostname)"
echo "Working Directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "=============================="

##############################
# UTILITY GPU SETUP (GPU 2)
# We group all "Sidecar" services on GPU 2:
# 1. FAISS Server
# 2. Embedding vLLM
# 3. Reranker Server
##############################
SERVICE_GPU=2

##############################
# 1. VLLM CLASSIFIER SERVER (GPU 0 - Shared with Training)
##############################
VLLM_CLASSIFIER_PID=""
if [ "${DIVERSITY_ENABLED}" = "true" ] && [ "${DIVERSITY_METHOD}" = "partition_vllm" ]; then
    echo "========================================="
    echo "Setting up vLLM Classifier Server (Diversity)"
    echo "========================================="
    
    export VLLM_SERVER_HOSTNAME="127.0.0.1"
    export VLLM_CLASSIFIER_PORT=8000
    export PYTHONUNBUFFERED=1
    CONTAINER=/projects/2/managed_datasets/containers/vllm/vllm_25.09.sif

    echo "Starting vLLM classifier on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 \
    apptainer exec --nv -B $PWD $CONTAINER bash -lc "
      python3 -m vllm.entrypoints.openai.api_server \
        --model ${CLASSIFIER_MODEL} \
        --task classify \
        --dtype float16 \
        --max-model-len 4096 \
        --port ${VLLM_CLASSIFIER_PORT} \
        --host 0.0.0.0 \
        --served-model-name similarity_gpu_0
    " > vllm_classifier.log 2>&1 &

    VLLM_CLASSIFIER_PID=$!
    echo "vLLM Classifier PID = ${VLLM_CLASSIFIER_PID}"

    # Wait for health check...
    echo "Waiting for vLLM classifier..."
    for i in {1..60}; do
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VLLM_CLASSIFIER_PORT}/health" || true)
        if [[ "$STATUS" == "200" ]]; then break; fi
        sleep 2
    done
    if [[ "$STATUS" != "200" ]]; then
        echo "ERROR: vLLM classifier failed"
        tail -n 20 vllm_classifier.log
        exit 1
    fi
    echo "vLLM Classifier Ready."
fi

##############################
# 2. RERANKER SERVER (GPU 2)
##############################
RERANKER_SERVER_PID=""
if [[ "$RETRIEVAL_REWARD_METHOD" == *"reranker"* ]]; then
    echo "========================================="
    echo "Setting up Reranker Server"
    echo "========================================="
    
    echo "Starting Reranker Server on GPU ${SERVICE_GPU} port ${RERANKER_PORT}..."
    
    # We use the local conda env (verlenv) for this, just like FAISS server
    CUDA_VISIBLE_DEVICES=${SERVICE_GPU} \
    python3 verl/verl/retrieval/engine/reranker_server.py --port ${RERANKER_PORT} \
    > reranker_server.log 2>&1 &
    
    RERANKER_SERVER_PID=$!
    echo "Reranker PID = ${RERANKER_SERVER_PID}"

    # Wait for Reranker to initialize
    echo "Waiting for Reranker Server to initialize..."
    # If your reranker server has a health endpoint, use curl. 
    # If not, use a simple sleep or grep log. Using sleep for safety based on your example.
    sleep 30 
    
    # Check if process is still running
    if ! kill -0 $RERANKER_SERVER_PID > /dev/null 2>&1; then
        echo "ERROR: Reranker server died immediately"
        tail -n 20 reranker_server.log
        exit 1
    fi
    
    echo "Reranker Server assumed ready."
fi

##############################
# 3. FAISS & EMBEDDING SERVER (GPU 2)
##############################
FAISS_SERVER_PID=""
VLLM_EMBEDDING_PID=""
if [ "${RETRIEVER_TYPE}" = "faiss" ]; then
    echo "========================================="
    echo "Setting up FAISS Retrieval Server"
    echo "========================================="

    VLLM_EMBEDDING_PORT=8001
    FAISS_SERVER_PORT=8002
    export PYTHONUNBUFFERED=1
    CONTAINER=/projects/2/managed_datasets/containers/vllm/vllm_25.09.sif

    # Start vLLM embedding server
    echo "Starting vLLM embedding server on GPU ${SERVICE_GPU}..."
    CUDA_VISIBLE_DEVICES=${SERVICE_GPU} \
    apptainer exec --nv -B $PWD $CONTAINER bash -lc "
      python3 -m vllm.entrypoints.openai.api_server \
        --model ${EMBEDDING_MODEL} \
        --dtype float16 \
        --max-model-len 512 \
        --port ${VLLM_EMBEDDING_PORT} \
        --host 0.0.0.0 \
        --no-enable-chunked-prefill \
        --task embedding 
    " > vllm_embedding.log 2>&1 &

    VLLM_EMBEDDING_PID=$!
    
    # Wait loop for Embedding...
    echo "Waiting for vLLM embedding..."
    for i in {1..60}; do
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VLLM_EMBEDDING_PORT}/health" || true)
        if [[ "$STATUS" == "200" ]]; then break; fi
        sleep 2
    done
    if [[ "$STATUS" != "200" ]]; then
        echo "ERROR: vLLM embedding failed"
        tail -n 20 vllm_embedding.log
        kill $RERANKER_SERVER_PID || true
        exit 1
    fi

    # Start FAISS server
    echo "Starting FAISS retrieval server on GPU ${SERVICE_GPU}..."
    CUDA_VISIBLE_DEVICES=${SERVICE_GPU} \
    FAISS_INDEX_PATH="${FAISS_INDEX}" \
    EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
    ID_MAPPING_PATH="${FAISS_ID_MAPPING}" \
    DEVICE="cuda" \
    INDEX_DEVICE="cuda" \
    EMBEDDING_MODE="vllm" \
    VLLM_SERVER_URL="http://localhost:${VLLM_EMBEDDING_PORT}" \
    MAX_SEQ_LEN=512 \
    PORT=${FAISS_SERVER_PORT} \
    HOST="0.0.0.0" \
    python3 -m verl.retrieval.engine.faiss_server > faiss_server.log 2>&1 &

    FAISS_SERVER_PID=$!

    # Wait loop for FAISS...
    echo "Waiting for FAISS..."
    for i in {1..60}; do
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${FAISS_SERVER_PORT}/health" || true)
        if [[ "$STATUS" == "200" ]]; then break; fi
        sleep 2
    done
    if [[ "$STATUS" != "200" ]]; then
        echo "ERROR: FAISS server failed"
        tail -n 20 faiss_server.log
        kill $RERANKER_SERVER_PID || true
        kill $VLLM_EMBEDDING_PID || true
        exit 1
    fi
    
    ID_MAPPING="${FAISS_ID_MAPPING}"
    echo "FAISS Retrieval Setup Complete."
else
    echo "Using BM25 Retriever."
    ID_MAPPING="${BM25_ID_MAPPING}"
fi

##############################
# GPU ALLOCATION FOR TRAINING
##############################
# Reserve GPUs 0-1 for Ray training workers
export CUDA_VISIBLE_DEVICES=0,1

echo "========================================="
echo "GPU Allocation:"
echo "  - Training (Ray): GPUs 0-1"
echo "  - Services (FAISS/Embed/Rerank): GPU 2"
echo "========================================="

##############################
# GPU USAGE MONITOR
##############################
(
    while true; do
        nvidia-smi > gpu_usage.log
        sleep 30
    done
) &
GPU_MONITOR_PID=$!

##############################
# EXPERIMENT CONFIGURATION
##############################
PROJECT_NAME="beir_retrieval_diversity"
EXPERIMENT_NAME="${BEIR_DATASET}_${RETRIEVER_TYPE}_${RETRIEVAL_REWARD_METHOD}@${K}_${DIVERSITY_METHOD}"
CHECKPOINT_DIR_FULL="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

##############################
# TRAINING
##############################

# Build retriever-specific config arguments
if [ "${RETRIEVER_TYPE}" = "faiss" ]; then
    RETRIEVER_ARGS=(
        "+reward_model.faiss_index_path=${FAISS_INDEX}"
        "+reward_model.id_mapping_path=${FAISS_ID_MAPPING}"
        "+reward_model.embedding_model=${EMBEDDING_MODEL}"
        "+reward_model.use_faiss_server=True"
        "+reward_model.faiss_server_url=http://localhost:8002"
        "+reward_model.nprobe=${NPROBE}"
        "+reward_model.encoding_batch_size=${ENCODING_BATCH_SIZE}"
        "+reward_model.search_batch_size=${SEARCH_BATCH_SIZE}"
    )
else
    RETRIEVER_ARGS=(
        "+reward_model.bm25_index_path=${BM25_INDEX}"
        "+reward_model.id_mapping_path=${BM25_ID_MAPPING}"
        "+reward_model.bm25_k1=${BM25_K1}"
        "+reward_model.bm25_b=${BM25_B}"
        "+reward_model.bm25_num_threads=${BM25_NUM_THREADS}"
    )
fi

# Build diversity-specific config arguments
REWARD_MANAGER="retrieval"
DIVERSITY_ARGS=()

if [ "${DIVERSITY_ENABLED}" = "true" ]; then
    DIVERSITY_ARGS+=( "+reward_model.diversity_reward_manager=diversity" )

    case "${DIVERSITY_METHOD}" in
        "ngram")
            DIVERSITY_ARGS+=(
                "+reward_model.custom_diversity_function.path=verl/verl/utils/reward_score/diversity_rewards.py"
                "+reward_model.custom_diversity_function.name=ngram"
                "+reward_model.reward_kwargs.n=${NGRAM_N}"
            )
            ;;
        "partition_vllm")
            DIVERSITY_ARGS+=(
                "+reward_model.custom_diversity_function.path=${PARTITION_REWARD_PATH}"
                "+reward_model.custom_diversity_function.name=partition"
            )
            ;;
        "unlikelihood")
            DIVERSITY_ARGS+=(
                "+reward_model.custom_diversity_function.path=verl/verl/utils/reward_score/diversity_rewards.py"
                "+reward_model.custom_diversity_function.name=unlikelihood"
            )
            ;;
        *)
            echo "ERROR: Unknown diversity method: ${DIVERSITY_METHOD}"
            exit 1
            ;;
    esac
fi

# Run PPO Training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="[${QUERIES_FILE},${TRAIN_QRELS}]" \
    data.val_files="[${QUERIES_FILE},${DEV_QRELS}]" \
    +data.custom_cls.path=verl.utils.dataset.beir_dataset \
    +data.custom_cls.name=BeirRLDataset \
    data.prompt_extender=${PROMPT_EXTENDER} \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=16 \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.use_flash_attn=True \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD_PARAMS} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD_OPTIMIZER} \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=${N_REWRITES} \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.val_kwargs.n=${N_REWRITES} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp \
    critic.strategy=fsdp \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    reward_model.enable=False \
    reward_model.reward_manager=${REWARD_MANAGER} \
    +reward_model.retriever_type=${RETRIEVER_TYPE} \
    +reward_model.beir_dataset_path=${BEIR_DIR} \
    +reward_model.qrels_file=${QRELS_FILE} \
    +reward_model.quality_method=${RETRIEVAL_REWARD_METHOD} \
    +reward_model.reranker_url="http://localhost:${RERANKER_PORT}/v1/score" \
    +reward_model.k=${K} \
    +reward_model.multiplicative=${MULTIPLICATIVE} \
    +reward_model.lambda_rm_rescale=${LAMBDA_RM} \
    +reward_model.lambda_rule_rescale=${LAMBDA_RULE} \
    "${RETRIEVER_ARGS[@]}" \
    "${DIVERSITY_ARGS[@]}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=0 \
    +trainer.log_rewrite_freq=${LOG_REWRITE_FREQ} \
    +trainer.log_num_queries=${LOG_NUM_QUERIES} \
    +trainer.log_num_rewrites=${LOG_NUM_REWRITES} \
    trainer.default_local_dir=${CHECKPOINT_DIR_FULL} \
    trainer.validation_data_dir=${CHECKPOINT_DIR_FULL}/rollouts \
    trainer.total_epochs=${EPOCHS} \
    $@

TRAIN_EXIT_CODE=$?

##############################################
# CLEANUP
##############################################
echo "========================================="
echo "Cleaning up..."
echo "========================================="

echo "Stopping GPU monitor..."
kill $GPU_MONITOR_PID || true

if [ -n "${VLLM_CLASSIFIER_PID}" ]; then
    echo "Stopping vLLM classifier..."
    kill ${VLLM_CLASSIFIER_PID} || true
fi

if [ -n "${RERANKER_SERVER_PID}" ]; then
    echo "Stopping Reranker server..."
    kill ${RERANKER_SERVER_PID} || true
fi

if [ "${RETRIEVER_TYPE}" = "faiss" ]; then
    if [ -n "${FAISS_SERVER_PID}" ]; then
        echo "Stopping FAISS server..."
        kill ${FAISS_SERVER_PID} || true
    fi

    if [ -n "${VLLM_EMBEDDING_PID}" ]; then
        echo "Stopping vLLM embedding server..."
        kill ${VLLM_EMBEDDING_PID} || true
    fi
fi

echo "========================================="
echo "Training completed with exit code: ${TRAIN_EXIT_CODE}"
echo "========================================="

exit ${TRAIN_EXIT_CODE}