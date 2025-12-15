#!/bin/bash
#SBATCH --job-name=BEIR_RETR
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/beir_retrieval_%j.out
#SBATCH --error=logs/beir_retrieval_%j.err

##############################
# CONFIGURATION
##############################
CHECKPOINT_DIR=/home/scur1900/scratch-shared/checkpoints

# BEIR Dataset Configuration (from train_retrieval_beir.sh)
BEIR_DATASET="fiqa"
BEIR_DIR="/home/scur1900/scratch_shared/${BEIR_DATASET}"
QUERIES_FILE="${BEIR_DIR}/queries.jsonl"
TRAIN_QRELS="${BEIR_DIR}/qrels/train.tsv"
DEV_QRELS="${BEIR_DIR}/qrels/dev.tsv"
QRELS_FILE="qrels/train.tsv"

# Retriever Configuration
RETRIEVER_TYPE="bm25"  # Options: "faiss" or "bm25"
FAISS_INDEX="${BEIR_DIR}/faiss/index"
FAISS_ID_MAPPING="${BEIR_DIR}/faiss/docid"
EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
BM25_INDEX="${BEIR_DIR}/bm25_index/index"
BM25_ID_MAPPING="${BEIR_DIR}/bm25_index/id_mapping.pkl"
BM25_K1=0.9
BM25_B=0.4
# BM25_NUM_THREADafter 50 steps the logs show this: 2025-12-13 02:15:30
# test_gen_batch meta info: {'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': True, 'validate': True}
# 2025-12-13 02:15:31
# validation generation end.  and it is bascally going forever so we had to stop the run. WHy does this happen?
BM25_NUM_THREADS=128  # Number of parallel search threads for BM25 (can be higher than CPU cores for I/O-bound tasks)

# Model Configuration
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE=256
N_REWRITES=10
MAX_PROMPT_LEN=256
MAX_RESPONSE_LEN=512
LEARNING_RATE=1e-6
EPOCHS=500

# Reward Configuration
QUALITY_METHOD="ndcg"
K=10

# FAISS Performance Configuration
NPROBE=8                     # Number of clusters to probe (lower=faster, 8-16 recommended for GPU)
ENCODING_BATCH_SIZE=512      # Batch size for embedding encoding (larger=better GPU utilization)

# Logging Configuration
LOG_REWRITE_FREQ=50          # Log rewrites every N steps
LOG_NUM_QUERIES=10            # Number of queries to log
LOG_NUM_REWRITES=10          # Number of rewrites per query to log

# Resource Configuration
GPU_MEMORY_UTIL=0.8
OFFLOAD_PARAMS=true
OFFLOAD_OPTIMIZER=false

##############################
# ENVIRONMENT SETUP
##############################
module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv

export HYDRA_FULL_ERROR=1
export DEBUG_LOG=0

# Enable strict error checking AFTER conda activation
# (conda activation scripts may have unbound variables like JAVA_HOME)
set -euo pipefail
export WANDB_API_KEY=9dbe3f48ba14149757b7b5b73b02ef3e9cab94b5

# Set working directory and PYTHONPATH to use correct verl version
WORK_DIR="/gpfs/home4/scur1900/darling_lukas"
cd ${WORK_DIR}
export PYTHONPATH="${WORK_DIR}/verl:${PYTHONPATH:-}"

echo "========== ENV INFO =========="
echo "Node: $(hostname)"
echo "Working Directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "=============================="

##############################
# VLLM EMBEDDING SERVER SETUP
# (Only needed for FAISS retriever)
##############################
if [ "${RETRIEVER_TYPE}" = "faiss" ]; then
    echo "========================================="
    echo "Setting up vLLM Embedding Server (FAISS)"
    echo "========================================="

    export VLLM_SERVER_HOSTNAME="localhost"
    export VLLM_EMBEDDING_PORT=8001
    export PYTHONUNBUFFERED=1

    CONTAINER=/projects/2/managed_datasets/containers/vllm/vllm_25.09.sif

    echo "Starting vLLM embedding server on GPU 0..."
    echo "Embedding Model: ${EMBEDDING_MODEL}"

    CUDA_VISIBLE_DEVICES=0 \
    apptainer exec --nv -B $PWD $CONTAINER bash -lc "
      python3 -m vllm.entrypoints.openai.api_server \
        --model ${EMBEDDING_MODEL} \
        --dtype float16 \
        --max-model-len 512 \
        --port ${VLLM_EMBEDDING_PORT} \
        --host 0.0.0.0 \
        --task embedding
    " > vllm_embedding.log 2>&1 &

    VLLM_EMBEDDING_PID=$!
    echo "vLLM Embedding PID = ${VLLM_EMBEDDING_PID}"

    echo "Waiting for vLLM embedding server at http://localhost:${VLLM_EMBEDDING_PORT}/health"
    for i in {1..60}; do
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VLLM_EMBEDDING_PORT}/health" || true)
        if [[ "$STATUS" == "200" ]]; then
            echo "vLLM Embedding Server READY!"
            break
        fi
        sleep 2
    done

    if [[ "$STATUS" != "200" ]]; then
        echo "ERROR: vLLM embedding server failed to start"
        tail -n 50 vllm_embedding.log
        kill ${VLLM_EMBEDDING_PID} || true
        exit 1
    fi

    # Set ID mapping for FAISS
    ID_MAPPING="${FAISS_ID_MAPPING}"

    echo "vLLM Embedding Server is ready!"
    echo "========================================="
else
    echo "========================================="
    echo "Using BM25 Retriever (no embedding model needed)"
    echo "========================================="
    ID_MAPPING="${BM25_ID_MAPPING}"
    VLLM_EMBEDDING_PID=""
fi

##############################
# GPU USAGE MONITOR
##############################
echo "Training will use GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

(
    while true; do
        echo "------ GPU USAGE @ $(date) ------"
        nvidia-smi
        sleep 30
    done
) &

GPU_MONITOR_PID=$!

##############################
# EXPERIMENT CONFIGURATION
##############################
PROJECT_NAME="beir_retrieval"
EXPERIMENT_NAME="${BEIR_DATASET}_${RETRIEVER_TYPE}_${QUALITY_METHOD}@${K}"
CHECKPOINT_DIR_FULL="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

echo "========================================="
echo "EXPERIMENT: ${EXPERIMENT_NAME}"
echo "Project: ${PROJECT_NAME}"
echo "Dataset: ${BEIR_DATASET}"
echo "Retriever: ${RETRIEVER_TYPE}"
echo "Quality Method: ${QUALITY_METHOD}@${K}"
echo "Model: ${MODEL_PATH}"
echo "Checkpoint Dir: ${CHECKPOINT_DIR_FULL}"
echo "========================================="

##############################
# TRAINING
##############################

# Build retriever-specific config arguments
if [ "${RETRIEVER_TYPE}" = "faiss" ]; then
    RETRIEVER_ARGS=(
        "+reward_model.faiss_index_path=${FAISS_INDEX}"
        "+reward_model.id_mapping_path=${FAISS_ID_MAPPING}"
        "+reward_model.embedding_model=${EMBEDDING_MODEL}"
        "+reward_model.embedding_mode=vllm"
        "+reward_model.vllm_server_url=http://localhost:8001"
        "+reward_model.faiss_device=cuda"
        "+reward_model.nprobe=${NPROBE}"
        "+reward_model.encoding_batch_size=${ENCODING_BATCH_SIZE}"
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

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="[${QUERIES_FILE},${TRAIN_QRELS}]" \
    data.val_files="[${QUERIES_FILE},${DEV_QRELS}]" \
    +data.custom_cls.path=verl.utils.dataset.beir_dataset \
    +data.custom_cls.name=BeirRLDataset \
    data.prompt_extender=bm25 \
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
    reward_model.reward_manager=retrieval \
    +reward_model.retriever_type=${RETRIEVER_TYPE} \
    +reward_model.beir_dataset_path=${BEIR_DIR} \
    +reward_model.qrels_file=${QRELS_FILE} \
    +reward_model.quality_method=${QUALITY_METHOD} \
    +reward_model.k=${K} \
    "${RETRIEVER_ARGS[@]}" \
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

if [ "${RETRIEVER_TYPE}" = "faiss" ] && [ -n "${VLLM_EMBEDDING_PID}" ]; then
    echo "Stopping vLLM embedding server..."
    kill ${VLLM_EMBEDDING_PID} || true
    echo "vLLM embedding server stopped"
fi

echo "========================================="
echo "Training completed with exit code: ${TRAIN_EXIT_CODE}"
echo "========================================="

exit ${TRAIN_EXIT_CODE}
