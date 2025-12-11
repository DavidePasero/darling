#!/bin/bash
# BEIR Retrieval Training - Quality Only (No Diversity)
# Uses BEIR format datasets (msmarco, nfcorpus, scifact, etc.)

set -x

# Enable Ray debug mode for breakpoint debugging (set to 0 for production)
export RAY_DEBUG_MODE=0

BEIR_DATASET="figa"
BEIR_DIR="datasets/${BEIR_DATASET}"
QUERIES_FILE="${BEIR_DIR}/queries.jsonl"
TRAIN_QRELS="${BEIR_DIR}/qrels/train.tsv"
DEV_QRELS="${BEIR_DIR}/qrels/dev.tsv"
QRELS_FILE="qrels/train.tsv"

RETRIEVER_TYPE="faiss"
FAISS_INDEX="${BEIR_DIR}/faiss_index.faiss"
FAISS_ID_MAPPING="${BEIR_DIR}/id_mapping.pkl"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"
BM25_INDEX="${BEIR_DIR}/bm25_index/index"
BM25_ID_MAPPING="${BEIR_DIR}/bm25_index/id_mapping.pkl"
BM25_K1=0.9
BM25_B=0.4

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE=8
N_REWRITES=2
MAX_PROMPT_LEN=128
MAX_RESPONSE_LEN=64
LEARNING_RATE=1e-6
EPOCHS=3

QUALITY_METHOD="ndcg"
K=10

GPU_MEMORY_UTIL=0.7
OFFLOAD_PARAMS=true
OFFLOAD_OPTIMIZER=false

if [ "${RETRIEVER_TYPE}" = "faiss" ]; then
    ID_MAPPING="${FAISS_ID_MAPPING}"
elif [ "${RETRIEVER_TYPE}" = "bm25" ]; then
    ID_MAPPING="${BM25_ID_MAPPING}"
else
    echo "Error: RETRIEVER_TYPE must be 'faiss' or 'bm25'"
    exit 1
fi

PROJECT_NAME="beir_retrieval"
EXPERIMENT_NAME="${BEIR_DATASET}_${RETRIEVER_TYPE}_${QUALITY_METHOD}@${K}"
CHECKPOINT_DIR="./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="[${QUERIES_FILE},${TRAIN_QRELS}]" \
    data.val_files="[${QUERIES_FILE},${DEV_QRELS}]" \
    +data.custom_cls.path=verl.utils.dataset.beir_dataset \
    +data.custom_cls.name=BeirRLDataset \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=16 \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.use_flash_attn=False \
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
    +reward_model.faiss_index_path=${FAISS_INDEX} \
    +reward_model.bm25_index_path=${BM25_INDEX} \
    +reward_model.bm25_k1=${BM25_K1} \
    +reward_model.bm25_b=${BM25_B} \
    +reward_model.id_mapping_path=${ID_MAPPING} \
    +reward_model.beir_dataset_path=${BEIR_DIR} \
    +reward_model.qrels_file=${QRELS_FILE} \
    +reward_model.quality_method=${QUALITY_METHOD} \
    +reward_model.k=${K} \
    +reward_model.embedding_model=${EMBEDDING_MODEL} \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.default_local_dir=${CHECKPOINT_DIR} \
    trainer.validation_data_dir=${CHECKPOINT_DIR}/rollouts \
    trainer.total_epochs=${EPOCHS} \
    $@