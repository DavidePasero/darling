#!/bin/bash
# Simple Retrieval Training Script - NO DIVERSITY
# Single GPU, Quality Rewards Only (NDCG/Recall)
# Use this to test the retrieval infrastructure first

set -x

#==============================================================================
# REQUIRED: Update these paths to your actual data
#==============================================================================
FAISS_INDEX="/path/to/your/msmarco.faiss"
ID_MAPPING="/path/to/your/id_mapping.pkl"
LABELS="/path/to/your/labels.pkl"
TRAIN_DATA="/path/to/your/train.parquet"
VAL_DATA="/path/to/your/val.parquet"

#==============================================================================
# Model Configuration
#==============================================================================
MODEL_PATH="Qwen/Qwen2.5-0.5B"  # Small model for single GPU
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"  # For FAISS retrieval

#==============================================================================
# Training Hyperparameters (optimized for single GPU)
#==============================================================================
BATCH_SIZE=32
N_REWRITES=4               # Generate 4 rewrites per query
MAX_PROMPT_LEN=128
MAX_RESPONSE_LEN=128
LEARNING_RATE=1e-6
EPOCHS=3

#==============================================================================
# Reward Configuration - QUALITY ONLY (NO DIVERSITY)
#==============================================================================
QUALITY_METHOD="ndcg"      # Options: ndcg, recall, hit
K=10                       # Top-k retrieval

#==============================================================================
# Memory Optimization Settings
#==============================================================================
GPU_MEMORY_UTIL=0.4
OFFLOAD_PARAMS=true
OFFLOAD_OPTIMIZER=false

#==============================================================================
# Logging Configuration
#==============================================================================
PROJECT_NAME="retrieval_simple_test"
EXPERIMENT_NAME="quality_only_ndcg@${K}"
CHECKPOINT_DIR="./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

#==============================================================================
# Launch Training - RETRIEVAL QUALITY ONLY
#==============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    `# Data Configuration` \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.prompt_key="prompt" \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=16 \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    `# Model Configuration` \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    `# Actor Training Configuration` \
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
    \
    `# Rollout Configuration (vLLM for generation)` \
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
    \
    `# Reference Model Configuration` \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp \
    \
    `# Critic Configuration` \
    critic.strategy=fsdp \
    \
    `# Algorithm Configuration` \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    \
    `# ========================================================================` \
    `# REWARD CONFIGURATION - QUALITY ONLY, NO DIVERSITY` \
    `# This completely bypasses Athene and any reward models` \
    `# ========================================================================` \
    reward_model.enable=False \
    reward_model.reward_manager=retrieval \
    \
    `# FAISS Infrastructure` \
    +reward_model.faiss_index_path=${FAISS_INDEX} \
    +reward_model.id_mapping_path=${ID_MAPPING} \
    +reward_model.labels_path=${LABELS} \
    \
    `# Quality Reward Settings` \
    +reward_model.quality_method=${QUALITY_METHOD} \
    +reward_model.k=${K} \
    +reward_model.embedding_model=${EMBEDDING_MODEL} \
    \
    `# DISABLE DIVERSITY - Just quality rewards` \
    +reward_model.diversity_enabled=False \
    \
    `# Custom Reward Function` \
    +reward_model.custom_diversity_function.path=verl.utils.reward_score.retrieval_rewards \
    +reward_model.custom_diversity_function.name=retrieval_reward \
    \
    `# Trainer Configuration` \
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

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"
echo "=========================================="