#!/bin/bash
# Single-GPU Retrieval-based DARLING Training Script
# This script runs query rewriting with FAISS retrieval rewards on a single GPU
# It completely avoids loading the Athene reward model

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
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"  # Smaller embedding model to save memory

#==============================================================================
# Training Hyperparameters (optimized for single GPU)
#==============================================================================
BATCH_SIZE=32              # Reduced for single GPU
N_REWRITES=4               # Reduced from 8 to save memory
MAX_PROMPT_LEN=128
MAX_RESPONSE_LEN=128
LEARNING_RATE=1e-6
EPOCHS=3

#==============================================================================
# Reward Configuration
#==============================================================================
QUALITY_METHOD="ndcg"      # Options: ndcg, recall, hit
K=10
DIVERSITY_ENABLED=true
DIVERSITY_METHOD="local"   # Use local DeBERTa (no VLLM servers needed)
DEBERTA_MODEL="microsoft/deberta-v3-large"
THRESHOLD=0.5

#==============================================================================
# Memory Optimization Settings
#==============================================================================
GPU_MEMORY_UTIL=0.4        # Conservative for single GPU
OFFLOAD_PARAMS=true        # Offload to save memory
OFFLOAD_OPTIMIZER=false    # Keep optimizer on GPU for speed

#==============================================================================
# Logging Configuration
#==============================================================================
PROJECT_NAME="retrieval_darling_single_gpu"
EXPERIMENT_NAME="qwen05b_ndcg_diversity_b${BATCH_SIZE}_n${N_REWRITES}"
CHECKPOINT_DIR="./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# Optional: Set W&B API key
# export WANDB_API_KEY="your_wandb_api_key"

#==============================================================================
# Launch Training
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
    `# Critic Configuration (GRPO doesn't need critic, but keeping for compatibility)` \
    critic.strategy=fsdp \
    \
    `# Algorithm Configuration` \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    \
    `# ========================================================================` \
    `# REWARD MODEL CONFIGURATION - CRITICAL FOR AVOIDING ATHENE` \
    `# ========================================================================` \
    reward_model.enable=False \
    reward_model.reward_manager=retrieval \
    \
    `# FAISS Infrastructure (required)` \
    +reward_model.faiss_index_path=${FAISS_INDEX} \
    +reward_model.id_mapping_path=${ID_MAPPING} \
    +reward_model.labels_path=${LABELS} \
    \
    `# Quality Reward Configuration` \
    +reward_model.quality_method=${QUALITY_METHOD} \
    +reward_model.k=${K} \
    +reward_model.embedding_model=${EMBEDDING_MODEL} \
    \
    `# Diversity Reward Configuration` \
    +reward_model.diversity_enabled=${DIVERSITY_ENABLED} \
    +reward_model.diversity_method=${DIVERSITY_METHOD} \
    +reward_model.deberta_model=${DEBERTA_MODEL} \
    +reward_model.threshold=${THRESHOLD} \
    +reward_model.device=cuda \
    \
    `# Custom Reward Function (points to retrieval_rewards module)` \
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