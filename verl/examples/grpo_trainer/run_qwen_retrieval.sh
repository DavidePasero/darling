#!/bin/bash
# Training script for query rewriting with GRPO + Retrieval Rewards (Quality × Diversity)
# This implements the DARLING methodology for information retrieval

set -x

# Export environment variables
export WANDB_API_KEY=${WANDB_API_KEY:-""}  # Set your W&B API key
export VLLM_SERVER_HOSTNAME=${VLLM_SERVER_HOSTNAME:-"localhost"}  # For VLLM diversity (if enabled)

# Configuration
CONFIG_PATH="examples/grpo_trainer/config/retrieval_grpo.yaml"

# Data paths - UPDATE THESE TO YOUR ACTUAL PATHS
FAISS_INDEX="~/data/msmarco/msmarco.faiss"
ID_MAPPING="~/data/msmarco/id_mapping.pkl"
LABELS="~/data/msmarco/labels.pkl"
TRAIN_DATA="~/data/msmarco/train.parquet"
VAL_DATA="~/data/msmarco/dev.parquet"

# Model configuration
MODEL_PATH="Qwen/Qwen2.5-0.5B"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"

# Training hyperparameters
BATCH_SIZE=512
N_REWRITES=8  # Number of rewrites per query
TEMPERATURE=0.7
EPOCHS=5

# Reward configuration
QUALITY_METHOD="ndcg"  # Options: ndcg, recall, hit
K=10  # Top-k retrieval
DIVERSITY_ENABLED=true
DIVERSITY_METHOD="local"  # Options: local (DeBERTa), vllm (server-based)
DEBERTA_MODEL="microsoft/deberta-v3-large"
THRESHOLD=0.5

# Run training
python3 -m verl.trainer.main_ppo \
  --config-path $CONFIG_PATH \
  \
  data.train_files=$TRAIN_DATA \
  data.val_files=$VAL_DATA \
  data.train_batch_size=$BATCH_SIZE \
  data.val_batch_size=256 \
  data.max_prompt_length=128 \
  data.max_response_length=128 \
  \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.rollout.n=$N_REWRITES \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  \
  algorithm=grpo \
  algorithm.adv_estimator=grpo \
  \
  reward_model.enable=false \
  reward_model.reward_manager=retrieval \
  reward_model.faiss_index_path=$FAISS_INDEX \
  reward_model.id_mapping_path=$ID_MAPPING \
  reward_model.labels_path=$LABELS \
  reward_model.quality_method=$QUALITY_METHOD \
  reward_model.k=$K \
  reward_model.diversity_enabled=$DIVERSITY_ENABLED \
  reward_model.diversity_method=$DIVERSITY_METHOD \
  reward_model.deberta_model=$DEBERTA_MODEL \
  reward_model.threshold=$THRESHOLD \
  reward_model.embedding_model=$EMBEDDING_MODEL \
  reward_model.custom_diversity_function.path=verl.utils.reward_score.retrieval_rewards \
  reward_model.custom_diversity_function.name=retrieval_reward \
  \
  trainer.total_epochs=$EPOCHS \
  trainer.project_name=retrieval-darling \
  trainer.experiment_name=qwen05b-msmarco-grpo-diversity \
  trainer.logger=['wandb'] \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=100