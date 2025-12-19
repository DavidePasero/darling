#!/usr/bin/env bash
# Train on a BEIR dataset with retrieval quality + diversity rewards.
# This mirrors the original training script but adds the diversity reward manager.

# ----------------------------------------------------------------------
# 1.  USER‑CONFIGURABLE PARAMETERS
# ----------------------------------------------------------------------
# FAISS index and ID‑mapping (generated beforehand with the BEIR data)
FAISS_INDEX="/path/to/faiss_index.faiss"
ID_MAPPING="/path/to/id_mapping.pkl"

# BEIR dataset location (directory that contains train/val parquet files)
BEIR_ROOT="/path/to/beir_dataset"

# Model and embedding configuration
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"          # LLM for rollout
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"    # Embedding model for retrieval

# Training hyper‑parameters
BATCH_SIZE=64          # PPO batch size
N_ROLLS=8              # Number of rollouts per query
MAX_PROMPT_LEN=128
MAX_RESPONSE_LEN=128
LR=1e-6                # Actor optimizer learning rate
TOTAL_EPOCHS=5

# Reward configuration
QUALITY_METHOD="ndcg"   # Retrieval quality metric (ndcg / recall / hit)
K=10                    # Top‑k for retrieval evaluation
DIV_METHOD="local"      # Diversity method (local = partition‑based)
MULTIPLICATIVE=true     # Combine quality & diversity multiplicatively
LAMBDA_RM=1.0           # Scale for quality reward
LAMBDA_RULE=1.0         # Scale for diversity reward

# ----------------------------------------------------------------------
# 2.  LAUNCH TRAINING
# ----------------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${BEIR_ROOT}/train.parquet" \
    data.val_files="${BEIR_ROOT}/dev.parquet" \
    data.prompt_key="prompt" \
    data.prompt_extender="dense_vector" \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${N_ROLLS} \
    actor_rollout_ref.rollout.val_kwargs.n=${N_ROLLS} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    reward_model.enable=False \
    reward_model.reward_manager=retrieval \
    +reward_model.diversity_reward_manager=diversity \
    +reward_model.retriever_type=faiss \
    +reward_model.faiss_index_path=${FAISS_INDEX} \
    +reward_model.id_mapping_path=${ID_MAPPING} \
    +reward_model.beir_dataset_path=${BEIR_ROOT} \
    +reward_model.quality_method=${QUALITY_METHOD} \
    +reward_model.k=${K} \
    +reward_model.embedding_model=${EMBEDDING_MODEL} \
    +reward_model.embedding_mode=local \
    +reward_model.device=cuda \
    +reward_model.faiss_device=cpu \
    +reward_model.diversity_method=${DIV_METHOD} \
    +reward_model.multiplicative=${MULTIPLICATIVE} \
    +reward_model.lambda_rm_rescale=${LAMBDA_RM} \
    +reward_model.lambda_rule_rescale=${LAMBDA_RULE} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='retrieval_beir_diversity' \
    trainer.experiment_name='beir_${QUALITY_METHOD}_k${K}_div${DIV_METHOD}' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_local_dir="checkpoints/beir_diversity/${QUALITY_METHOD}_k${K}" \
    trainer.validation_data_dir="checkpoints/beir_diversity/${QUALITY_METHOD}_k${K}/rollouts" \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@"