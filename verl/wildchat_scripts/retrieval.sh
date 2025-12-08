#!/bin/bash
# Retrieval-based DARLING training script for query rewriting
# Combines FAISS retrieval quality (NDCG/Recall) with partition-based diversity

# Hyperparameters
B=64        # Batch size
N=8         # Number of rewrites per query
L=128       # Max response length (queries are short)

# FAISS Infrastructure Paths - UPDATE THESE
FAISS_INDEX="/path/to/msmarco.faiss"
ID_MAPPING="/path/to/id_mapping.pkl"
LABELS="/path/to/labels.pkl"

# Data Paths
TRAIN_DATA="/path/to/msmarco_train.parquet"
VAL_DATA="/path/to/msmarco_dev.parquet"

# Model Configuration
MODEL_PATH="Qwen/Qwen2.5-0.5B"  # Or your preferred model
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"

# Reward Configuration
QUALITY_METHOD="ndcg"  # Options: ndcg, recall, hit
K=10
DIVERSITY_METHOD="local"  # Options: local (DeBERTa), vllm (server)
DEBERTA_MODEL="microsoft/deberta-v3-large"
THRESHOLD=0.5

# For VLLM diversity (if DIVERSITY_METHOD=vllm)
export VLLM_SERVER_HOSTNAME=${VLLM_SERVER_HOSTNAME:-"localhost"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.prompt_key="prompt" \
    data.train_batch_size=${B} \
    data.max_prompt_length=128 \
    data.max_response_length=${L} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.val_kwargs.n=${N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    reward_model.enable=False \
    reward_model.reward_manager=retrieval \
    +reward_model.faiss_index_path=${FAISS_INDEX} \
    +reward_model.id_mapping_path=${ID_MAPPING} \
    +reward_model.labels_path=${LABELS} \
    +reward_model.quality_method=${QUALITY_METHOD} \
    +reward_model.k=${K} \
    +reward_model.diversity_enabled=True \
    +reward_model.diversity_method=${DIVERSITY_METHOD} \
    +reward_model.deberta_model=${DEBERTA_MODEL} \
    +reward_model.threshold=${THRESHOLD} \
    +reward_model.embedding_model=${EMBEDDING_MODEL} \
    +reward_model.custom_diversity_function.path=verl.utils.reward_score.retrieval_rewards \
    +reward_model.custom_diversity_function.name=retrieval_reward \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='retrieval_darling' \
    trainer.experiment_name=retrieval_quality_diversity_b${B}_n${N}_l${L} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_local_dir=checkpoints/retrieval_darling/retrieval_quality_diversity_b${B}_n${N}_l${L} \
    trainer.validation_data_dir=checkpoints/retrieval_darling/retrieval_quality_diversity_b${B}_n${N}_l${L}/rollouts \
    trainer.total_epochs=5 $@