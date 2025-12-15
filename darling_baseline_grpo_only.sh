#!/bin/bash
#SBATCH --job-name=GRPO_BASELINE
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs_grpo/darling_grpo_baseline_%j.out
#SBATCH --error=logs_grpo/darling_grpo_baseline_%j.err

set -euo pipefail

CHECKPOINT_DIR=/scratch-shared/tmp.fdofatle2c/sharded_checkpoints/grpo

##############################
# ENVIRONMENT
##############################
module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv
export WANDB_API_KEY=c512aca8678cbec8fbbf6bf08be3f6fe644ac7dc

echo "========== ENV INFO =========="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "=============================="

##############################
# PATHS
##############################
TRAIN_DS="/home/scur1900/datasets_1/wildchat10k.parquet"
VAL_DS="/home/scur1900/datasets_1/wildchat_valid.parquet"

LLAMA_PATH="/home/scur1900/models/llama-3b"
ATHENE_PATH="/home/scur1900/models/athene-rm-8b"

##############################################
# START LOCAL VLLM CLASSIFIER SERVER ON GPU 0
##############################################

##############################
# HYPERPARAMETERS
##############################
B=16
N=4
L=750

##############################
# TRAINING
##############################
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DS \
    data.val_files=$VAL_DS \
    data.prompt_key="prompt" \
    data.train_batch_size=$B \
    data.max_prompt_length=512 \
    data.max_response_length=$L \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$LLAMA_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
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
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$HOME/darling_lukas/verl/verl/utils/reward_score/diversity_rewards.py \
    custom_reward_function.name=ngram \
    reward_model.reward_manager=diversity \
    reward_model.enable=True \
    reward_model.model.input_tokenizer=$ATHENE_PATH \
    reward_model.model.path=$ATHENE_PATH \
    reward_model.micro_batch_size_per_gpu=16 \
    trainer.critic_warmup=0 \
    trainer.logger="['console', 'wandb']" \
    trainer.project_name="darling_llama32_3b_grpo" \
    trainer.experiment_name="darling_baseline" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=0 \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.validation_data_dir="${CHECKPOINT_DIR}/rollouts" \
    trainer.total_epochs=2 \
    +reward_model.lambda_rule_rescale=0.0 \
    +trainer.extra_generation_kwargs.pad_token_id=128009


##############################################
# CLEANUP
##############################################
echo "Stopping GPU monitor..."
kill $GPU_MONITOR_PID || true

echo "Killing classifier server..."
kill $VLLM_PID || true