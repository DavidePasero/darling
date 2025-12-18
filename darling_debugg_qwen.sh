#!/bin/bash
#SBATCH --job-name=TRAIN
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/darling_fix_%j.out
#SBATCH --error=logs/darling_fix_%j.err

set -euo pipefail

CHECKPOINT_DIR=/home/scur1900/scratch_shared/checkpoints/qwen_baseline

##############################
# ENVIRONMENT
##############################
module purge
module load 2023
module load Anaconda3/2023.07-2

eval "$(conda shell.bash hook)"
conda activate verlenv
export WANDB_API_KEY=9dbe3f48ba14149757b7b5b73b02ef3e9cab94b5

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

QWEN_PATH="/home/scur1900/models/Qwen2.5-3B-Instruct"
ATHENE_PATH="/home/scur1900/models/athene-rm-8b"
PARTITION_REWARD="/home/scur1900/darling_davide/verl/verl/utils/reward_score/partition_reward_vllm_serve_modernbert.py"

##############################################
# START LOCAL VLLM CLASSIFIER SERVER ON GPU 0
##############################################
export VLLM_SERVER_HOSTNAME="localhost"
export VLLM_PORT=8000
export PYTHONUNBUFFERED=1

MODEL="/home/scur1900/models/dogtooth"
CONTAINER=/projects/2/managed_datasets/containers/vllm/vllm_25.09.sif

echo "Starting VLLM classifier on GPU0..."

CUDA_VISIBLE_DEVICES=0 \
apptainer exec --nv -B $PWD $CONTAINER bash -lc "
  python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --dtype float16 \
    --max-model-len 4096 \
    --port $VLLM_PORT \
    --host 0.0.0.0
" > vllm_classifier.log 2>&1 &

VLLM_PID=$!
echo "VLLM PID = $VLLM_PID"

echo "Waiting for LOCAL VLLM server at http://localhost:${VLLM_PORT}/health"
for i in {1..60}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${VLLM_PORT}/health" || true)
    if [[ "$STATUS" == "200" ]]; then
        echo "LOCAL VLLM READY!"
        break
    fi
    sleep 2
done

if [[ "$STATUS" != "200" ]]; then
    echo "ERROR: LOCAL VLLM failed to start"
    tail -n 50 vllm_classifier.log
    kill $VLLM_PID || true
    exit 1
fi

##############################################
# GPU USAGE MONITOR (BACKGROUND)
##############################################
echo "Training will use GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

(
    while true; do
        echo "------ GPU USAGE @ $(date) ------"
        nvidia-smi
        sleep 20
    done
) &

GPU_MONITOR_PID=$!

##############################
# HYPERPARAMETERS
##############################
B=16
N=4
L=750

##############################
# TRAINING
##############################
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct', 
    local_dir='/home/scur1900/models/Qwen2.5-3B-Instruct',
    resume_download=True
    # Note: We removed the ignore_patterns list so it fetches the weights
)
"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DS \
    data.val_files=${VAL_DS} \
    data.prompt_key="prompt" \
    data.train_batch_size=$B \
    data.val_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=$L \
    data.filter_overlong_prompts=True \
    data.truncation=error \
\
    actor_rollout_ref.model.path=$QWEN_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +trainer.extra_generation_kwargs.pad_token_id=128009 \
\
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
\
    reward_model.enable=True \
    reward_model.reward_manager=diversity \
    reward_model.model.path=$ATHENE_PATH \
    reward_model.model.input_tokenizer=$ATHENE_PATH \
    reward_model.micro_batch_size_per_gpu=16 \
    +reward_model.custom_diversity_function.path=$PARTITION_REWARD \
    +reward_model.custom_diversity_function.name=partition \
    +reward_model.multiplicative=True \
    trainer.project_name="darling_llama32_3b_multiplicative" \
    trainer.experiment_name="qwen_baseline" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.logger="[wandb]" \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.validation_data_dir="${CHECKPOINT_DIR}/rollouts" \
    trainer.critic_warmup=0 \
    trainer.save_freq=250 \
    trainer.test_freq=0




##############################################
# CLEANUP
##############################################
echo "Stopping GPU monitor..."
kill $GPU_MONITOR_PID || true

echo "Killing classifier server..."
kill $VLLM_PID || true