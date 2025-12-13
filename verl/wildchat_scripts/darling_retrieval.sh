#!/bin/bash
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export DEBUG_LOG=1
export RERANKER_PORT=8000

# Configuration
B=4
N=2
L=128
RETRIEVAL_REWARD_METHOD="reranker+ndcg"

if [[ "$RETRIEVAL_REWARD_METHOD" == *"reranker"* ]]; then
    # ---------------------------------------------------------
    # 1. Start Reranker Server (Background Process)
    # ---------------------------------------------------------
    echo "Starting Reranker Server on port $RERANKER_PORT..."
    python verl/verl/retrieval/engine/reranker_server.py --port $RERANKER_PORT &
    SERVER_PID=$!

    # Wait for server to initialize (simple sleep, or you can poll the port)
    echo "Waiting for Reranker Server to initialize..."
    sleep 10 

    # ---------------------------------------------------------
    # 2. Run Training
    # ---------------------------------------------------------
    echo "Starting Training..."

    # Use 'trap' to ensure the server is killed even if training fails (Ctrl+C or Error)
    trap "echo 'Stopping Reranker Server...'; kill $SERVER_PID" EXIT
fi

python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="['datasets/fiqa/queries.jsonl', 'datasets/fiqa/qrels/train.tsv']" \
        data.val_files="['datasets/fiqa/queries.jsonl', 'datasets/fiqa/qrels/test.tsv']" \
        +data.custom_cls.path=verl.utils.dataset.beir_dataset \
        +data.custom_cls.name=BeirRLDataset \
        data.train_batch_size=$B \
        data.val_batch_size=$B \
        data.max_prompt_length=$L \
        data.max_response_length=$L \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.n=$N \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        +actor_rollout_ref.model.use_flash_attn=False \
        actor_rollout_ref.rollout.val_kwargs.n=4 \
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
        +reward_model.retriever_type=bm25 \
        +reward_model.faiss_index_path=datasets/fiqa/faiss_index.faiss \
        +reward_model.bm25_index_path=datasets/fiqa/bm25_index/index \
        +reward_model.bm25_k1=0.9 \
        +reward_model.bm25_b=0.4 \
        +reward_model.id_mapping_path=datasets/fiqa/bm25_index/id_mapping.pkl \
        +reward_model.beir_dataset_path=datasets/fiqa \
        +reward_model.qrels_file=qrels/train.tsv \
        +reward_model.quality_method=$RETRIEVAL_REWARD_METHOD \
        +reward_model.reranker_url=http://localhost:$RERANKER_PORT/v1/score \
        +reward_model.k=10 \
        +reward_model.embedding_model=Qwen/Qwen3-Embedding-0.6B \
        trainer.critic_warmup=0 \
        trainer.logger=console \
        trainer.project_name=beir_retrieval \
        trainer.experiment_name=figa_bm25_ndcg@10 \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.default_local_dir=./checkpoints/beir_retrieval/fiqa_bm25_ndcg@10 \
        trainer.validation_data_dir=./checkpoints/beir_retrieval/fiqa_bm25_ndcg@10/rollouts \
        trainer.total_epochs=3

# The 'trap' command above will handle killing the server when this script exits.