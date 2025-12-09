# Required Parameters for Retrieval Training

## Minimal Configuration (Quality-Only)

### ✅ Required to Avoid Loading Athene

```bash
reward_model.enable=False                    # CRITICAL: Don't load any reward model
reward_model.reward_manager=retrieval        # Use retrieval reward manager instead
```

### ✅ Required FAISS Infrastructure

```bash
+reward_model.faiss_index_path=/path/to/msmarco.faiss
+reward_model.id_mapping_path=/path/to/id_mapping.pkl
+reward_model.labels_path=/path/to/labels.pkl
```

### ✅ Required Quality Configuration

```bash
+reward_model.quality_method=ndcg            # Options: ndcg, recall, hit
+reward_model.k=10                           # Top-k for retrieval
+reward_model.embedding_model=Qwen/Qwen3-Embedding-0.6B
```

### ✅ Required Diversity Configuration (Start Disabled)

```bash
+reward_model.diversity_enabled=False        # Start with False to test
```

### ✅ Required Reward Function

```bash
+reward_model.custom_diversity_function.path=verl.utils.reward_score.retrieval_rewards
+reward_model.custom_diversity_function.name=retrieval_reward
```

### ✅ Required Data Format

Your Parquet training data **MUST** include:
```python
{
    "prompt": "your query here",
    "uid": "unique_query_id"
}
```

The `uid` is used to:
1. Look up ground truth labels from `labels.pkl`
2. Group multiple rewrites for diversity calculation (when enabled)

## Complete Minimal Command

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/val.parquet \
    data.prompt_key="prompt" \
    data.train_batch_size=32 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B \
    actor_rollout_ref.rollout.n=4 \
    \
    reward_model.enable=False \
    reward_model.reward_manager=retrieval \
    +reward_model.faiss_index_path=/path/to/msmarco.faiss \
    +reward_model.id_mapping_path=/path/to/id_mapping.pkl \
    +reward_model.labels_path=/path/to/labels.pkl \
    +reward_model.quality_method=ndcg \
    +reward_model.k=10 \
    +reward_model.embedding_model=Qwen/Qwen3-Embedding-0.6B \
    +reward_model.diversity_enabled=False \
    +reward_model.custom_diversity_function.path=verl.utils.reward_score.retrieval_rewards \
    +reward_model.custom_diversity_function.name=retrieval_reward \
    \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1
```

## Optional Parameters (For Later)

### When Adding Diversity

```bash
+reward_model.diversity_enabled=True
+reward_model.diversity_method=local         # Or 'vllm'
+reward_model.deberta_model=microsoft/deberta-v3-large
+reward_model.threshold=0.5
+reward_model.device=cuda
```

### For VLLM Diversity (Advanced)

```bash
+reward_model.diversity_method=vllm
+reward_model.vllm_hostname=localhost        # Or set VLLM_SERVER_HOSTNAME env var
+reward_model.num_vllm_servers=8
```

## File Requirements Checklist

- [ ] FAISS index file (`.faiss`)
- [ ] ID mapping pickle file (`.pkl`) - list of document IDs
- [ ] Labels pickle file (`.pkl`) - dict of `{uid: [relevant_doc_ids]}`
- [ ] Training data (`.parquet`) with `prompt` and `uid` fields
- [ ] Validation data (`.parquet`) with `prompt` and `uid` fields

## What Gets Loaded in Memory

### ❌ What is NOT loaded (Athene is skipped):
- Athene reward model (or any other reward model)
- Reward model tokenizer
- Reward model weights

### ✅ What IS loaded:
- FAISS index (for retrieval)
- Embedding model (for query encoding)
- ID mapping (small, just a list)
- Labels dictionary (small, just query->doc mappings)
- DeBERTa classifier (only if diversity is enabled with `local` method)

## Single GPU Memory Estimate

Approximate VRAM usage:
- Base model (Qwen 0.5B): ~2GB
- Embedding model (0.6B): ~2GB
- FAISS index: minimal (loaded on demand)
- vLLM rollout: ~2GB (controlled by `gpu_memory_utilization`)
- Training overhead: ~2GB

**Total: ~8-10GB** (fits on RTX 3090 / 4090 / A6000)

Reduce if needed:
- Use smaller model (0.5B instead of 7B)
- Lower `gpu_memory_utilization=0.3`
- Enable `param_offload=True`
- Reduce batch size