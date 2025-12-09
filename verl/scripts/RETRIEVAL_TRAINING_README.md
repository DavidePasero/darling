# Retrieval Reward Training Guide

## Quick Start - Simple Quality-Only Training

Use `train_retrieval_simple.sh` to test your retrieval infrastructure first (NO diversity).

### Required Configuration

**1. Update these paths in the script:**
```bash
FAISS_INDEX="/path/to/your/msmarco.faiss"
ID_MAPPING="/path/to/your/id_mapping.pkl"
LABELS="/path/to/your/labels.pkl"
TRAIN_DATA="/path/to/your/train.parquet"
VAL_DATA="/path/to/your/val.parquet"
```

**2. Your training data (Parquet) must include:**
- `prompt`: Original query string
- `uid`: Unique query identifier (used to look up labels)

Example:
```python
{
    "prompt": "what is the capital of france",
    "uid": "msmarco_123456"
}
```

**3. Your labels file (`labels.pkl`) should be:**
```python
# Dictionary mapping uid -> list of relevant document IDs
{
    "msmarco_123456": [12345, 67890, 11111],  # Relevant doc IDs for this query
    "msmarco_789012": [22222, 33333],
    ...
}
```

### Key Parameters Explained

#### Critical for Avoiding Athene Model

```bash
reward_model.enable=False                    # DON'T load any reward model
reward_model.reward_manager=retrieval        # Use retrieval reward manager
```

This ensures Athene is **NEVER loaded into memory**.

#### Required Retrieval Parameters

```bash
# FAISS infrastructure
+reward_model.faiss_index_path=${FAISS_INDEX}
+reward_model.id_mapping_path=${ID_MAPPING}
+reward_model.labels_path=${LABELS}

# Quality metric
+reward_model.quality_method=ndcg            # Options: ndcg, recall, hit
+reward_model.k=10                           # Top-k retrieval

# Embedding model (must match FAISS index!)
+reward_model.embedding_model=Qwen/Qwen3-Embedding-0.6B

# Diversity (start with disabled)
+reward_model.diversity_enabled=False

# Reward function
+reward_model.custom_diversity_function.path=verl.utils.reward_score.retrieval_rewards
+reward_model.custom_diversity_function.name=retrieval_reward
```

#### Single GPU Optimization

```bash
trainer.n_gpus_per_node=1
trainer.nnodes=1

actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.4

actor_rollout_ref.actor.fsdp_config.param_offload=true
```

## How It Works

### 1. Quality Reward Computation

For each generated query rewrite:
1. Embed the rewrite using `embedding_model`
2. Search FAISS index for top-k documents
3. Look up ground truth labels using `uid` from training data
4. Calculate quality metric (NDCG@k, Recall@k, or Hit@k)

Example:
```python
# Original query: "capital of france"
# Rewrite: "what is the capital city of france"
#
# FAISS retrieves: [doc_123, doc_456, doc_789, ...]
# Ground truth (from labels.pkl): [doc_123, doc_999]
#
# NDCG@10 = 0.45  (doc_123 found at rank 1)
```

### 2. No Diversity (Simple Mode)

When `diversity_enabled=False`:
- Only quality rewards are computed
- Diversity component = 1.0 for all rewrites
- Final reward = quality_reward × 1.0 = quality_reward

This is the **recommended starting point** to verify:
- FAISS index loads correctly
- ID mapping works
- Labels dictionary has correct format
- UID field exists in training data

## Running the Script

```bash
# 1. Edit paths in the script
vim verl/scripts/train_retrieval_simple.sh

# 2. Run training
cd verl
bash scripts/train_retrieval_simple.sh
```

## Expected Output

During training, you should see:
```
================================================================================
Initializing RetrievalRewardManager
================================================================================
Loading FAISS infrastructure...
✓ Loaded FAISS index: /path/to/msmarco.faiss
✓ Loaded ID mapping: 8841823 documents
✓ Loaded labels: 502939 queries
Quality reward: ndcg@10
Diversity method: False
================================================================================

[Retrieval Reward Example - retrieval]
UID: msmarco_123456
[prompt] what is the capital of france...
[rewrite] what is the capital city of france...
[score] 0.4532
================================================================================
```

## Troubleshooting

### Issue: "FAISS index path not found"
- Check that `FAISS_INDEX` path is correct
- Verify index file exists: `ls -lh /path/to/msmarco.faiss`

### Issue: "KeyError: 'uid'"
- Your Parquet data is missing the `uid` field
- Add `uid` column to your training data

### Issue: "Low quality rewards (all near 0.0)"
- Check that embedding model matches FAISS index
- Verify labels dictionary format: `{uid: [doc_ids]}`
- Confirm ID mapping is correct

### Issue: "OOM (Out of Memory)"
- Reduce `BATCH_SIZE` (try 16 or 8)
- Reduce `N_REWRITES` (try 2)
- Set `OFFLOAD_PARAMS=true`
- Lower `GPU_MEMORY_UTIL` (try 0.3)

## Next Steps

Once simple quality-only training works:
1. Switch to `train_retrieval_single_gpu.sh`
2. Enable diversity: `+reward_model.diversity_enabled=True`
3. Choose diversity method: `local` (DeBERTa) or `vllm` (servers)

## Quality Metrics Comparison

| Metric | Description | When to Use |
|--------|-------------|-------------|
| **ndcg** | Normalized Discounted Cumulative Gain | Default - rank matters |
| **recall** | Fraction of relevant docs retrieved | Simple baseline |
| **hit** | Binary - any relevant doc in top-k | Very sparse labels |

For MS MARCO, use **ndcg@10** (industry standard).