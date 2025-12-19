# Retrieval-Based Rewards for Query Rewriting

This guide explains how to use the retrieval-based reward system for training query rewriting models with DARLING-style diversity rewards.

## Overview

The retrieval reward system combines two types of rewards:

1. **Quality Reward**: Measures retrieval effectiveness (NDCG@k, Recall@k, or Hit@k) using FAISS vector search
2. **Diversity Reward**: Measures semantic diversity of query rewrites using partition-based clustering

**Final Reward**: `r_final = r_quality × normalized_diversity` (multiplicative combination, following DARLING)

## Architecture

```
┌─────────────────────────────────────────────┐
│ Input: Original Query                       │
│ Model generates N rewrites                  │
└────────────┬────────────────────────────────┘
             │
    ┌────────▼─────────────┐
    │ For each rewrite:    │
    │ 1. FAISS retrieval   │
    │ 2. Compute NDCG@10   │
    └────────┬─────────────┘
             │
    ┌────────▼─────────────────────────┐
    │ Group rewrites by original query │
    │ Partition by semantic similarity │
    │ Calculate diversity scores       │
    └────────┬─────────────────────────┘
             │
    ┌────────▼─────────────────────────┐
    │ Combine: quality × diversity     │
    │ Normalize per query group        │
    └──────────────────────────────────┘
```

## Prerequisites

### 1. Build FAISS Index

First, you need to build a FAISS vector database for your document collection:

```python
from verl.retrieval.engine.vectordatabase import FaissDB
from verl.retrieval.dataset_utils import get_ms_marco_dataset
import pickle

# Load your corpus
corpus, corpus_ids, queries, qrels = get_ms_marco_dataset(split="train")

# Build FAISS index
db = FaissDB(model_name="Qwen/Qwen3-Embedding-8B", device="cuda")
db.build_index(corpus, nlist=4096, m=32)
db.save_index("msmarco.faiss")

# Save ID mapping
with open("id_mapping.pkl", "wb") as f:
    pickle.dump(corpus_ids, f)

# Save labels (query ID -> relevant doc IDs)
with open("labels.pkl", "wb") as f:
    pickle.dump(qrels, f)
```

### 2. Prepare Training Data

Your training data (Parquet format) must include:

- `prompt`: Original user query
- `uid`: Unique identifier for the query (used for grouping rewrites)
- Additional metadata as needed

Example Parquet schema:
```python
{
    "prompt": "what is the capital of france",
    "uid": "query_12345",
    "data_source": "msmarco_train"
}
```

The `uid` field is critical - it's used to:
1. Look up ground truth labels from `labels.pkl`
2. Group multiple rewrites for diversity calculation

## Configuration

### Option 1: Using Configuration File

Create a YAML config (e.g., `config/retrieval_grpo.yaml`):

```yaml
reward_model:
  enable: False  # Disable learned RM
  reward_manager: retrieval  # Use RetrievalRewardManager

  # FAISS Infrastructure
  faiss_index_path: ~/data/msmarco/msmarco.faiss
  id_mapping_path: ~/data/msmarco/id_mapping.pkl
  labels_path: ~/data/msmarco/labels.pkl

  # Quality Reward
  quality_method: ndcg  # Options: ndcg, recall, hit
  k: 10

  # Diversity Configuration
  diversity_enabled: True
  diversity_method: local  # Options: local, vllm
  deberta_model: microsoft/deberta-v3-large
  threshold: 0.5

  # Embedding model
  embedding_model: Qwen/Qwen3-Embedding-8B

  # Custom reward function
  custom_diversity_function:
    path: verl.utils.reward_score.retrieval_rewards
    name: retrieval_reward
```

### Option 2: Using Shell Script

```bash
#!/bin/bash
FAISS_INDEX="/path/to/msmarco.faiss"
ID_MAPPING="/path/to/id_mapping.pkl"
LABELS="/path/to/labels.pkl"

python3 -m verl.trainer.main_ppo \
    reward_model.enable=False \
    reward_model.reward_manager=retrieval \
    +reward_model.faiss_index_path=${FAISS_INDEX} \
    +reward_model.id_mapping_path=${ID_MAPPING} \
    +reward_model.labels_path=${LABELS} \
    +reward_model.quality_method=ndcg \
    +reward_model.k=10 \
    +reward_model.diversity_enabled=True \
    +reward_model.diversity_method=local \
    +reward_model.custom_diversity_function.path=verl.utils.reward_score.retrieval_rewards \
    +reward_model.custom_diversity_function.name=retrieval_reward \
    ...
```

## Diversity Methods

### Local DeBERTa (Recommended for Getting Started)

Uses a local DeBERTa model for semantic similarity classification.

**Pros**:
- Simple setup (no external servers)
- Works out of the box

**Cons**:
- Slower than VLLM (sequential pairwise comparisons)
- Limited to single GPU

**Configuration**:
```yaml
diversity_method: local
deberta_model: microsoft/deberta-v3-large
threshold: 0.5
device: cuda
```

### VLLM Server (Recommended for Production)

Uses VLLM-served classifiers for distributed, asynchronous similarity checks.

**Pros**:
- Much faster (parallel async requests)
- Scalable to multiple GPUs

**Cons**:
- Requires setting up VLLM servers

**Configuration**:
```yaml
diversity_method: vllm
vllm_hostname: your-server.com  # Or set VLLM_SERVER_HOSTNAME env var
num_vllm_servers: 8
threshold: 0.5
```

**Setup VLLM Servers**:
```bash
# Start 8 classifier servers on ports 8000-8007
for i in {0..7}; do
    vllm serve microsoft/deberta-v3-large \
        --port $((8000 + i)) \
        --gpu-memory-utilization 0.3 \
        --tensor-parallel-size 1 &
done

export VLLM_SERVER_HOSTNAME=localhost
```

## Training Examples

### Quick Start (Single GPU)

```bash
cd verl/wildchat_scripts
bash retrieval.sh
```

Edit the script to update paths:
- `FAISS_INDEX`
- `ID_MAPPING`
- `LABELS`
- `TRAIN_DATA`
- `VAL_DATA`

### Full Configuration

See `verl/examples/grpo_trainer/run_qwen_retrieval.sh` for a complete example with all hyperparameters.

## Hyperparameter Tuning

### Quality Reward Method

| Method | Description | Use When |
|--------|-------------|----------|
| **ndcg** | Normalized Discounted Cumulative Gain | Rank matters (preferred) |
| **recall** | Fraction of relevant docs retrieved | Simple baseline |
| **hit** | Binary: any relevant doc in top-k | Very sparse labels |

### Top-k Value

- `k=10`: Standard for MS MARCO
- `k=100`: Better for recall-oriented tasks
- Higher k = more lenient rewards

### Diversity Threshold

- `threshold=0.5`: Balanced (default)
- Lower threshold (0.3): Stricter diversity (fewer partitions)
- Higher threshold (0.7): Looser diversity (more partitions)

### Temperature

- `temperature=0.7`: Moderate diversity in rewrites
- Higher (1.0): More diverse but potentially lower quality
- Lower (0.3): Less diverse but higher quality

### Number of Rewrites (N)

- `N=8`: Standard for DARLING (balances compute and diversity)
- `N=4`: Faster training, less diversity signal
- `N=16`: Stronger diversity signal, more compute

## Monitoring

### Key Metrics to Track

1. **Quality Reward** (NDCG@10):
   - Should improve over training
   - Expected range: 0.2-0.6 for MS MARCO

2. **Diversity Score**:
   - Number of unique semantic partitions
   - Higher = more diverse rewrites

3. **Final Reward** (Quality × Diversity):
   - Combined metric
   - Should increase steadily

### W&B Logging

The system automatically logs:
- `reward_tensor`: Final combined rewards
- `acc`: Mean reward per batch (stored in data.batch["acc"])

Enable detailed logging:
```yaml
trainer:
  logger: ['console', 'wandb']
  log_val_generations: 10  # Log example rewrites
```

## Troubleshooting

### Issue: Low Quality Rewards

**Possible causes**:
1. FAISS index doesn't match embedding model
2. ID mapping incorrect
3. Labels dictionary has wrong format

**Solution**:
```python
# Verify index loading
from verl.retrieval.engine.vectordatabase import FaissDB
db = FaissDB(model_name="Qwen/Qwen3-Embedding-8B")
db.load_index("msmarco.faiss")
scores, indices = db.search(["test query"], k=10)
print(indices)  # Should be valid indices

# Verify labels
import pickle
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)
print(labels)  # Should be {uid: [doc_ids]}
```

### Issue: OOM During Training

**Solutions**:
1. Reduce `N` (number of rewrites)
2. Reduce `actor_rollout_ref.rollout.gpu_memory_utilization`
3. Enable offloading: `actor_rollout_ref.actor.fsdp_config.param_offload=True`
4. Use smaller embedding model: `embedding_model: Qwen/Qwen3-Embedding-0.6B`

### Issue: Diversity Rewards Always Same

**Possible causes**:
1. Threshold too low/high
2. Model generating identical rewrites
3. Classifier not loaded

**Solution**:
- Check printed examples during training
- Try different `threshold` values
- Verify DeBERTa model loads correctly

### Issue: VLLM Servers Timeout

**Solutions**:
1. Check server status: `curl http://localhost:8000/health`
2. Increase timeout in `retrieval_rewards.py`
3. Reduce `num_vllm_servers` if servers are slow

## Advanced Usage

### Custom Quality Metrics

Modify `verl/retrieval/reward/reward_manager.py` to add custom metrics:

```python
def _calculate_mrr(self, retrieved_pids, gold_set):
    """Mean Reciprocal Rank"""
    for i, pid in enumerate(retrieved_pids, 1):
        if pid in gold_set:
            return 1.0 / i
    return 0.0
```

Then set: `quality_method: mrr`

### Different Diversity Methods

You can implement custom diversity functions following the signature:

```python
def custom_diversity(
    queries: List[str],
    uid: List[str],
    **kwargs
) -> List[float]:
    """Return diversity score per query"""
    ...
```

Register in config:
```yaml
custom_diversity_function:
  path: path.to.your.module
  name: custom_diversity
```

### Per-UID Normalization

The system automatically normalizes diversity scores per UID group to ensure fair comparison within each original query's rewrites.

## References

1. **DARLING Paper**: "Jointly Reinforcing Diversity and Quality of Language Model Generations"
2. **FAISS**: Vector similarity search library
3. **GRPO**: Group Relative Policy Optimization
4. **MS MARCO**: Microsoft Machine Reading Comprehension dataset

## Example Results

Expected improvements on MS MARCO Dev:

| Metric | Baseline (Original) | Prompt-Only | Quality RL | Quality×Diversity RL |
|--------|---------------------|-------------|------------|---------------------|
| nDCG@10 | 0.35 | 0.38 | 0.42 | 0.45 |
| Recall@100 | 0.72 | 0.75 | 0.78 | 0.80 |
| Diversity | 1.0 | 2.1 | 1.2 | 5.4 |

(Numbers are illustrative - actual results depend on model and hyperparameters)

## Citation

If you use this retrieval reward system, please cite:

```bibtex
@article{darling2025,
  title={Jointly Reinforcing Diversity and Quality in Language Model Generations},
  author={Li, Tianjian and Zhang, Yiming and Yu, Ping and Saha, Swarnadeep and Khashabi, Daniel and Weston, Jason and Lanchantin, Jack and Wang, Tianlu},
  journal={arXiv preprint arXiv:2509.02534},
  year={2025}
}
```