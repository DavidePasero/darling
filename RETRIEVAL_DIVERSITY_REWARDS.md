# Retrieval Quality + Diversity Rewards

## Overview

The system now supports combining retrieval-based quality rewards with diversity rewards, similar to the DARLING setup but using retrieval metrics instead of Athene.

## Architecture

```
Quality Reward (retrieval):
  RetrievalRewardManager → NDCG/Recall/Precision/Hit

Diversity Reward (diversity):
  DiversityRewardManager → Partition-based diversity

Combined Reward:
  multiplicative: quality * (1 + diversity)
  OR additive: quality + diversity
```

## Configuration

### Basic Setup (Quality Only)

```bash
python3 -m verl.trainer.main_ppo \
    reward_model.reward_manager=retrieval \
    +reward_model.retriever_type=faiss \
    +reward_model.quality_method=ndcg \
    +reward_model.k=10 \
    ...
```

### Quality + Diversity

```bash
python3 -m verl.trainer.main_ppo \
    reward_model.reward_manager=retrieval \
    +reward_model.diversity_reward_manager=diversity \
    +reward_model.retriever_type=faiss \
    +reward_model.quality_method=ndcg \
    +reward_model.k=10 \
    +reward_model.multiplicative=true \
    +reward_model.lambda_rm_rescale=1.0 \
    +reward_model.lambda_rule_rescale=0.1 \
    ...
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward_manager` | `"naive"` | Quality reward manager (`"retrieval"`) |
| `diversity_reward_manager` | `"diversity"` | Diversity reward manager |
| `multiplicative` | `false` | Use multiplicative combination |
| `lambda_rm_rescale` | `1.0` | Quality reward scale |
| `lambda_rule_rescale` | `1.0` | Diversity reward scale |
| `normalize_reward_before_combine` | `false` | Normalize before combining |
| `add_bias` | `false` | Add bias to make rewards positive |

## Reward Combination Modes

### 1. Multiplicative (Recommended)

```python
combined = quality * (1 + diversity * lambda_rule_rescale)
```

```bash
+reward_model.multiplicative=true \
+reward_model.lambda_rm_rescale=1.0 \
+reward_model.lambda_rule_rescale=0.1
```

### 2. Additive

```python
combined = quality * lambda_rm_rescale + diversity * lambda_rule_rescale
```

```bash
+reward_model.multiplicative=false \
+reward_model.lambda_rm_rescale=1.0 \
+reward_model.lambda_rule_rescale=0.1
```

### 3. Normalized Additive

```python
quality_norm = normalize_by_uid(quality)
diversity_norm = normalize_by_uid(diversity)
combined = quality_norm * lambda_rm_rescale + diversity_norm * lambda_rule_rescale
```

```bash
+reward_model.normalize_reward_before_combine=true \
+reward_model.lambda_rm_rescale=1.0 \
+reward_model.lambda_rule_rescale=1.0
```

## Complete Example

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=datasets/fiqa/fiqa_train.parquet \
    data.val_files=datasets/fiqa/fiqa_dev.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    \
    reward_model.enable=False \
    reward_model.reward_manager=retrieval \
    +reward_model.diversity_reward_manager=diversity \
    \
    +reward_model.retriever_type=faiss \
    +reward_model.faiss_index_path=datasets/fiqa/faiss_index.faiss \
    +reward_model.id_mapping_path=datasets/fiqa/id_mapping.pkl \
    +reward_model.beir_dataset_path=datasets/fiqa \
    +reward_model.quality_method=ndcg \
    +reward_model.k=10 \
    +reward_model.embedding_mode=local \
    +reward_model.device=cuda \
    +reward_model.faiss_device=cpu \
    \
    +reward_model.multiplicative=true \
    +reward_model.lambda_rm_rescale=1.0 \
    +reward_model.lambda_rule_rescale=0.1 \
    \
    trainer.n_gpus_per_node=1 \
    trainer.total_epochs=3
```

## How It Works

1. **Quality Reward** (`reward_fn`):
   - Loaded with `diversity=False`
   - Uses `reward_model.reward_manager` (e.g., `"retrieval"`)
   - Computes NDCG/Recall/etc. for each response

2. **Diversity Reward** (`diversity_reward_fn`):
   - Loaded with `diversity=True`
   - Uses `reward_model.diversity_reward_manager` (default: `"diversity"`)
   - Computes partition-based diversity

3. **Combination** (in `ray_trainer.py`):
   - Both rewards computed per response
   - Combined using multiplicative or additive mode
   - Scaled by lambda parameters

## Metrics Logged

```
actor/quality          - Mean quality reward (NDCG/Recall/etc.)
actor/diversity        - Mean diversity reward
actor/rm_norm_mean     - Normalized quality (if normalize=true)
actor/rule_norm_mean   - Normalized diversity (if normalize=true)
```

## Troubleshooting

### Diversity reward is 0
Check that `diversity_reward_manager` is set:
```bash
+reward_model.diversity_reward_manager=diversity
```

### Only getting quality rewards
Ensure `lambda_rule_rescale` is non-zero:
```bash
+reward_model.lambda_rule_rescale=0.1
```

### Rewards are negative
Use multiplicative mode or add bias:
```bash
+reward_model.multiplicative=true
# OR
+reward_model.add_bias=true
```

## Summary

- **Quality**: Retrieval metrics (NDCG, Recall, etc.)
- **Diversity**: Partition-based diversity
- **Combination**: Multiplicative or additive
- **Configuration**: Separate reward managers for quality and diversity

This setup matches the DARLING approach but uses retrieval rewards instead of Athene!
