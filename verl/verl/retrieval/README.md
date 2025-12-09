# Retrieval System Architecture

## Overview

This retrieval system provides a clean, modular architecture for query rewriting with FAISS-based retrieval rewards.

**Design Philosophy:**
- **Separation of concerns**: Retrieval, reward computation, and training are decoupled
- **Batched operations**: Efficient encoding and retrieval
- **Multiple rewrites**: Native support for query rewriting scenarios
- **Testing before training**: Standalone test script to verify FAISS setup

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval System                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌────────────────────┐               │
│  │  Retriever   │───▶│   FAISS Index      │               │
│  │              │    │   (GPU/CPU)        │               │
│  └──────────────┘    └────────────────────┘               │
│        │                                                    │
│        │ uses                                               │
│        ↓                                                    │
│  ┌──────────────────────┐                                  │
│  │  Embedding Model     │                                  │
│  │  (SentenceTransformer)│                                 │
│  └──────────────────────┘                                  │
│                                                             │
│  ┌────────────────────────────────────────┐               │
│  │  DocumentDataset                        │               │
│  │  - Stores labels (UID -> doc_ids)       │               │
│  │  - Computes NDCG, Recall, Precision    │               │
│  └────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. `Retriever` (engine/retriever.py)

**Purpose**: Unified interface for embedding + FAISS retrieval

**Key Features**:
- Batched text encoding with proper model prompts
- FAISS search (GPU/CPU)
- Multiple rewrites per query with union/intersection/first modes
- Document ID mapping

**Interface**:
```python
retriever = Retriever(
    faiss_index_path="msmarco.faiss",
    embedding_model="Qwen/Qwen3-Embedding-0.6B",
    id_mapping_path="id_mapping.pkl"
)

# Simple search
scores, indices = retriever.search(["query 1", "query 2"], k=10)

# Query rewriting scenario (list of lists)
query_rewrites = [
    ["capital of france", "france capital", "paris"],  # Query 1 rewrites
    ["deep learning", "neural networks"]  # Query 2 rewrites
]
results = retriever.retrieve_batch(query_rewrites, k=10, mode="union")
```

**Modes for Multiple Rewrites**:
- `union`: Combine all retrieved docs, average scores (default)
- `intersection`: Only docs retrieved by ALL rewrites
- `first`: Use only first rewrite

### 2. `DocumentDataset` (engine/document_dataset.py)

**Purpose**: Manage labels and compute retrieval rewards

**Key Features**:
- Memory efficient (stores only UID -> doc_ids mapping)
- Supports NDCG, Recall, Precision, Hit@k
- Batched reward computation

**Interface**:
```python
doc_dataset = DocumentDataset(labels_path="labels.pkl")

# Get relevant docs
relevant = doc_dataset.get_relevant_docs("msmarco_123")

# Compute single reward
ndcg = doc_dataset.compute_ndcg(
    query_uid="msmarco_123",
    retrieved_doc_ids=[10, 25, 30, 50],
    k=10
)

# Batch computation
rewards = doc_dataset.compute_rewards_batch(
    query_uids=["uid1", "uid2", "uid3"],
    retrieved_doc_ids_batch=[[10, 25], [5, 15], [100]],
    method="ndcg",
    k=10
)
```

## Testing Your Setup

Before training, test your FAISS infrastructure:

```bash
cd verl

python scripts/test_retrieval_system.py \
    --faiss-index /path/to/msmarco.faiss \
    --id-mapping /path/to/id_mapping.pkl \
    --labels /path/to/labels.pkl \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --device cuda
```

**This will test**:
1. ✅ FAISS index loads correctly
2. ✅ Embedding model works
3. ✅ Query encoding is batched
4. ✅ Retrieval returns valid results
5. ✅ Union/intersection modes work
6. ✅ Reward computation (NDCG, Recall) works

## Integration with Reward Module

The reward module (`verl/workers/reward_manager/retrieval.py`) uses these components:

```python
# In RetrievalRewardManager.__init__
self.retriever = Retriever(
    faiss_index_path=config.faiss_index_path,
    embedding_model=config.embedding_model,
    id_mapping_path=config.id_mapping_path
)

self.doc_dataset = DocumentDataset(
    labels_path=config.labels_path
)

# In verify() method
def verify(self, data):
    # Decode responses (query rewrites)
    responses_str = [...]
    uid = data.non_tensor_batch["uid"]

    # Retrieve for each rewrite
    scores, indices = self.retriever.search(responses_str, k=self.k)
    doc_ids = self.retriever.map_indices_to_ids(indices)

    # Compute rewards
    quality_rewards = self.doc_dataset.compute_rewards_batch(
        query_uids=uid,
        retrieved_doc_ids_batch=doc_ids.tolist(),
        method=self.quality_method,
        k=self.k
    )

    return quality_rewards
```

## File Requirements

### FAISS Index (`msmarco.faiss`)
- Binary FAISS index file
- Created using `FaissDB.build_index()` or external tools
- Contains dense embeddings of document corpus

### ID Mapping (`id_mapping.pkl`)
- Python pickle file
- Format: `List[int]`
- Maps FAISS index → real document IDs
- Example: `[pid_0, pid_1, pid_2, ...]`

### Labels (`labels.pkl`)
- Python pickle file
- Format: `Dict[str, List[int]]`
- Maps query UID → relevant document IDs
- Example:
  ```python
  {
      "msmarco_123456": [10, 25, 30],
      "msmarco_789012": [5, 15, 20, 35],
      ...
  }
  ```

### Training Data (Parquet)
Must include:
- `prompt`: Original query string
- `uid`: Query identifier (must match keys in labels.pkl)

Example:
```python
{
    "prompt": "what is the capital of france",
    "uid": "msmarco_123456"
}
```

## Reward Computation Flow

```
Training Data (Parquet)
    │
    │ uid: "msmarco_123"
    │ prompt: "capital of france"
    │
    ↓
Model generates N rewrites
    │
    ↓
["capital of france", "france capital city", "paris location"]
    │
    ↓
Retriever.search() → FAISS retrieval
    │
    ↓
Retrieved doc IDs: [[10, 25, 30, ...], [10, 50, 25, ...], ...]
    │
    ↓
DocumentDataset.compute_rewards_batch()
    │
    ├─ Lookup: labels["msmarco_123"] = [10, 25, 100]
    │
    └─ Compute NDCG@10 for each rewrite
    │
    ↓
Quality Rewards: [0.85, 0.72, 0.91, ...]
    │
    ↓
(Optional) Diversity Computation
    │
    ↓
Final Rewards: quality × diversity
```

## Performance Considerations

### Memory Optimization
- Use smaller embedding model (`0.6B` vs `8B`)
- GPU index with FP16: `co.useFloat16 = True`
- Batch encoding: processes multiple texts at once

### Speed Optimization
- Batched retrieval: encodes all queries in one batch
- GPU FAISS index: much faster than CPU
- Proper `nprobe` setting: balance speed vs quality
  - Low (16): Fast but less accurate
  - Medium (64): Good balance (default)
  - High (128+): Slower but more accurate

### Typical Performance (single GPU)
- Embedding encoding: ~1000 queries/sec
- FAISS search: ~5000 queries/sec (GPU)
- End-to-end: ~500-800 queries/sec including reward computation

## Refactoring Benefits

This new architecture enables:

1. **Independent Testing**: Test FAISS without RL framework
2. **Easy Debugging**: Clear separation of encoding, retrieval, rewards
3. **Reusability**: `Retriever` and `DocumentDataset` can be used elsewhere
4. **Flexibility**: Easy to swap embedding models or retrieval methods
5. **Performance**: Batched operations throughout

## Next Steps

1. **Test your setup**:
   ```bash
   python scripts/test_retrieval_system.py --faiss-index ... --id-mapping ... --labels ...
   ```

2. **Train with quality-only rewards**:
   ```bash
   bash scripts/train_retrieval_simple.sh
   ```

3. **Add diversity** (after quality works):
   - Enable diversity in config
   - Choose `local` (DeBERTa) or `vllm` (servers)

4. **Monitor training**:
   - Quality reward trends (should increase)
   - Retrieval metrics (NDCG@10, Recall@10)
   - Model generation quality

## Troubleshooting

### "Cannot load FAISS index"
- Check path is correct
- Verify file exists: `ls -lh /path/to/index.faiss`
- Check file permissions

### "Dimension mismatch"
- Embedding model must match FAISS index
- If you changed embedding model, rebuild FAISS index

### "KeyError in labels"
- UID in training data doesn't exist in labels.pkl
- Verify UIDs match between data and labels
- Check: `with open('labels.pkl', 'rb') as f: print(pickle.load(f).keys())`

### "Low rewards (all near 0)"
- Check labels format: should be `{uid: [doc_ids]}`
- Verify ID mapping is correct
- Try different queries to ensure FAISS works

### "OOM during retrieval"
- Reduce batch size in `retriever.search()`
- Use CPU index instead of GPU
- Use smaller embedding model

## References

- **FAISS**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/
- **MS MARCO**: https://microsoft.github.io/msmarco/
- **NDCG**: https://en.wikipedia.org/wiki/Discounted_cumulative_gain