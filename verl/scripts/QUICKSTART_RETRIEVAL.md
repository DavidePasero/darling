# Quick Start: Retrieval-Based Training

## What We Built

A **clean, modular retrieval system** for query rewriting with FAISS rewards:

1. **`Retriever`** - Handles embedding + FAISS retrieval (batched, efficient)
2. **`DocumentDataset`** - Manages labels and computes rewards (NDCG, Recall, Precision)
3. **Test script** - Verify FAISS works BEFORE training
4. **Simple training script** - Quality-only rewards (no diversity) to start

## Step-by-Step Guide

### Step 1: Test Your FAISS Setup

**Before any training**, verify your retrieval infrastructure:

```bash
cd /home/lukas/Projects/darling/verl

python scripts/test_retrieval_system.py \
    --faiss-index /path/to/your/msmarco.faiss \
    --id-mapping /path/to/your/id_mapping.pkl \
    --labels /path/to/your/labels.pkl \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --device cuda
```

**This tests**:
- ✅ FAISS index loads
- ✅ Embedding model works
- ✅ Query encoding (batched)
- ✅ Retrieval returns results
- ✅ Union/intersection modes
- ✅ Reward computation (NDCG, Recall)

**Expected output**:
```
================================================================================
TESTING RETRIEVAL SYSTEM
================================================================================

================================================================================
Step 1: Initialize Retriever
================================================================================
Loading embedding model: Qwen/Qwen3-Embedding-0.6B
Model dimension: 768
Loading FAISS index: /path/to/msmarco.faiss
Index on GPU: 8841823 vectors
Loading ID mapping: /path/to/id_mapping.pkl
✓ Loaded 8841823 document IDs
Retriever ready!

================================================================================
Step 2: Initialize Document Dataset
================================================================================
Loading document labels: /path/to/labels.pkl
✓ Loaded labels for 502939 queries
✓ Average relevant docs per query: 1.12

...

================================================================================
✅ ALL TESTS PASSED!
================================================================================
```

### Step 2: Train with Quality-Only Rewards

Once Step 1 passes, start training:

```bash
# 1. Edit the script with your paths
vim scripts/train_retrieval_simple.sh

# Update these lines:
FAISS_INDEX="/path/to/your/msmarco.faiss"
ID_MAPPING="/path/to/your/id_mapping.pkl"
LABELS="/path/to/your/labels.pkl"
TRAIN_DATA="/path/to/your/train.parquet"
VAL_DATA="/path/to/your/val.parquet"

# 2. Run training
bash scripts/train_retrieval_simple.sh
```

**Key parameters in the script**:
```bash
reward_model.enable=False                    # ← DON'T load Athene
reward_model.reward_manager=retrieval        # ← Use retrieval manager
+reward_model.diversity_enabled=False        # ← Start without diversity
```

### Step 3: Monitor Training

Watch for these in console output:

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

**Good signs**:
- FAISS loads without errors
- Rewards are > 0 (not all zeros)
- Rewards increase over epochs
- No OOM errors

### Step 4: Add Diversity (Optional)

Once quality-only works, enable diversity:

```bash
# In train_retrieval_single_gpu.sh, change:
+reward_model.diversity_enabled=True
+reward_model.diversity_method=local         # Uses DeBERTa
+reward_model.deberta_model=microsoft/deberta-v3-large
+reward_model.threshold=0.5
```

## Architecture Benefits

### Old Way (Current Code)
```python
# Everything mixed together in one place
def compute_quality_rewards(...):
    # Encode queries
    # Search FAISS
    # Map IDs
    # Compute NDCG
    # All in one function
```

### New Way (Refactored)
```python
# Clean separation of concerns

# 1. Retrieval
retriever = Retriever(faiss_index, embedding_model, id_mapping)
results = retriever.retrieve_batch(query_rewrites, k=10, mode="union")

# 2. Rewards
doc_dataset = DocumentDataset(labels)
rewards = doc_dataset.compute_rewards_batch(uids, doc_ids, method="ndcg")

# 3. Testing
# Can test retriever independently before training!
```

**Advantages**:
- ✅ **Testable**: Verify FAISS works before training
- ✅ **Debuggable**: Clear where issues occur
- ✅ **Reusable**: Use `Retriever` for other tasks
- ✅ **Efficient**: Batched operations throughout
- ✅ **Flexible**: Easy to swap embedding models or retrieval methods

## File Structure

```
verl/
├── retrieval/
│   ├── engine/
│   │   ├── retriever.py          ← NEW: Unified retrieval interface
│   │   ├── document_dataset.py   ← NEW: Reward computation
│   │   └── vectordatabase.py     ← Existing FAISS wrapper
│   ├── reward/
│   │   └── reward_manager.py     ← OLD: Can be refactored to use new classes
│   └── README.md                 ← NEW: Architecture documentation
│
├── scripts/
│   ├── test_retrieval_system.py  ← NEW: Test script
│   ├── train_retrieval_simple.sh ← NEW: Simple training (quality-only)
│   ├── train_retrieval_single_gpu.sh ← NEW: Full training (quality+diversity)
│   ├── RETRIEVAL_TRAINING_README.md  ← Usage guide
│   └── REQUIRED_PARAMETERS.md        ← Parameter reference
│
└── verl/
    ├── trainer/ppo/reward.py     ← UPDATED: Added retrieval case
    └── workers/reward_manager/
        └── retrieval.py          ← Can be refactored to use new classes
```

## What Gets Loaded in Memory

### ❌ What is NOT loaded (Athene bypassed):
- Athene reward model
- Any other reward model weights

### ✅ What IS loaded:
- FAISS index (GPU, with FP16 compression)
- Embedding model (Qwen3-Embedding-0.6B)
- ID mapping (small, ~8M integers)
- Labels dictionary (small, ~500K mappings)
- DeBERTa (only if diversity enabled with `local` method)

**Memory estimate (single GPU)**:
- Base model (Qwen 0.5B): ~2GB
- Embedding model (0.6B): ~2GB
- FAISS index (GPU, FP16): ~2-4GB
- vLLM rollout: ~2GB
- Training overhead: ~2GB
- **Total: ~10-12GB** (fits on RTX 3090/4090)

## Typical Workflow

1. **Build FAISS index** (one-time):
   ```python
   from verl.retrieval.engine.vectordatabase import FaissDB

   db = FaissDB(model_name="Qwen/Qwen3-Embedding-8B")
   db.build_index(corpus, nlist=4096, m=32)
   db.save_index("msmarco.faiss")
   ```

2. **Test retrieval** (verify setup):
   ```bash
   python scripts/test_retrieval_system.py --faiss-index ... --id-mapping ... --labels ...
   ```

3. **Train quality-only** (baseline):
   ```bash
   bash scripts/train_retrieval_simple.sh
   ```

4. **Add diversity** (full DARLING):
   ```bash
   # Edit script to enable diversity
   bash scripts/train_retrieval_single_gpu.sh
   ```

5. **Evaluate** (use evals/):
   ```bash
   cd evals/math_evaluation
   bash sh/eval.sh ...
   ```

## Common Issues & Solutions

### Issue: "FAISS index loading fails"
**Solution**: Check paths and file permissions
```bash
ls -lh /path/to/msmarco.faiss
file /path/to/msmarco.faiss  # Should say "data"
```

### Issue: "All rewards are 0.0"
**Solution**: Check labels format and UID matching
```python
import pickle
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)
print(f"Format: {type(labels)}")  # Should be dict
print(f"Sample: {list(labels.items())[:3]}")
```

### Issue: "KeyError: 'uid'"
**Solution**: Add `uid` column to your Parquet data
```python
import pandas as pd
df = pd.read_parquet("train.parquet")
print(df.columns)  # Should include 'uid' and 'prompt'
```

### Issue: "OOM during training"
**Solution**: Reduce memory usage
```bash
# In script, adjust:
BATCH_SIZE=16                    # Reduce from 32
N_REWRITES=2                     # Reduce from 4
GPU_MEMORY_UTIL=0.3              # Reduce from 0.4
OFFLOAD_PARAMS=true              # Enable offloading
```

## Next Steps

Once basic training works:

1. **Experiment with quality metrics**:
   - `ndcg` - Rank-aware (default)
   - `recall` - Count-based
   - `hit` - Binary success

2. **Enable diversity**:
   - Start with `local` method (DeBERTa)
   - Try different thresholds (0.3 - 0.7)
   - Switch to `vllm` for production

3. **Scale up**:
   - Larger model (7B instead of 0.5B)
   - More GPUs (`trainer.n_gpus_per_node=4`)
   - Bigger batches

4. **Evaluate**:
   - MS MARCO dev set
   - nDCG@10, Recall@100
   - Diversity metrics

## Questions?

See detailed docs:
- `retrieval/README.md` - Architecture details
- `scripts/RETRIEVAL_TRAINING_README.md` - Training guide
- `scripts/REQUIRED_PARAMETERS.md` - Parameter reference
- `docs/retrieval_rewards_guide.md` - Full usage guide