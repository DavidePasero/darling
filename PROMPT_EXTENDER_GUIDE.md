# Prompt Extender Configuration Guide

## Overview

The prompt extender system allows you to add instructional prompts to your queries before they are processed by the model. This is **critical** for query rewriting tasks, as it tells the model what to do with the input query.

## What Was Added

### 1. **Prompt Extender Classes** (`verl/verl/utils/prompt_extension/`)

Seven different prompt extension strategies:

- **`NoOpPromptExtender`**: No instruction, just wraps query in user message (default for backward compatibility)
- **`RewritePromptExtender`**: Basic query rewriting instruction
- **`SemanticRewritePromptExtender`**: Semantic enrichment with synonyms and concepts
- **`MinimalRewritePromptExtender`**: Minimal instruction
- **`ContextAwareRewritePromptExtender`**: Context-aware rewriting with intent preservation
- **`BM25RewritePromptExtender`**: Optimized for BM25 keyword-based retrieval
- **`DenseVectorRewritePromptExtender`**: Optimized for dense vector semantic search (FAISS)

### 2. **Configuration Parameter**

Added `prompt_extender` to the data configuration in:
- `verl/verl/trainer/config/ppo_trainer.yaml`
- Training scripts: `train_retrieval_simple.sh`, `retrieval.sh`

### 3. **Dataset Integration**

Updated `BeirRLDataset` to accept and use the `prompt_extender` parameter.

### 4. **Trainer Integration**

Modified `create_rl_dataset()` in `main_ppo.py` to automatically pass the prompt_extender to datasets that support it.

## How to Use

### In Training Scripts

Simply set the `PROMPT_EXTENDER` variable and pass it to the training command:

```bash
# For FAISS (dense vector) retrieval
PROMPT_EXTENDER="dense_vector"

# For BM25 retrieval
PROMPT_EXTENDER="bm25"

# For general rewriting
PROMPT_EXTENDER="rewrite"

python3 -m verl.trainer.main_ppo \
    data.prompt_extender=${PROMPT_EXTENDER} \
    ...
```

### In Config Files

Add to your YAML config:

```yaml
data:
  prompt_extender: dense_vector  # Choose your strategy
  ...
```

### Via Command Line

```bash
python3 -m verl.trainer.main_ppo \
    data.prompt_extender=dense_vector \
    ...
```

## Prompt Extender Options

### 1. **no_op** (Default)
```
Input: "what is the capital of france"
Output: [{"role": "user", "content": "what is the capital of france"}]
```
**Use when**: You want no instruction (backward compatibility)

### 2. **rewrite**
```
System: "You are a query rewriting assistant. Your task is to rewrite user queries to make them more effective for document retrieval."
User: "Rewrite the following query:\nwhat is the capital of france"
```
**Use when**: General query rewriting task

### 3. **semantic**
```
System: "You are an expert in semantic query reformulation. Your task is to enhance user queries by expanding them with meaningful concepts, entities, and synonyms that improve retrieval performance."
User: "Please provide a semantically enriched rewrite of the following query:\nwhat is the capital of france"
```
**Use when**: You want semantic expansion with synonyms

### 4. **minimal**
```
System: "Rewrite the user's query."
User: "Query: what is the capital of france"
```
**Use when**: You want minimal instruction overhead

### 5. **context_aware**
```
System: "You are a context-aware query rewriting assistant. Rewrite user queries so they become clearer, more specific, and more effective for information retrieval, while strictly preserving the user's intent."
User: "Rewrite the following query in a clearer and more retrieval-oriented way:\nwhat is the capital of france"
```
**Use when**: You want intent-preserving rewrites

### 6. **bm25** (Recommended for BM25 retrieval)
```
System: "You rewrite user queries specifically for BM25 retrieval. BM25 benefits from keyword-rich queries. Rewrite the query by extracting the essential keywords, adding useful synonyms, and removing unnecessary words while preserving intent."
User: "Rewrite the query as BM25-optimized keywords (with optional synonyms):\nwhat is the capital of france"
```
**Use when**: Using BM25/sparse retrieval

### 7. **dense_vector** (Recommended for FAISS retrieval)
```
System: "You rewrite queries for dense vector semantic search. Dense search works best when the query expresses clear intent, context, and meaning. Rewrite the query to make it more explicit, semantically rich, and unambiguous, without altering the user's intent."
User: "Rewrite the following query to improve semantic embedding quality:\nwhat is the capital of france"
```
**Use when**: Using FAISS/dense vector retrieval

## Recommendations

### For Your Current Setup (FAISS Retrieval):
```bash
PROMPT_EXTENDER="dense_vector"
```

### For BM25 Retrieval:
```bash
PROMPT_EXTENDER="bm25"
```

### For Experimentation:
Try different options and compare retrieval performance:
```bash
# Baseline (no instruction)
data.prompt_extender=no_op

# Simple rewriting
data.prompt_extender=rewrite

# Optimized for your retrieval method
data.prompt_extender=dense_vector  # or bm25
```

## Example Training Command

```bash
#!/bin/bash
FAISS_INDEX="/path/to/msmarco.faiss"
ID_MAPPING="/path/to/id_mapping.pkl"
LABELS="/path/to/labels.pkl"
TRAIN_DATA="/path/to/train.parquet"
VAL_DATA="/path/to/val.parquet"

# Use dense_vector for FAISS retrieval
PROMPT_EXTENDER="dense_vector"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.prompt_key="prompt" \
    data.prompt_extender=${PROMPT_EXTENDER} \
    data.train_batch_size=32 \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    reward_model.enable=False \
    reward_model.reward_manager=retrieval \
    +reward_model.faiss_index_path=${FAISS_INDEX} \
    +reward_model.id_mapping_path=${ID_MAPPING} \
    +reward_model.labels_path=${LABELS} \
    +reward_model.quality_method=ndcg \
    +reward_model.k=10 \
    ...
```

## Debugging

To see what prompts are being generated, you can add print statements or check the debug logs during training. The prompt extender's `__repr__` method shows the system and user prompts being used.

## Custom Prompt Extenders

To create your own prompt extender:

1. Create a new class in `rewrite_prompt_extenders.py`:

```python
class MyCustomPromptExtender(BasePromptExtender):
    _REWRITE_SYSTEM_PROMPT = "Your custom system prompt"
    _REWRITE_USER_PROMPT = "Your custom user prompt prefix"
    
    def extend_prompt(self, prompt: str) -> str:
        return [
            {"role": "system", "content": self._REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": self._REWRITE_USER_PROMPT + "\n" + prompt}
        ]
    
    def __repr__(self) -> str:
        return f"MyCustomPromptExtender: {self._REWRITE_SYSTEM_PROMPT}"
```

2. Add it to `__init__.py`
3. Add it to the if-elif chain in `beir_dataset.py`
4. Use it: `data.prompt_extender=my_custom`

## Summary

The prompt extender system ensures your model knows it should **rewrite queries** rather than answer them or continue them randomly. This is essential for effective query rewriting training!
