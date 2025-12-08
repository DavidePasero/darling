# Athene Reward Model Scoring Script

This script allows you to compute reward scores for prompt-response pairs using the Athene reward model.

## Setup

Make sure you have the required dependencies installed:

```bash
pip install torch transformers tqdm
```

## Usage

### Basic Usage

```bash
python score_athene_rewards.py --input example_input.json --output rewards_output.json
```

### With Custom Model Path

```bash
python score_athene_rewards.py \
    --model-path /path/to/Athene-RM-8B \
    --input your_data.json \
    --output your_rewards.json \
    --batch-size 16
```

### All Options

```bash
python score_athene_rewards.py \
    --model-path /checkpoint/ram/tianjian/reward_models/Athene-RM-8B \
    --input data.json \
    --output rewards.json \
    --batch-size 8 \
    --device auto \
    --log-file rewards.log
```

## Input Format

Your input JSON file should be a list of objects with `prompt` and `generation` fields:

```json
[
  {
    "prompt": "What is the capital of France?",
    "generation": "The capital of France is Paris."
  },
  {
    "prompt": "Write a poem about nature.",
    "generation": "Trees whisper in the breeze..."
  }
]
```

See `example_input.json` for a complete example.

## Output Format

The script produces a JSON file with rewards for each prompt-response pair:

```json
[
  {
    "prompt": "What is the capital of France?",
    "generation": "The capital of France is Paris.",
    "reward": 2.45
  },
  {
    "prompt": "Write a poem about nature.",
    "generation": "Trees whisper in the breeze...",
    "reward": 1.87
  }
]
```

The script also prints summary statistics:

```
=== Reward Statistics ===
Number of samples: 100
Mean reward: 2.34
Min reward: -1.23
Max reward: 5.67
Median reward: 2.31
```

## Understanding Rewards

- **Higher rewards** indicate better quality responses according to the Athene reward model
- The Athene model is trained to score responses based on helpfulness, accuracy, and instruction-following
- Rewards are typically in the range of -5 to +5, but can vary

## Batch Processing

For large datasets, adjust the batch size based on your GPU memory:

- **8GB GPU**: `--batch-size 4`
- **16GB GPU**: `--batch-size 8`
- **24GB+ GPU**: `--batch-size 16` or higher

## Troubleshooting

### Out of Memory Error

Reduce the batch size:
```bash
python score_athene_rewards.py --input data.json --batch-size 4
```

### Model Not Found

Update the model path to point to your local Athene model:
```bash
python score_athene_rewards.py --model-path /your/path/to/Athene-RM-8B --input data.json
```

### Flash Attention Error

If you get flash attention errors, you may need to install it:
```bash
pip install flash-attn --no-build-isolation
```

Or modify the script to use standard attention by changing `attn_implementation="flash_attention_2"` to `attn_implementation="eager"`.

## Advanced Usage

### Processing Multiple Files

You can create a simple bash script to process multiple files:

```bash
#!/bin/bash
for file in data/*.json; do
    output="results/$(basename $file)"
    python score_athene_rewards.py --input "$file" --output "$output"
done
```

### Filtering by Reward Threshold

After scoring, you can filter results with Python:

```python
import json

with open('rewards_output.json', 'r') as f:
    results = json.load(f)

# Filter for high-quality responses (reward > 2.0)
high_quality = [r for r in results if r['reward'] > 2.0]

with open('high_quality.json', 'w') as f:
    json.dump(high_quality, f, indent=2)
```

## Notes

- The script uses `torch.bfloat16` precision for efficiency
- Responses are truncated to 4096 tokens
- The CLS token (ID: 128003) is used to extract the final reward score
- All computations are done in inference mode (no gradients)