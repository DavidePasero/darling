# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DARLING implements the paper "Jointly Reinforcing Diversity and Quality of Language Model Generations". This repository contains:
- A modified version of **verl** (Volcano Engine Reinforcement Learning for LLMs) - a production-ready RL training library
- Evaluation harnesses for creative writing, math tasks, and novelty benchmarks
- Training scripts for both verifiable (math) and non-verifiable (wildchat) tasks

## Environment Setup

### Creating the Environment

```bash
conda create -n verlenv python=3.10
conda activate verlenv
```

### Installing Dependencies

```bash
# Install PyTorch (tested on CUDA 12.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install verl
cd verl
pip install -e ./

# Install vLLM and other dependencies
# Set USE_MEGATRON=0 if you only need FSDP (no Megatron support)
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install vllm==0.8.3
pip3 install flash-attn --no-build-isolation
```

### Environment Variables

```bash
# For W&B tracking (optional)
export WANDB_API_KEY=<your_api_key>

# For DARLING-specific classifier server
export VLLM_SERVER_HOSTNAME=<your_hostname>
```

## Training

### DARLING Training

DARLING requires a partition classifier to be served before training:

```bash
# Start classifier server
bash serve_classifier.sh <PATH_TO_CLASSIFIER_HF>

# Update hostname in verl/verl/utils/reward_score/partition_reward_vllm_serve.py
# OR set: export VLLM_SERVER_HOSTNAME=<your_hostname>

# Run DARLING on math tasks (Qwen-4B-Base)
# Training script: verl/math_scripts/darling.batch

# Run DARLING on wildchat (Llama-3.1-8B-Instruct)
# Training script: verl/wildchat_scripts/darling.batch
```

### Training Script Locations

- **Verifiable tasks (math)**: `verl/math_scripts/`
- **Non-verifiable tasks (wildchat)**: `verl/wildchat_scripts/`
- **Example trainers**: `verl/examples/` (PPO, GRPO, GMPO, etc.)
- **Recipe implementations**: `verl/recipe/` (DAPO, SPIN, SPPO, etc.)

## Evaluation

All evaluation harnesses are in the `evals/` directory:

### Math Evaluation

```bash
cd evals/math_evaluation

# Install dependencies
cd latex2sympy && pip install -e . && cd ..
pip install -r requirements.txt
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3

# Run evaluation
export CUDA_VISIBLE_DEVICES="0"
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
```

### Creative Writing Benchmark

```bash
cd evals/creative-writing-bench

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict')"

# Configure API keys
cp .env.example .env
# Edit .env with your API keys

# Run benchmark
python3 creative_writing_bench.py \
    --test-model "your-model-provider/your-model-name" \
    --judge-model "anthropic/claude-3.7-sonnet" \
    --runs-file "creative_bench_runs.json" \
    --creative-prompts-file "data/creative_writing_prompts_v3.json" \
    --run-id "my_model_run_1" \
    --threads 500 \
    --iterations 3
```

### Novelty Benchmark

Located in `evals/novelty-bench/` - used to evaluate model diversity/novelty.

## Architecture Overview

### verl Framework Structure

verl uses a **hybrid-controller programming model** that enables flexible RL dataflow execution:

- **Training backends**: FSDP, FSDP2, Megatron-LM
- **Inference backends**: vLLM (0.8.3+), SGLang, HF Transformers
- **Worker abstraction**: Decouples training and generation across different GPU sets
- **Ray-based orchestration**: Coordinates distributed workers

Key components:
- `verl/trainer/` - Main training loops (PPO, SFT, evaluation, generation)
- `verl/single_controller/` - Worker and worker group abstractions for Ray
- `verl/models/` - Model implementations (Llama, Qwen2, transformers, Megatron)
- `verl/workers/` - Worker implementations (FSDP, Megatron, vLLM, SGLang)
- `verl/utils/` - Utilities including reward functions, datasets, etc.
- `verl/third_party/` - Custom integrations with vLLM and SGLang

### DARLING Architecture

The DARLING method partitions the generation space into diverse regions and applies different reward strategies per region:

1. **Partition classifier** - Served via vLLM, assigns outputs to diversity partitions
2. **Partition-aware reward** - Located in `verl/verl/utils/reward_score/partition_reward_vllm_serve.py`
3. **Training scripts** - SLURM-based batch scripts in `verl/math_scripts/` and `verl/wildchat_scripts/`

## Common Development Patterns

### Running Single Tests

Most training examples follow this pattern:
```bash
cd verl/examples/<trainer_type>
bash run_<model>_<task>.sh
```

### Configuration Files

Training is configured via YAML files:
- Example configs: `verl/examples/*/config/*.yaml`
- Recipe configs: `verl/recipe/*/config/*.yaml`
- Trainer configs: `verl/verl/trainer/config/*.yaml`

### Data Preprocessing

Data preprocessing scripts are in `verl/examples/data_preprocess/`:
- `gsm8k.py` - GSM8K math dataset
- `gsm8k_multiturn_w_tool.py` - Multi-turn with tool usage
- `geo3k.py` - Geometry problems
- `preprocess_search_r1_dataset.py` - Search-augmented data

## Multi-Turn and Tool Usage

verl supports multi-turn conversations with tool calling:
- Configuration: `verl/examples/sglang_multiturn/config/`
- Tool definitions: `verl/verl/tools/` (base_tool.py, gsm8k_tool.py)
- Tool configs: YAML files defining available tools per task
- Interaction configs: Define multi-turn interaction patterns

## Backend-Specific Notes

### FSDP vs Megatron

- **FSDP/FSDP2**: Simpler setup, good for most use cases, supports models up to 70B+
- **Megatron-LM**: Required for very large models (671B DeepSeek, 235B Qwen3), enables expert parallelism for MoE models

Configuration examples demonstrate both backends (e.g., `run_qwen2-7b_math.sh` vs `run_qwen2-7b_math_megatron.sh`)

### vLLM Version

The codebase uses vLLM 0.8.3. Avoid vLLM 0.7.x due to bugs. See `verl/docs/README_vllm0.8.md` for details.

### Ray Cluster Setup

Training scripts (`.batch` files) handle Ray cluster initialization on SLURM:
1. Start head node with Ray
2. Connect worker nodes to head
3. Launch training via Ray actors

## Key Files to Understand

- `verl/verl/trainer/main_ppo.py` - Main PPO training loop
- `verl/verl/trainer/fsdp_sft_trainer.py` - Supervised fine-tuning
- `verl/verl/workers/fsdp_workers.py` - FSDP actor/critic/rollout workers
- `verl/verl/workers/rollout/vllm_rollout.py` - vLLM-based generation
- `verl/verl/utils/reward_function/` - Reward function implementations

## Testing

Test suites are organized in `verl/tests/`:
- `verl/tests/trainer/` - Trainer tests
- `verl/tests/workers/` - Worker tests
- `verl/tests/special_e2e/` - End-to-end tests
- `verl/tests/special_standalone/` - Standalone component tests

Run tests from the `verl/` directory.