# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

<<<<<<< Updated upstream
DARLING implements the paper "Jointly Reinforcing Diversity and Quality of Language Model Generations". This repository contains:
- A modified version of **verl** (Volcano Engine Reinforcement Learning for LLMs) - a production-ready RL training library
- Evaluation harnesses for creative writing, math tasks, and novelty benchmarks
- Training scripts for both verifiable (math) and non-verifiable (wildchat) tasks
=======
DARLING is the official implementation of "Jointly Reinforcing Diversity and Quality of Language Model Generations". The project is built on top of **verl** (Volcano Engine Reinforcement Learning for LLMs), a flexible and efficient RL training framework for large language models.

This repository contains:
- **darling/**: Main implementation of the DARLING algorithm
- **verl/**: Core RL training framework (verl library)
- **evals/**: Evaluation benchmarks (novelty-bench, eqbench, math tasks, creative writing)
>>>>>>> Stashed changes

## Environment Setup

### Creating the Environment

```bash
conda create -n verlenv python=3.10
conda activate verlenv
```

### Installing Dependencies

```bash
<<<<<<< Updated upstream
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
=======
# Install PyTorch with CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install verl and dependencies
cd verl
pip install -e ./

# Install FSDP-only setup (without Megatron)
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# Install vLLM
pip install vllm==0.8.3

# Install flash-attention
pip3 install flash-attn --no-build-isolation
```

### Setting Up Wandb (Optional)

```bash
export WANDB_API_KEY=<your_api_key>
```

## Core Architecture

### verl Framework Structure

The verl framework follows a modular architecture with these key components:

- **Workers** (`verl/workers/`): Distributed computation units
  - `actor/`: Policy model training
  - `rollout/`: Generation using vLLM/SGLang
  - `critic/`: Value function training (for PPO)
  - `reward_model/`: Reward model inference
  - `reward_manager/`: Reward computation coordination

- **Trainer** (`verl/trainer/`):
  - `main_ppo.py`: Main entry point for PPO/GRPO training
  - `main_generation.py`: Generation-only entry point
  - `main_eval.py`: Evaluation entry point
  - `ppo/`: PPO algorithm implementation
  - `config/`: Hydra configuration files

- **Models** (`verl/models/`):
  - `llama/`, `qwen2/`: Model-specific implementations
  - `transformers/`: HuggingFace Transformers backend
  - `mcore/`: Megatron-LM backend

- **Single Controller** (`verl/single_controller/`):
  - `ray/`: Ray-based distributed training coordination
  - Implements the hybrid-controller programming model

- **Utils** (`verl/utils/`):
  - `dataset/`: Data loading and processing
  - `reward_score/`: Reward function implementations
  - `checkpoint/`: Model checkpointing utilities
  - `logger/`: Experiment tracking (wandb, tensorboard, etc.)

### Data Flow

1. **Prompts** → **Rollout Worker** (vLLM/SGLang) → **Responses**
2. **Responses** → **Reward Manager** → **Rewards**
3. **Trajectories + Rewards** → **Actor/Critic Workers** → **Updated Models**

The framework uses `DataProto` (defined in `verl/protocol.py`) as the core data structure for passing data between workers.

## Running DARLING

### 1. Serve the Partition Classifier

First, serve the partition classifier model on 8 GPUs:

```bash
bash serve_classifier.sh <PATH_TO_CLASSIFIER_HF>
```

This script launches 8 vLLM instances (one per GPU) on ports 8000-8007.

### 2. Set Hostname Environment Variable

```bash
export VLLM_SERVER_HOSTNAME=<your_hostname>
```

Alternatively, manually edit the hostname in `verl/verl/utils/reward_score/partition_reward_vllm_serve.py`.

### 3. Launch Training

**For math tasks (verifiable):**
```bash
cd verl
# Edit math_scripts/darling.batch to set your SCRATCH_DIR and paths
# Submit via SLURM or run directly
```

**For non-verifiable tasks (e.g., WildChat):**
```bash
cd verl
# Edit wildchat_scripts/darling.batch to set your data paths
# Submit via SLURM or run directly
```

Both scripts use the main PPO trainer with:
- `algorithm.adv_estimator=grpo` (GRPO algorithm)
- `reward_model.reward_manager=diversity` (DARLING diversity reward)
- Custom diversity function via `+reward_model.custom_diversity_function.path=...`

## Common Commands

### Training

**PPO/GRPO Training:**
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=<train_data.parquet> \
    data.val_files=<val_data.parquet> \
    actor_rollout_ref.model.path=<model_path> \
    [additional_config_overrides...]
```

**Supervised Fine-tuning:**
```bash
# See examples/sft/ for SFT examples
python3 -m verl.trainer.fsdp_sft_trainer [config_overrides...]
```

### Generation Only

```bash
python3 -m verl.trainer.main_generation \
    data.val_files=<data.parquet> \
    actor_rollout_ref.model.path=<model_path> \
    [additional_config_overrides...]
```

### Testing

**Run CPU tests:**
```bash
pytest -v tests/**/test_*_on_cpu.py
```

**Run all unit tests:**
```bash
python -m pytest -m unit
```

**Run specific test categories:**
- GPU unit tests: Tests without `on_cpu.py` suffix
- Distributed tests: `tests/special_distributed/`
- End-to-end tests: `tests/special_e2e/`
- Sanity tests: `tests/special_sanity/`

### Linting and Formatting

```bash
# Install pre-commit hooks
pre-commit install

# Run linting (uses ruff)
ruff check .

# Auto-format code
ruff format .
```

## Configuration System

The project uses Hydra for configuration management. Config files are in `verl/trainer/config/`.

### Key Configuration Groups

- **algorithm**: PPO, GRPO, REINFORCE++, RLOO, etc.
- **data**: Dataset paths, batch sizes, sequence lengths
- **actor_rollout_ref**: Actor model, rollout engine (vLLM/SGLang), reference model
- **critic**: Critic model (for PPO with GAE)
- **reward_model**: Reward model or reward manager configuration
- **trainer**: Training hyperparameters, logging, checkpointing

### Override Patterns

```bash
# Use dot notation for nested configs
python3 -m verl.trainer.main_ppo \
    trainer.total_epochs=10 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.name=vllm \
    data.train_batch_size=256
```

### Adding New Config Fields

Use `+` prefix to add new fields not in the base schema:
```bash
+reward_model.custom_diversity_function.path=/path/to/reward.py \
+reward_model.custom_diversity_function.name=partition
```

## Development Patterns

### Adding New Reward Functions

1. Create a Python file with your reward function
2. The function should accept prompts and responses
3. Pass it via config:
```bash
+custom_reward_function.path=/path/to/reward.py \
+custom_reward_function.name=my_reward_fn
```

### Backend Selection

**Training backends:**
- FSDP (default, recommended): `actor_rollout_ref.actor.strategy=fsdp`
- FSDP2 (newer): `actor_rollout_ref.actor.strategy=fsdp2`
- Megatron-LM: For large-scale MoE models (DeepSeek-671B, Qwen3-236B)

**Inference backends:**
- vLLM: `actor_rollout_ref.rollout.name=vllm`
- SGLang: `actor_rollout_ref.rollout.name=sglang`

### Memory Optimization

Common memory-saving techniques:
```bash
# Enable gradient checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing=True

# Offload parameters/optimizer
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# Use remove padding for variable-length sequences
actor_rollout_ref.model.use_remove_padding=True

# Adjust GPU memory utilization for rollout
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
```

### Distributed Training

The framework uses Ray for distributed coordination:
- Set `trainer.n_gpus_per_node` and `trainer.nnodes`
- For SLURM clusters, use the batch scripts in `math_scripts/` or `wildchat_scripts/` as templates
- Ray head node is automatically configured in the batch scripts

## Evaluation

Evaluation scripts are in the `evals/` directory:

- **Math evaluation**: `evals/math_evaluation/`
- **Novelty-bench**: `evals/novelty-bench/`
- **Creative writing**: `evals/creative-writing-bench/`

## Important Notes

- **vLLM version**: Use vLLM 0.8.x. Avoid 0.7.x (contains bugs causing OOMs).
- **Data format**: Training data should be in Parquet format with a `prompt` key (configurable via `data.prompt_key`).
- **Checkpoint compatibility**: Use `verl/tools/` for checkpoint conversion between formats.
- **Multi-turn support**: See `examples/sglang_multiturn/` for multi-turn rollout examples.
- **Sequence parallelism**: Supported via DeepSpeed Ulysses (`actor_rollout_ref.rollout.use_sp=True`).
>>>>>>> Stashed changes
