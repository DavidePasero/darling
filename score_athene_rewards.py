#!/usr/bin/env python3
"""
Script to compute Athene reward model scores for prompt-response pairs.
Usage:
    python score_athene_rewards.py --input data.json --output rewards.json

Input JSON format:
    [
        {
            "prompt": "What is the capital of France?",
            "generation": "The capital of France is Paris."
        },
        ...
    ]
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, LlamaModel, LlamaPreTrainedModel
from torch import nn
from typing import List, Dict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AtheneForSequenceClassification(LlamaPreTrainedModel):
    """Athene reward model for sequence classification."""

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.CLS_ID = 128003
        self.post_init()

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)

        bs = int(input_ids.shape[0])

        for i in range(bs):
            c_inds = (input_ids[i] == self.CLS_ID).nonzero()
            if len(c_inds) == 0:
                # CLS token not found, use the last token instead
                c_ind = input_ids[i].shape[0] - 1
                logger.warning(f"CLS token (ID={self.CLS_ID}) not found in input {i}. Using last token at position {c_ind}")
            else:
                c_ind = c_inds[-1].item()
            scores.append(rewards[i, c_ind])
        scores = torch.stack(scores)
        return {"scores": scores}


def load_reward_model(model_path: str, device: str = "auto"):
    """Load the Athene reward model and tokenizer."""
    logger.info(f"Loading Athene reward model from {model_path}")

    import os

    # Check if it's a local path
    is_local = os.path.exists(model_path) or os.path.isabs(model_path)

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device,
        "attn_implementation": "flash_attention_2",
    }

    if is_local and not os.path.exists(model_path):
        logger.error(f"Local path specified but does not exist: {model_path}")
        logger.info("Available public Athene models:")
        logger.info("  - Nexusflow/Athene-RM-8B")
        logger.info("  - Nexusflow/Athene-RM-70B")
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if is_local:
        logger.info(f"Loading from local path")
        load_kwargs["local_files_only"] = True

    try:
        model = AtheneForSequenceClassification.from_pretrained(
            model_path,
            **load_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("\nIf using a local path, ensure the model files exist.")
        logger.info("If using HuggingFace, try one of these public models:")
        logger.info("  - Nexusflow/Athene-RM-8B")
        logger.info("  - Nexusflow/Athene-RM-70B")
        raise

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure CLS token is properly set
    if tokenizer.cls_token is None:
        logger.info("Setting CLS token to <|reserved_special_token_0|>")
        tokenizer.cls_token = "<|reserved_special_token_0|>"
        tokenizer.cls_token_id = 128003

    logger.info(f"CLS token: {tokenizer.cls_token}, ID: {tokenizer.cls_token_id}")
    logger.info("Model loaded successfully")
    return model, tokenizer


def compute_rewards(
    model,
    tokenizer,
    prompt_response_pairs: List[Dict[str, str]],
    batch_size: int = 8
) -> List[Dict[str, any]]:
    """
    Compute rewards for prompt-response pairs.

    Args:
        model: Athene reward model
        tokenizer: Model tokenizer
        prompt_response_pairs: List of dicts with 'prompt' and 'generation' keys
        batch_size: Batch size for processing

    Returns:
        List of dicts with prompt, generation, and reward
    """
    results = []

    # Process in batches
    for i in tqdm(range(0, len(prompt_response_pairs), batch_size), desc="Computing rewards"):
        batch = prompt_response_pairs[i:i + batch_size]

        # Format as chat conversations
        convs = [
            [
                {"content": item["prompt"], "role": "user"},
                {"content": item["generation"], "role": "assistant"},
            ]
            for item in batch
        ]

        # Apply chat template and add CLS token
        formatted = tokenizer.apply_chat_template(convs, tokenize=False)
        formatted = [f + tokenizer.cls_token for f in formatted]

        # Tokenize
        inputs = tokenizer(
            formatted,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=4096,
        )

        # Move to device
        inputs = {k: v.to(model.get_device()) for k, v in inputs.items()}

        # Compute rewards
        with torch.no_grad():
            outputs = model(**inputs)
            rewards = outputs["scores"].cpu().float().tolist()

        # Store results
        for item, reward in zip(batch, rewards):
            results.append({
                "prompt": item["prompt"],
                "generation": item["generation"],
                "reward": reward
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Score generations with Athene reward model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Nexusflow/Athene-RM-8B",
        help="Path to Athene reward model (local path or HuggingFace model ID like 'Nexusflow/Athene-RM-8B')"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with prompt-response pairs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rewards_output.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="rewards.log",
        help="Log file for detailed output"
    )

    args = parser.parse_args()

    # Setup logging to file
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Load input data
    logger.info(f"Loading input from {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} prompt-response pairs")

    # Load model
    model, tokenizer = load_reward_model(args.model_path, args.device)

    # Compute rewards
    logger.info("Computing rewards...")
    results = compute_rewards(model, tokenizer, data, args.batch_size)

    # Save results
    logger.info(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary statistics
    rewards = [r["reward"] for r in results]
    logger.info("\n=== Reward Statistics ===")
    logger.info(f"Number of samples: {len(rewards)}")
    logger.info(f"Mean reward: {sum(rewards) / len(rewards):.4f}")
    logger.info(f"Min reward: {min(rewards):.4f}")
    logger.info(f"Max reward: {max(rewards):.4f}")
    logger.info(f"Median reward: {sorted(rewards)[len(rewards)//2]:.4f}")

    print("\n=== Reward Statistics ===")
    print(f"Number of samples: {len(rewards)}")
    print(f"Mean reward: {sum(rewards) / len(rewards):.4f}")
    print(f"Min reward: {min(rewards):.4f}")
    print(f"Max reward: {max(rewards):.4f}")
    print(f"Median reward: {sorted(rewards)[len(rewards)//2]:.4f}")

    # Print some examples
    print("\n=== Sample Results ===")
    for i, result in enumerate(results[:5]):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {result['prompt'][:100]}...")
        print(f"Generation: {result['generation'][:100]}...")
        print(f"Reward: {result['reward']:.4f}")

    logger.info(f"Done! Results saved to {args.output}")


if __name__ == "__main__":
    main()