"""
BEIR Dataset for Query Rewriting with Retrieval Rewards

This dataset loads BEIR format queries and ensures they include:
- prompt: The original query text
- uid: The query_id for looking up relevance labels
"""

from typing import List, Union, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig, ListConfig
import torch
import json
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


class BeirRLDataset(Dataset):
    """
    BEIR format dataset for query rewriting RL training.

    Loads queries from BEIR queries.jsonl file.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        """
        Initialize BEIR RL dataset.

        Args:
            data_files: Path to queries.jsonl file
            tokenizer: Tokenizer for encoding queries
            config: Dataset configuration
            processor: Optional processor (not used for text-only)
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.max_prompt_length = config.get("max_prompt_length", 512)
        self.truncation = config.get("truncation", "error")

        if isinstance(data_files, str):
            queries_file = data_files
            qrels_file = None
        elif isinstance(data_files, (list, ListConfig)):
            queries_file = data_files[0]
            qrels_file = data_files[1] if len(data_files) > 1 else None
        else:
            raise ValueError(f"data_files must be str or list, got {type(data_files)}")

        print(f"Loading BEIR queries from: {queries_file}")

        # Load all queries
        all_queries = {}
        with open(queries_file, 'r') as f:
            for line in f:
                query_data = json.loads(line)
                all_queries[query_data["_id"]] = query_data["text"]

        # Filter to only queries with labels in qrels (if qrels file is provided)
        if qrels_file:
            print(f"Filtering queries using qrels: {qrels_file}")

            valid_query_ids = set()
            with open(qrels_file, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 1:
                        valid_query_ids.add(parts[0])

            self.queries = [
                {"query_id": qid, "text": text}
                for qid, text in all_queries.items()
                if qid in valid_query_ids
            ]
            print(f"Filtered to {len(self.queries)} queries with labels (from {len(all_queries)} total)")
        else:
            self.queries = [
                {"query_id": qid, "text": text}
                for qid, text in all_queries.items()
            ]
            print(f"Loaded {len(self.queries)} queries (no filtering)")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        """
        Get a single item.

        Returns dict with:
            - input_ids: Tokenized query
            - attention_mask: Attention mask
            - position_ids: Position IDs
            - uid: Query ID (for reward lookup)
            - query: Original query text
        """
        row = self.queries[idx]

        query_text = row["text"]
        query_id = row["query_id"]

        messages = [{"role": "user", "content": query_text}]

        raw_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        model_inputs = self.tokenizer(
            raw_prompt,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "uid": query_id,
            "query": query_text,
        }