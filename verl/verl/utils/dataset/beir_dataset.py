from typing import Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from omegaconf import DictConfig
import os
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.prompt_extension import EXTENDER_REGISTRY


class BeirRLDataset(Dataset):
    """BEIR dataset for query rewriting RL training using UnifiedDataset."""

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        prompt_extender: Optional[str] = "rewrite",
        unified_dataset=None,
    ):
        if unified_dataset is None:
            raise ValueError("unified_dataset is required. Pass UnifiedDataset from RetrievalRewardManager.")
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.unified_dataset = unified_dataset

        self.max_prompt_length = config.get("max_prompt_length", 512)
        self.truncation = config.get("truncation", "error")

        self.prompt_extender = EXTENDER_REGISTRY.get(prompt_extender)

        print(f"Using prompt extender: {self.prompt_extender}")

        if self.prompt_extender is None:
            raise ValueError(f"Invalid prompt_extender: {prompt_extender}")
        self.prompt_extender = self.prompt_extender()

        self.queries = [
            {"query_id": qid, "text": text}
            for qid, text in unified_dataset.queries.items()
            if qid in unified_dataset.qrels
        ]
        
        print(f"BeirRLDataset initialized with {len(self.queries)} queries")
        print("*"*80)
        print("Queries:")
        for query in self.queries[:5]:
            print(f"  {query['query_id']}: {query['text']}")
        print()


    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        row = self.queries[idx]
        query_text = row["text"]
        query_id = row["query_id"]
        messages = self.prompt_extender.extend_prompt(query_text)

        raw_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        if os.getenv("DEBUG_LOG", "0") == "1":
            print("Query: {query_text}")
            print()
            print("Messages:")
            for message in messages:
                print(message)
            print()
            print("Raw prompt:")
            print(raw_prompt)
            print()

        model_inputs = self.tokenizer(
            raw_prompt,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} exceeds {self.max_prompt_length}")

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "uid": query_id,
            "query_id": query_id,
            "query": query_text,
        }