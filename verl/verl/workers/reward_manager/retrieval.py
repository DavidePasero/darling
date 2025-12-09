from collections import defaultdict
import torch
from verl import DataProto

from verl.retrieval.engine.retriever import Retriever
from verl.retrieval.engine.document_dataset import DocumentDataset


class RetrievalRewardManager:
    """
    Reward manager for retrieval tasks with FAISS-based quality rewards.
    Supports NDCG, Recall, Precision, and Hit metrics.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score,
        reward_fn_key="data_source",
        faiss_index_path=None,
        id_mapping_path=None,
        beir_dataset_path=None,
        qrels_file="qrels/train.tsv",
        quality_method="ndcg",
        k=10,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
        **kwargs
    ):
        """
        Initialize retrieval reward manager.

        Args:
            tokenizer: Tokenizer for decoding responses
            num_examine: Number of examples to print during training
            compute_score: Reward computation function (not used)
            reward_fn_key: Key to access data source in non_tensor_batch
            faiss_index_path: Path to FAISS index file
            id_mapping_path: Path to pickle file with ID mapping
            beir_dataset_path: Path to BEIR dataset directory
            qrels_file: Relative path to qrels file (default: "qrels/train.tsv")
            quality_method: Metric to use ('ndcg', 'recall', 'precision', 'hit')
            k: Top-k for retrieval
            embedding_model: HuggingFace model for embeddings
            device: Device for models
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.quality_method = quality_method
        self.k = k
        self.device = device

        if faiss_index_path is None:
            raise ValueError("faiss_index_path must be provided")
        if id_mapping_path is None:
            raise ValueError("id_mapping_path must be provided")
        if beir_dataset_path is None:
            raise ValueError("beir_dataset_path must be provided")

        print("=" * 80)
        print("Initializing RetrievalRewardManager")
        print("=" * 80)

        self.retriever = Retriever(
            faiss_index_path=faiss_index_path,
            embedding_model=embedding_model,
            id_mapping_path=id_mapping_path,
            device=device,
            verbose=True
        )

        from verl.retrieval.engine.document_dataset import BeirAdapter
        adapter = BeirAdapter(data_path=beir_dataset_path, split="train")
        self.doc_dataset = adapter.to_unified()

        if verbose := True:
            print(f"Loaded {len(self.doc_dataset.qrels)} queries with relevance labels")

        print(f"Quality reward: {quality_method}@{k}")
        print("=" * 80)
        print()

    def verify(self, data):
        """
        Extract and decode responses, then compute retrieval rewards.

        Args:
            data: DataProto object

        Returns:
            List of reward scores
        """
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        uid = data.non_tensor_batch["uid"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        scores, indices = self.retriever.search(responses_str, k=self.k, nprobe=64)
        retrieved_doc_ids = self.retriever.map_indices_to_ids(indices)

        rewards = self.doc_dataset.compute_rewards_batch(
            query_uids=uid,
            retrieved_doc_ids_batch=retrieved_doc_ids.tolist(),
            method=self.quality_method,
            k=self.k
        )

        return rewards

    def __call__(self, data: DataProto, return_dict=False):
        """
        Main entry point for reward computation.

        Args:
            data: DataProto object containing batch data
            return_dict: Whether to return dict with extra info

        Returns:
            Reward tensor or dict with reward_tensor and reward_extra_info
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, ['retrieval'] * len(data))

        scores = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            reward = scores[i]
            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(
                    data.batch["responses"][i][:length],
                    skip_special_tokens=True
                )
                prompt_str = self.tokenizer.decode(
                    data.batch["prompts"][i],
                    skip_special_tokens=True
                )
                uid_val = data.non_tensor_batch["uid"][i]

                print("=" * 80)
                print(f"Retrieval Reward Example - {data_source}")
                print(f"UID: {uid_val}")
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str[:200]}...")
                print(f"[{self.quality_method}@{self.k}] {reward:.4f}")
                print("=" * 80)
                print()

                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor