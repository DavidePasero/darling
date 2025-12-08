"""
RetrievalRewardManager for query rewriting with FAISS-based retrieval rewards.

Integrates existing FAISS infrastructure with DARLING-style diversity rewards.
"""

from collections import defaultdict
import os
import pickle
import torch

from verl.retrieval.reward.reward_manager import RewardManager
from verl.retrieval.engine.vectordatabase import FaissDB
from verl import DataProto


class RetrievalRewardManager:
    """
    Reward manager for retrieval tasks combining quality (NDCG/Recall) + diversity.

    Follows the same interface as DiversityRewardManager.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score,
        reward_fn_key="data_source",
        # FAISS infrastructure paths
        faiss_index_path: str = None,
        id_mapping_path: str = None,
        labels_path: str = None,
        # Quality reward configuration
        quality_method: str = 'ndcg',  # 'ndcg', 'recall', 'hit'
        k: int = 10,
        # Diversity configuration
        diversity_enabled: bool = True,
        diversity_method: str = 'local',  # 'local' or 'vllm'
        deberta_model: str = 'microsoft/deberta-v3-large',
        vllm_hostname: str = None,
        num_vllm_servers: int = 8,
        threshold: float = 0.5,
        device: str = 'cuda',
        # Embedding model for FAISS
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        **kwargs
    ):
        """
        Initialize RetrievalRewardManager.

        Args:
            tokenizer: Tokenizer for decoding responses
            num_examine: Number of examples to print during training
            compute_score: Reward computation function (retrieval_reward)
            reward_fn_key: Key to access data source in non_tensor_batch
            faiss_index_path: Path to FAISS index file
            id_mapping_path: Path to pickle file with ID mapping
            labels_path: Path to pickle file with {uid: [relevant_doc_ids]}
            quality_method: 'ndcg', 'recall', or 'hit'
            k: Top-k for retrieval
            diversity_enabled: Whether to compute diversity rewards
            diversity_method: 'local' (DeBERTa) or 'vllm' (server)
            deberta_model: Model name for local diversity classifier
            vllm_hostname: Hostname for VLLM servers (if method='vllm')
            num_vllm_servers: Number of VLLM servers
            threshold: Similarity threshold for partitioning
            device: Device for local models
            embedding_model: HuggingFace model for FAISS embeddings
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

        # Validate required paths
        if faiss_index_path is None:
            raise ValueError("faiss_index_path must be provided")
        if id_mapping_path is None:
            raise ValueError("id_mapping_path must be provided")
        if labels_path is None:
            raise ValueError("labels_path must be provided")

        print(f"\n{'='*80}")
        print(f"Initializing RetrievalRewardManager")
        print(f"{'='*80}")

        # Load FAISS infrastructure
        print(f"Loading FAISS infrastructure...")
        self._load_faiss_infrastructure(
            faiss_index_path=faiss_index_path,
            id_mapping_path=id_mapping_path,
            labels_path=labels_path,
            embedding_model=embedding_model,
            device=device
        )

        self.reward_manager = RewardManager(method=quality_method, k=k)
        print(f"Quality reward: {quality_method}@{k}")

        # Setup VLLM servers if needed
        vllm_servers = None
        if diversity_enabled and diversity_method == 'vllm':
            if vllm_hostname is None:
                vllm_hostname = os.environ.get('VLLM_SERVER_HOSTNAME', 'localhost')

            vllm_servers = [
                {
                    "url": f"http://{vllm_hostname}:{8000+i}",
                    "model": f"similarity_gpu_{i}"
                }
                for i in range(num_vllm_servers)
            ]
            print(f"Diversity method: VLLM ({num_vllm_servers} servers at {vllm_hostname})")
        else:
            print(f"Diversity method: {diversity_method}")

        # Store all reward kwargs
        self.reward_kwargs = {
            'faiss_db': self.faiss_db,
            'reward_manager': self.reward_manager,
            'id_mapping': self.id_mapping,
            'labels_dict': self.labels_dict,
            'diversity_enabled': diversity_enabled,
            'diversity_method': diversity_method,
            'deberta_model': deberta_model,
            'vllm_servers': vllm_servers,
            'threshold': threshold,
            'device': device,
        }

        print(f"{'='*80}\n")

    def _load_faiss_infrastructure(
        self,
        faiss_index_path: str,
        id_mapping_path: str,
        labels_path: str,
        embedding_model: str,
        device: str
    ):
        """Load FAISS database, ID mapping, and labels."""

        # Load FAISS database
        self.faiss_db = FaissDB(model_name=embedding_model, device=device)
        self.faiss_db.load_index(faiss_index_path)
        print(f"✓ Loaded FAISS index: {faiss_index_path}")

        # Load ID mapping
        with open(id_mapping_path, 'rb') as f:
            self.id_mapping = pickle.load(f)
        print(f"✓ Loaded ID mapping: {len(self.id_mapping)} documents")

        # Load labels
        with open(labels_path, 'rb') as f:
            self.labels_dict = pickle.load(f)
        print(f"✓ Loaded labels: {len(self.labels_dict)} queries")

    def verify(self, data):
        """
        Extract and decode responses, then compute rewards.

        Follows DiversityRewardManager.verify() pattern.
        """
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        uid = data.non_tensor_batch["uid"]
        log_probs = data.batch["old_log_probs"].sum(dim=-1) if "old_log_probs" in data.batch else None

        # Normalize log_probs by length
        if log_probs is not None:
            log_probs = log_probs / attention_mask.sum(dim=-1)

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # Decode responses
        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        # Extract ground truths (not used in retrieval - labels come from labels_dict)
        if data.non_tensor_batch.get("reward_model") is not None:
            ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        else:
            ground_truths = [None] * len(data)

        # Extract metadata
        data_sources = data.non_tensor_batch[self.reward_fn_key] if self.reward_fn_key in data.non_tensor_batch else ['retrieval'] * len(data)
        extras = data.non_tensor_batch.get("extra_info", None)
        correctness = data.non_tensor_batch.get("correctness", None)

        # Decode prompts
        prompts_str = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in prompt_ids]

        # Compute retrieval rewards
        scores = self.compute_score(
            data_source=data_sources,
            solution_str=responses_str,
            ground_truth=ground_truths,
            extra_info=extras,
            prompts=prompts_str,
            uid=uid,
            log_probs=log_probs,
            correctness=correctness,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        """
        Main entry point - compute rewards and create reward tensor.

        Follows DiversityRewardManager.__call__() pattern.
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key] if self.reward_fn_key in data.non_tensor_batch else ['retrieval'] * len(data)
        reward_extra_info = data.non_tensor_batch.get("extra_info", None)

        # Compute scores
        scores = self.verify(data)
        rewards = []
        already_printed = {}

        # Place rewards in tensor
        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            # Print examples
            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                uid_val = data.non_tensor_batch["uid"][i]
                print(f"\n{'='*80}")
                print(f"[Retrieval Reward Example - {data_source}]")
                print(f"UID: {uid_val}")
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[rewrite] {response_str[:200]}...")
                print(f"[score] {scores[i]:.4f}")
                print(f"{'='*80}\n")
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        # Store accuracy for logging
        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor