import numpy as np


class RewardManager:
    def __init__(
        self,
        method: str = "ndcg",
        k: int = 10,
    ):
        """
        Manages reward calculation strategies for Query Rewriting RL.

        Args:
            method: Strategy to use ("recall", "ndcg", "cross_encoder").
            k: The depth of retrieval to consider (e.g., top-10).
            ce_model_name: HuggingFace model ID for the Cross-Encoder (only used if method="cross_encoder").
            device: Device to load the Cross-Encoder on.
        """
        self.method = method
        self.k = k

    def compute_rewards(
        self,
        indices_batch: np.ndarray,
        labels: list[int],
        id_mapping: list[int],
    ) -> list[float]:
        """
        Computes rewards for a batch of rewritten queries.
        All rewrites in this batch correspond to the same original query and truth labels.

        Args:
            indices_batch: (num_rewrites, k) Integer indices returned by FAISS.
            labels: List of relevant MS MARCO Passage IDs (Ground Truth).
            id_mapping: List where id_mapping[faiss_idx] = real_passage_id.
            original_query: The original user query (Required for 'cross_encoder' method).
            corpus_text_lookup: List where corpus_text_lookup[faiss_idx] = passage_text (Required for 'cross_encoder').

        Returns:
            List of float rewards, one for each rewritten query.
        """
        rewards = []
        gold_set = set(labels)

        for indices in indices_batch:
            # 1. Map FAISS Indices to Real PIDs (and handle invalid -1 indices)
            retrieved_pids = []
            valid_indices = []

            for idx in indices:
                if idx != -1 and idx < len(id_mapping):
                    retrieved_pids.append(id_mapping[idx])
                    valid_indices.append(idx)
                else:
                    retrieved_pids.append(-1)
                    valid_indices.append(-1)

            # 2. Calculate Reward based on configured method
            if self.method == "recall":
                rewards.append(self._calculate_recall(retrieved_pids, gold_set))

            elif self.method == "ndcg":
                rewards.append(self._calculate_ndcg(retrieved_pids, gold_set))

            else:
                print(f"⚠️ Unknown reward method '{self.method}', returning 0.0")
                rewards.append(0.0)

        return rewards

    def _calculate_recall(self, retrieved_pids: list[int], gold_set: set[int]) -> float:
        """
        Recall@K: Fraction of relevant documents retrieved.
        Pros: Simple, robust.
        Cons: Doesn't care about rank (rank 1 vs rank 10 are equal).
        """
        if not gold_set:
            return 0.0
        hits = sum(1 for pid in retrieved_pids if pid in gold_set)
        return float(hits) / len(gold_set)

    def _calculate_ndcg(self, retrieved_pids: list[int], gold_set: set[int]) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        Pros: Rank-aware (rewards putting relevant docs at the top). Standard in IR.
        Cons: Slightly more complex calculation.
        """
        if not gold_set:
            return 0.0

        dcg = 0.0
        idcg = 0.0

        # 1. Calculate DCG (Discounted Cumulative Gain) for retrieved list
        for i, pid in enumerate(retrieved_pids):
            if pid in gold_set:
                # Relevance is binary (1.0) for MS MARCO, decayed by log position
                dcg += 1.0 / np.log2(i + 2)  # i+2 because rank is 1-based and log(1)=0

        # 2. Calculate IDCG (Ideal DCG) - Perfect ranking
        # The ideal list has all 'n' relevant docs at the very top
        num_relevant = len(gold_set)
        # We can only fill up to 'len(retrieved)' slots (usually K)
        max_possible_matches = min(num_relevant, len(retrieved_pids))

        for i in range(max_possible_matches):
            idcg += 1.0 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0
