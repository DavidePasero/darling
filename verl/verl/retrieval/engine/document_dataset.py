import math
from abc import ABC, abstractmethod
from typing import Dict, List
from beir.datasets.data_loader import GenericDataLoader


class UnifiedDataset:
    def __init__(self, queries: Dict[str, str], corpus: Dict[str, str], qrels: Dict[str, List[str]]):
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels

        self.max_k = 1000
        self.idcg = self._precompute_idcg(self.max_k)

    @staticmethod
    def _precompute_idcg(max_k: int):
        idcg = [0.0] * (max_k + 1)
        s = 0.0
        for i in range(max_k):
            s += 1.0 / math.log2(i + 2)
            idcg[i + 1] = s
        return idcg

    def get_relevant_docs(self, query_id: str) -> List[str]:
        return self.qrels.get(query_id, [])

    def compute_ndcg(self, query_id: str, docs: List[str], k: int = 10) -> float:
        rel = set(self.get_relevant_docs(query_id))
        if not rel:
            return 0.0
        docs_k = docs[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(docs_k) if d in rel)
        ideal = min(len(rel), len(docs_k))
        return dcg / self.idcg[ideal] if ideal > 0 else 0.0

    def compute_recall(self, query_id: str, docs: List[str], k: int = 10) -> float:
        rel = set(self.get_relevant_docs(query_id))
        if not rel:
            return 0.0
        return len(set(docs[:k]) & rel) / len(rel)

    def compute_precision(self, query_id: str, docs: List[str], k: int = 10) -> float:
        docs_k = docs[:k]
        if not docs_k:
            return 0.0
        rel = set(self.get_relevant_docs(query_id))
        return sum(d in rel for d in docs_k) / len(docs_k)

    def compute_hit(self, query_id: str, docs: List[str], k: int = 10) -> float:
        rel = set(self.get_relevant_docs(query_id))
        return 1.0 if any(d in rel for d in docs[:k]) else 0.0

    def compute_rewards_batch(
        self,
        query_uids: List[str],
        retrieved_doc_ids_batch: List[List[str]],
        method: str = "ndcg",
        k: int = 10
    ) -> List[float]:
        """
        Compute rewards for a batch of queries.

        Args:
            query_uids: List of query IDs
            retrieved_doc_ids_batch: List of retrieved doc ID lists
            method: Metric to use ('ndcg', 'recall', 'precision', 'hit')
            k: Top-k cutoff

        Returns:
            List of reward scores
        """
        if method == "ndcg":
            compute_fn = self.compute_ndcg
        elif method == "recall":
            compute_fn = self.compute_recall
        elif method == "precision":
            compute_fn = self.compute_precision
        elif method == "hit":
            compute_fn = self.compute_hit
        else:
            raise ValueError(f"Unknown method: {method}")

        rewards = []
        for query_id, retrieved_docs in zip(query_uids, retrieved_doc_ids_batch):
            reward = compute_fn(query_id, retrieved_docs, k=k)
            rewards.append(reward)

        return rewards


class BaseDatasetAdapter(ABC):
    @abstractmethod
    def to_unified(self) -> UnifiedDataset:
        pass


class BeirAdapter(BaseDatasetAdapter):
    def __init__(self, data_path: str, split: str = "dev"):
        self.data_path = data_path
        self.split = split

    def to_unified(self) -> UnifiedDataset:
        corpus, queries, qrels = GenericDataLoader(self.data_path).load(split=self.split)

        corpus_texts = {doc_id: doc["text"] for doc_id, doc in corpus.items()}
        query_texts = {qid: text for qid, text in queries.items()}

        qrels_flat = {
            qid: [doc_id for doc_id, rel in docs.items() if rel > 0]
            for qid, docs in qrels.items()
        }

        return UnifiedDataset(query_texts, corpus_texts, qrels_flat)


if __name__ == "__main__":
    adapter = BeirAdapter("datasets/msmarco")
    dataset = adapter.to_unified()

    qid = "1102434"
    docs = ["D123", "D19", "D202"]

    print(dataset.compute_ndcg(qid, docs))
    print(dataset.compute_recall(qid, docs))
