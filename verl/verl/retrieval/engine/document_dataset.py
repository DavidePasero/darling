import math
from abc import ABC, abstractmethod
from typing import Dict, List
from beir.datasets.data_loader import GenericDataLoader
from functools import partial
import numpy as np
import requests


class UnifiedDataset:
    def __init__(self, queries: Dict[str, str], corpus: Dict[str, str], qrels: Dict[str, List[str]]):
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels

        self.max_k = 1000
        # Convert to numpy array for faster indexing in batch operations
        self.idcg = np.array(self._precompute_idcg(self.max_k))

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

    def compute_ndcg_batch(
        self, query_uids: List[str], retrieved_doc_ids_batch: List[List[str]], k: int = 10
    ) -> List[float]:
        rewards = []
        # Pre-compute discounts for the maximum possible length k
        discounts = 1.0 / np.log2(np.arange(k) + 2)

        for query_id, docs in zip(query_uids, retrieved_doc_ids_batch):
            rel_docs = set(self.get_relevant_docs(query_id))
            if not rel_docs:
                rewards.append(0.0)
                continue

            docs_k = docs[:k]
            # Create binary relevance vector
            relevance = np.array([1.0 if d in rel_docs else 0.0 for d in docs_k])

            # Compute DCG
            # Handle case where docs_k length is less than k
            current_discounts = discounts[: len(relevance)]
            dcg = np.sum(relevance * current_discounts)

            # Compute IDCG
            ideal_count = min(len(rel_docs), len(docs_k))
            idcg_val = self.idcg[ideal_count]

            rewards.append(dcg / idcg_val if idcg_val > 0 else 0.0)

        return rewards

    def compute_recall_batch(
        self, query_uids: List[str], retrieved_doc_ids_batch: List[List[str]], k: int = 10
    ) -> List[float]:
        rewards = []
        for query_id, docs in zip(query_uids, retrieved_doc_ids_batch):
            rel_docs = set(self.get_relevant_docs(query_id))
            if not rel_docs:
                rewards.append(0.0)
                continue

            # Intersection count
            hits = sum(1 for d in docs[:k] if d in rel_docs)
            rewards.append(hits / len(rel_docs))
        return rewards

    def compute_precision_batch(
        self, query_uids: List[str], retrieved_doc_ids_batch: List[List[str]], k: int = 10
    ) -> List[float]:
        rewards = []
        for query_id, docs in zip(query_uids, retrieved_doc_ids_batch):
            docs_k = docs[:k]
            if not docs_k:
                rewards.append(0.0)
                continue

            rel_docs = set(self.get_relevant_docs(query_id))
            hits = sum(1 for d in docs_k if d in rel_docs)
            rewards.append(hits / len(docs_k))
        return rewards

    def compute_hit_batch(
        self, query_uids: List[str], retrieved_doc_ids_batch: List[List[str]], k: int = 10
    ) -> List[float]:
        rewards = []
        for query_id, docs in zip(query_uids, retrieved_doc_ids_batch):
            rel_docs = set(self.get_relevant_docs(query_id))
            # 1.0 if any relevant doc in top k, else 0.0
            rewards.append(1.0 if any(d in rel_docs for d in docs[:k]) else 0.0)
        return rewards

    def compute_reranker_score_batch(
        self,
        query_uids: List[str],
        retrieved_doc_ids_batch: List[List[str]],
        k: int = 10,
        reranker_url: str = "http://localhost:8000/v1/score",
    ) -> List[float]:
        # Flatten pairs for batch inference
        all_pairs = []
        doc_counts = []

        for query_id, docs in zip(query_uids, retrieved_doc_ids_batch):
            docs_k = docs[:k]
            query_text = self.queries[query_id]
            pairs = [[query_text, self.corpus[doc]] for doc in docs_k]
            all_pairs.extend(pairs)
            doc_counts.append(len(pairs))

        if not all_pairs:
            return [0.0] * len(query_uids)

        # Prepare payload for FastAPI (split into parallel lists)
        text_1 = [p[0] for p in all_pairs]
        text_2 = [p[1] for p in all_pairs]

        payload = {"text_1": text_1, "text_2": text_2}

        # Single batch call to the remote reranker backend
        try:
            response = requests.post(reranker_url, json=payload)
            response.raise_for_status()

            # Parse response: {"data": [{"score": 0.1, "index": 0}, ...]}
            data = response.json().get("data", [])
            all_scores = [item["score"] for item in data]

            # Safety check: ensure we got back the same number of scores
            if len(all_scores) != len(all_pairs):
                print(f"Warning: Reranker returned {len(all_scores)} scores for {len(all_pairs)} pairs.")
                # Pad with zeros if necessary to prevent crash
                all_scores.extend([0.0] * (len(all_pairs) - len(all_scores)))

        except Exception as e:
            print(f"Error calling reranker API: {e}")
            all_scores = [0.0] * len(all_pairs)

        # Split scores back to per-query structure and compute discounted sum
        rewards = []
        start_idx = 0
        discounts_cache = 1.0 / np.log2(np.arange(k) + 2)

        for count in doc_counts:
            if count == 0:
                rewards.append(0.0)
                continue

            # Convert to numpy array for vector operations
            scores = np.array(all_scores[start_idx : start_idx + count])
            current_discounts = discounts_cache[:count]

            # Apply discounts: score / log2(rank + 2)
            discounted_scores = scores * current_discounts
            rewards.append(float(np.sum(discounted_scores)))

            start_idx += count

        return rewards

    def compute_rewards_batch(
        self,
        query_uids: List[str],
        retrieved_doc_ids_batch: List[List[str]],
        method: str = "ndcg",
        k: int = 10,
        reranker_url: str = "http://localhost:8000/v1/score",
    ) -> List[float]:
        """
        Compute rewards for a batch of queries using batched operations.
        """
        print("RETRIEVED DOCS SHAPE:", np.array(retrieved_doc_ids_batch).shape)
        if method == "ndcg":
            return self.compute_ndcg_batch(query_uids, retrieved_doc_ids_batch, k=k)
        elif method == "recall":
            return self.compute_recall_batch(query_uids, retrieved_doc_ids_batch, k=k)
        elif method == "precision":
            return self.compute_precision_batch(query_uids, retrieved_doc_ids_batch, k=k)
        elif method == "hit":
            return self.compute_hit_batch(query_uids, retrieved_doc_ids_batch, k=k)
        elif method == "reranker":
            return self.compute_reranker_score_batch(
                query_uids, retrieved_doc_ids_batch, k=k, reranker_url=reranker_url
            )
        else:
            raise ValueError(f"Unknown method: {method}")


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

        qrels_flat = {qid: [doc_id for doc_id, rel in docs.items() if rel > 0] for qid, docs in qrels.items()}

        return UnifiedDataset(query_texts, corpus_texts, qrels_flat)


if __name__ == "__main__":
    adapter = BeirAdapter("datasets/msmarco")
    dataset = adapter.to_unified()

    qid = "1102434"
    docs = ["D123", "D19", "D202"]

    print(dataset.compute_ndcg(qid, docs))
    print(dataset.compute_recall(qid, docs))
