import math
from abc import ABC, abstractmethod
from typing import Dict, List
from beir.datasets.data_loader import GenericDataLoader
from functools import partial
import numpy as np
import requests


class UnifiedDataset:
    def __init__(
        self,
        queries: Dict[str, str],        # query_id -> query_text
        corpus: Dict[str, str],         # doc_id   -> doc_text
        qrels: Dict[str, List[str]],    # query_id -> [doc_ids]
    ):
        self.queries = {str(k): v for k, v in queries.items()}
        self.corpus = {str(k): v for k, v in corpus.items()}
        self.qrels = {str(k): [str(d) for d in v] for k, v in qrels.items()}

        # ---- sanity check (SAFE) ----
        qid, qtext = next(iter(self.queries.items()))
        print(f"[UnifiedDataset] example query_id={qid} ({type(qid)})")
        print(f"[UnifiedDataset] example query_text='{qtext[:60]}'")

        qid2, rels = next(iter(self.qrels.items()))
        print(f"[UnifiedDataset] qrels for query_id={qid2}: {len(rels)} docs")

        self.max_k = 1000
        self.idcg = np.array(self._precompute_idcg(self.max_k))

    def get_relevant_docs(self, query_id: str) -> List[str]:
        return self.qrels.get(str(query_id), [])



    def compute_ndcg_batch(
        self,
        query_uids: List[str],
        retrieved_doc_ids_batch: List[List[str]],
        k: int = 10,
    ) -> List[float]:
        rewards = []
        discounts = 1.0 / np.log2(np.arange(k) + 2)

        for query_id, docs in zip(query_uids, retrieved_doc_ids_batch):
            rel_docs = set(self.get_relevant_docs(query_id))
            if not rel_docs:
                rewards.append(0.0)
                continue

            docs_k = docs[:k]
            relevance = np.array([1.0 if d in rel_docs else 0.0 for d in docs_k])
            dcg = np.sum(relevance * discounts[: len(relevance)])

            ideal_count = min(len(rel_docs), len(docs_k))
            idcg = self.idcg[ideal_count]

            rewards.append(dcg / idcg if idcg > 0 else 0.0)

        return rewards

    def _precompute_idcg(self, max_k: int):
        """
        Precompute ideal DCG values for binary relevance.
        idcg[n] = sum_{i=1..n} 1 / log2(i + 1)
        """
        idcg = [0.0]
        for i in range(1, max_k + 1):
            idcg.append(idcg[-1] + 1.0 / np.log2(i + 1))
        return idcg


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
        # Pre-allocate lists
        queries_txt = []
        docs_txt = []
        doc_counts = []

        # 1. Single Pass Construction
        for query_id, docs in zip(query_uids, retrieved_doc_ids_batch):
            docs_k = docs[:k]

            if not docs_k:
                doc_counts.append(0)
                continue

            # Extract valid documents first (lookup + filter)
            valid_docs_text = [self.corpus[doc] for doc in docs_k if doc in self.corpus]

            count = len(valid_docs_text)

            # If all docs were missing from corpus, skip
            if count == 0:
                doc_counts.append(0)
                continue

            # Extend the query list by the exact count of valid docs
            current_query_text = self.queries[query_id]
            queries_txt.extend([current_query_text] * count)

            docs_txt.extend(valid_docs_text)
            doc_counts.append(count)

        if not queries_txt:
            return [0.0] * len(query_uids)

        # 2. Payload Construction
        payload = {"text_1": queries_txt, "text_2": docs_txt, "normalize": True}

        # 3. Request
        response = requests.post(reranker_url, json=payload)
        response.raise_for_status()

        # 4. Score Processing
        data = response.json().get("data", [])
        all_scores = np.array([item["score"] for item in data], dtype=np.float32)

        rewards = []
        start_idx = 0

        for count in doc_counts:
            if count == 0:
                rewards.append(0.0)
                continue

            # Slice the scores belonging to this query
            scores = all_scores[start_idx : start_idx + count]

            # STRATEGY: MAX SCORE
            # Reward the model based on the single best document found.
            # This ignores the noise in lower ranks.
            best_score = float(np.max(scores))
            rewards.append(best_score)

            start_idx += count

        return rewards

    def compute_rewards_batch(
        self,
        query_uids: List[str],
        retrieved_doc_ids_batch: List[List[str]],
        method: str | tuple = ("ndcg"),
        k: int = 10,
        reranker_url: str = "http://localhost:8000/v1/score",
    ) -> List[float]:
        """
        Compute rewards for a batch of queries using batched operations.
        """
        if isinstance(method, str):
            method = (method, )
        result = []
        for fn in method:
            if fn == "ndcg":
                result.append(self.compute_ndcg_batch(query_uids, retrieved_doc_ids_batch, k=k))
            elif fn == "recall":
                result.append(self.compute_recall_batch(query_uids, retrieved_doc_ids_batch, k=k))
            elif fn == "precision":
                result.append(self.compute_precision_batch(query_uids, retrieved_doc_ids_batch, k=k))
            elif fn == "hit":
                result.append(self.compute_hit_batch(query_uids, retrieved_doc_ids_batch, k=k))
            elif fn == "reranker":
                result.append(
                    self.compute_reranker_score_batch(
                        query_uids, retrieved_doc_ids_batch, k=k, reranker_url=reranker_url
                    )
                )
            else:
                raise ValueError(f"Unknown method: {method}")
        result = np.array(result)
        return np.mean(result, axis=0)


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
