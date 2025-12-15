from typing import List, Literal, Optional, Tuple
import numpy as np
from pyserini.search.lucene import LuceneSearcher
import os

from .base_retriever import BaseRetriever


class Bm25Retriever(BaseRetriever):

    def __init__(
        self,
        index_path: str,
        k1: float = 0.9,
        b: float = 0.4,
        id_mapping_path: Optional[str] = None,
        num_threads: Optional[int] = None,
        verbose: bool = True
    ):
        super().__init__(id_mapping_path=id_mapping_path, verbose=verbose)

        self.index_path = index_path
        self.k1 = k1
        self.b = b

        # Default to number of CPUs if not specified
        if num_threads is None:
            num_threads = min(32, (os.cpu_count() or 1))
        self.num_threads = num_threads

        if self.verbose:
            print(f"Loading BM25 index: {index_path}")

        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=k1, b=b)

        if self.verbose:
            print(f"BM25 parameters: k1={k1}, b={b}")
            print(f"Index size: {self.searcher.num_docs} documents")
            print(f"Number of search threads: {self.num_threads}")
            print("BM25 Retriever ready!\n")

    def search(
        self,
        queries: List[str],
        k: int = 10,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched BM25 search using Pyserini's built-in batch_search.

        Args:
            queries: List of query strings
            k: Number of results per query

        Returns:
            Tuple (scores, doc_ids):
                - scores: np.ndarray of shape (len(queries), k)
                - doc_ids: np.ndarray of shape (len(queries), k)
        """
        if len(queries) == 0:
            return np.array([]), np.array([])

        qids = [str(i) for i in range(len(queries))]

        batch_hits = self.searcher.batch_search(
            queries=queries,
            qids=qids,
            k=k,
            threads=self.num_threads
        )

        scores = np.full((len(queries), k), -1.0, dtype=np.float32)
        doc_ids = np.full((len(queries), k), "-1", dtype=object)

        for qi, qid in enumerate(qids):
            hits = batch_hits.get(qid, [])
            for j, hit in enumerate(hits):
                scores[qi, j] = hit.score
                doc_ids[qi, j] = hit.docid

        return scores, doc_ids

    def map_indices_to_ids(self, indices: np.ndarray) -> np.ndarray:
        """
        For BM25, indices are already document IDs (strings).
        """
        if self.id_mapping is None:
            return indices

        mapped_ids = np.zeros_like(indices, dtype=object)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                doc_id = indices[i, j]
                if doc_id != "-1":
                    try:
                        idx = int(doc_id)
                        if 0 <= idx < len(self.id_mapping):
                            mapped_ids[i, j] = self.id_mapping[idx]
                        else:
                            mapped_ids[i, j] = doc_id
                    except (ValueError, TypeError):
                        mapped_ids[i, j] = doc_id
                else:
                    mapped_ids[i, j] = "-1"

        return mapped_ids

    def retrieve_batch(
        self,
        query_rewrites: List[List[str]],
        k: int = 10,
        mode: Literal["union", "intersection", "first"] = "union",
        **kwargs
    ) -> List[dict]:
        """
        Retrieve documents for batches of query rewrites using batched BM25 search.
        """
        flat_rewrites, mapping = [], []
        for qi, rewrites in enumerate(query_rewrites):
            for ri, r in enumerate(rewrites):
                flat_rewrites.append(r)
                mapping.append((qi, ri))

        if len(flat_rewrites) == 0:
            return []

        scores_flat, doc_ids_flat = self.search(flat_rewrites, k=k)

        if self.id_mapping is not None:
            doc_ids_flat = self.map_indices_to_ids(doc_ids_flat)

        Q = len(query_rewrites)
        rewrite_results = [[] for _ in range(Q)]

        for flat_i, (qi, ri) in enumerate(mapping):
            rewrite_results[qi].append({
                "doc_ids": doc_ids_flat[flat_i],
                "scores": scores_flat[flat_i]
            })

        results = []
        for qi, rewrites in enumerate(query_rewrites):
            if len(rewrites) == 0:
                results.append({
                    "doc_ids": np.array([]),
                    "scores": np.array([]),
                    "rewrite_results": []
                })
                continue

            per_rw = rewrite_results[qi]

            if mode == "first":
                merged_doc_ids = per_rw[0]["doc_ids"]
                merged_scores = per_rw[0]["scores"]

            elif mode == "intersection":
                doc_sets = [set(r["doc_ids"]) for r in per_rw]
                common = set.intersection(*doc_sets)
                common.discard("-1")

                doc_to_scores = {}
                for doc_id in common:
                    vals = []
                    for r in per_rw:
                        if doc_id in r["doc_ids"]:
                            idx = np.where(r["doc_ids"] == doc_id)[0][0]
                            vals.append(r["scores"][idx])
                    doc_to_scores[doc_id] = np.mean(vals)

                sorted_docs = sorted(doc_to_scores.items(), key=lambda x: x[1], reverse=True)
                merged_doc_ids = np.array([d for d, _ in sorted_docs[:k]], dtype=object)
                merged_scores = np.array([s for _, s in sorted_docs[:k]], dtype=np.float32)

            else:  # union
                doc_to_scores = {}
                for r in per_rw:
                    for doc_id, score in zip(r["doc_ids"], r["scores"]):
                        if doc_id != "-1":
                            doc_to_scores.setdefault(doc_id, []).append(score)

                doc_mean = {doc: np.mean(vals) for doc, vals in doc_to_scores.items()}
                sorted_docs = sorted(doc_mean.items(), key=lambda x: x[1], reverse=True)

                merged_doc_ids = np.array([d for d, _ in sorted_docs[:k]], dtype=object)
                merged_scores = np.array([s for _, s in sorted_docs[:k]], dtype=np.float32)

            results.append({
                "doc_ids": merged_doc_ids,
                "scores": merged_scores,
                "rewrite_results": per_rw
            })

        return results

    def get_index_size(self) -> int:
        """Get the number of documents in the BM25 index."""
        return self.searcher.num_docs


if __name__ == "__main__":
    retriever = Bm25Retriever(
        index_path="datasets/msmarco/bm25_index",
        k1=0.9,
        b=0.4
    )

    query_rewrites = [
        ["capital of france", "france capital city", "paris location"],
        ["deep learning", "neural networks", "machine learning"]
    ]

    print("=== UNION mode ===")
    results = retriever.retrieve_batch(query_rewrites, k=5, mode="union")
    for i, result in enumerate(results):
        print(f"Query {i+1}: {query_rewrites[i]}")
        print(f"Top Docs: {result['doc_ids'][:5]}")
        print(f"Scores: {result['scores'][:5]}")

    print("\n=== INTERSECTION mode ===")
    results = retriever.retrieve_batch(query_rewrites, k=5, mode="intersection")
    for i, result in enumerate(results):
        print(f"Query {i+1}: {query_rewrites[i]}")
        print(f"Common Docs: {result['doc_ids'][:5]}")
        print(f"Scores: {result['scores'][:5]}")
