from typing import List, Literal, Optional, Tuple
import numpy as np
from pyserini.search.lucene import LuceneSearcher

from .base_retriever import BaseRetriever


class Bm25Retriever(BaseRetriever):

    def __init__(
        self,
        index_path: str,
        k1: float = 0.9,
        b: float = 0.4,
        id_mapping_path: Optional[str] = None,
        verbose: bool = True
    ):
        super().__init__(id_mapping_path=id_mapping_path, verbose=verbose)

        self.index_path = index_path
        self.k1 = k1
        self.b = b

        if self.verbose:
            print(f"Loading BM25 index: {index_path}")

        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=k1, b=b)

        if self.verbose:
            print(f"BM25 parameters: k1={k1}, b={b}")
            print(f"Index size: {self.searcher.num_docs} documents")
            print(f"BM25 Retriever ready!\n")

    def search(
        self,
        queries: List[str],
        k: int = 10,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search BM25 index for queries.

        Args:
            queries: List of query strings
            k: Number of results to return per query
            **kwargs: Additional arguments (ignored for BM25)

        Returns:
            Tuple of (scores, doc_ids) as np.ndarrays
            - scores: shape (len(queries), k)
            - doc_ids: shape (len(queries), k) - contains document IDs as strings
        """
        all_scores = []
        all_doc_ids = []

        for query in queries:
            hits = self.searcher.search(query, k=k)

            # Extract scores and doc IDs
            scores = np.zeros(k, dtype=np.float32)
            doc_ids = np.empty(k, dtype=object)

            for i, hit in enumerate(hits):
                scores[i] = hit.score
                doc_ids[i] = hit.docid

            # Pad with -1 if fewer than k results
            if len(hits) < k:
                scores[len(hits):] = -1.0
                doc_ids[len(hits):] = "-1"

            all_scores.append(scores)
            all_doc_ids.append(doc_ids)

        scores_array = np.array(all_scores, dtype=np.float32)
        doc_ids_array = np.array(all_doc_ids, dtype=object)

        return scores_array, doc_ids_array

    def map_indices_to_ids(self, indices: np.ndarray) -> np.ndarray:
        """
        For BM25, indices are already document IDs (strings).
        This method is kept for interface compatibility.

        Args:
            indices: Document IDs from search (already mapped)

        Returns:
            Document IDs (same as input for BM25)
        """
        # BM25 already returns document IDs, not internal indices
        # But if id_mapping is provided, we can still apply it
        if self.id_mapping is None:
            return indices

        # Apply mapping if provided
        mapped_ids = np.zeros_like(indices, dtype=object)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                doc_id = indices[i, j]
                if doc_id != "-1":
                    try:
                        # Try to use as integer index into mapping
                        idx = int(doc_id)
                        if 0 <= idx < len(self.id_mapping):
                            mapped_ids[i, j] = self.id_mapping[idx]
                        else:
                            mapped_ids[i, j] = doc_id
                    except (ValueError, TypeError):
                        # If not an integer, use as-is
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
        flat_rewrites = []
        mapping = []

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

            else:
                doc_to_scores = {}
                for r in per_rw:
                    for doc_id, score in zip(r["doc_ids"], r["scores"]):
                        if doc_id != "-1":
                            if doc_id not in doc_to_scores:
                                doc_to_scores[doc_id] = []
                            doc_to_scores[doc_id].append(score)

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
    # Example usage
    print("=" * 80)
    print("Testing BM25 Retriever Class")
    print("=" * 80)

    # This assumes you have a Pyserini index
    # Update this path to your actual index
    retriever = Bm25Retriever(
        index_path="datasets/msmarco/bm25_index",
        k1=0.9,
        b=0.4
    )

    # Test with multiple rewrites per query
    query_rewrites = [
        ["capital of france", "france capital city", "paris location"],
        ["deep learning", "neural networks", "machine learning"]
    ]

    print("\n" + "=" * 80)
    print("Testing UNION mode (default)")
    print("=" * 80)
    results = retriever.retrieve_batch(query_rewrites, k=5, mode="union")

    for i, result in enumerate(results):
        print(f"\nQuery {i+1}: {query_rewrites[i]}")
        print(f"  Merged Top-5 Doc IDs: {result['doc_ids'][:5]}")
        print(f"  Merged Scores: {result['scores'][:5]}")
        print(f"  Number of rewrites: {len(result['rewrite_results'])}")

    print("\n" + "=" * 80)
    print("Testing INTERSECTION mode")
    print("=" * 80)
    results = retriever.retrieve_batch(query_rewrites, k=5, mode="intersection")

    for i, result in enumerate(results):
        print(f"\nQuery {i+1}: {query_rewrites[i]}")
        print(f"  Common Doc IDs: {result['doc_ids'][:5]}")
        print(f"  Scores: {result['scores'][:5]}")