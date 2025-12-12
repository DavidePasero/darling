from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Literal
import pickle
import numpy as np


class BaseRetriever(ABC):

    def __init__(
        self,
        id_mapping_path: Optional[str] = None,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.id_mapping = None

        if id_mapping_path:
            if self.verbose:
                print(f"Loading ID mapping: {id_mapping_path}")
            
            try:
                with open(id_mapping_path, 'rb') as f:
                    self.id_mapping = pickle.load(f)
            except Exception:
                if self.verbose:
                    print(f"Pickle load failed, trying text mode for: {id_mapping_path}")
                with open(id_mapping_path, 'r') as f:
                    self.id_mapping = [line.strip() for line in f]

            if self.verbose:
                print(f"Loaded {len(self.id_mapping)} document IDs")

    @abstractmethod
    def search(
        self,
        queries: List[str],
        k: int = 10,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k documents for each query.

        Args:
            queries: List of query strings
            k: Number of results to return per query
            **kwargs: Retriever-specific parameters

        Returns:
            Tuple of (scores, indices) as np.ndarrays
            - scores: shape (len(queries), k)
            - indices: shape (len(queries), k)
        """
        pass

    def map_indices_to_ids(self, indices: np.ndarray) -> np.ndarray:
        """
        Map internal indices to real document IDs.

        Args:
            indices: Internal indices, shape (batch, k)

        Returns:
            Document IDs, same shape as input
        """
        if self.id_mapping is None:
            return indices

        # Vectorized mapping
        doc_ids = np.zeros_like(indices, dtype=object)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                idx = indices[i, j]
                if 0 <= idx < len(self.id_mapping):
                    doc_ids[i, j] = self.id_mapping[idx]
                else:
                    doc_ids[i, j] = -1

        return doc_ids

    @abstractmethod
    def retrieve_batch(
        self,
        query_rewrites: List[List[str]],
        k: int = 10,
        mode: Literal["union", "intersection", "first"] = "union",
        **kwargs
    ) -> List[dict]:
        """
        Retrieve documents for multiple queries with multiple rewrites each.

        Args:
            query_rewrites: Outer list = queries, inner list = rewrites per query
            k: Number of results to return per query
            mode: How to combine rewrites ("union", "intersection", "first")
            **kwargs: Retriever-specific parameters

        Returns:
            List of dicts with keys:
            - "doc_ids": np.ndarray of document IDs
            - "scores": np.ndarray of scores
            - "rewrite_results": List of per-rewrite results
        """
        pass

    @abstractmethod
    def get_index_size(self) -> int:
        """
        Get the number of documents in the index.

        Returns:
            Number of indexed documents
        """
        pass

    def __repr__(self) -> str:
        """String representation of retriever."""
        return f"{self.__class__.__name__}(documents={self.get_index_size()})"