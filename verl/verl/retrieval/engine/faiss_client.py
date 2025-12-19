"""
HTTP client for FAISS retrieval server.
"""
import logging
from typing import List, Optional
import asyncio
import httpx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaissClient:
    """
    Client for communicating with FAISS retrieval FastAPI server.
    """

    def __init__(self, server_url: str, timeout: float = 300.0, verbose: bool = True):
        """
        Initialize FAISS client.

        Args:
            server_url: URL of FAISS server (e.g., "http://localhost:8002")
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.verbose = verbose

        if self.verbose:
            logger.info(f"Initialized FaissClient with server: {self.server_url}")

    def search(
        self,
        queries: List[str],
        k: int = 10,
        nprobe: Optional[int] = 64,
        batch_size: int = 64,
        search_batch_size: int = 256,
        return_scores: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k documents via FAISS server.

        Args:
            queries: List of query strings
            k: Number of results per query
            nprobe: Number of clusters to probe
            batch_size: Encoding batch size
            search_batch_size: FAISS search batch size
            return_scores: Whether to return scores (always True for this client)

        Returns:
            Tuple of (scores, indices) as np.ndarrays
        """
        return asyncio.run(self._search_async(queries, k, nprobe, batch_size, search_batch_size))

    async def _search_async(
        self,
        queries: List[str],
        k: int,
        nprobe: Optional[int],
        batch_size: int,
        search_batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Async implementation of search."""
        payload = {
            "queries": queries,
            "k": k,
            "nprobe": nprobe,
            "batch_size": batch_size,
            "search_batch_size": search_batch_size
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.server_url}/search", json=payload)
            response.raise_for_status()
            data = response.json()

        scores = np.array(data["scores"], dtype=np.float32)
        indices = np.array(data["indices"], dtype=np.int64)

        return scores, indices

    def retrieve_batch(
        self,
        query_rewrites: List[List[str]],
        k: int = 10,
        mode: str = "union",
        nprobe: Optional[int] = 64,
        batch_size: int = 128,
        search_batch_size: int = 256
    ) -> List[dict]:
        """
        Retrieve documents for multiple queries with multiple rewrites.

        Args:
            query_rewrites: List of lists of rewritten queries
            k: Number of results per query
            mode: Combination mode ("union", "intersection", "first")
            nprobe: Number of clusters to probe
            batch_size: Encoding batch size
            search_batch_size: FAISS search batch size

        Returns:
            List of result dicts with doc_ids, scores, and rewrite_results
        """
        return asyncio.run(self._retrieve_batch_async(query_rewrites, k, mode, nprobe, batch_size, search_batch_size))

    async def _retrieve_batch_async(
        self,
        query_rewrites: List[List[str]],
        k: int,
        mode: str,
        nprobe: Optional[int],
        batch_size: int,
        search_batch_size: int
    ) -> List[dict]:
        """Async implementation of retrieve_batch."""
        payload = {
            "query_rewrites": query_rewrites,
            "k": k,
            "mode": mode,
            "nprobe": nprobe,
            "batch_size": batch_size,
            "search_batch_size": search_batch_size
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.server_url}/retrieve_batch", json=payload)
            response.raise_for_status()
            data = response.json()

        results = []
        for res in data["results"]:
            results.append({
                "doc_ids": np.array(res["doc_ids"]),
                "scores": np.array(res["scores"]),
                "rewrite_results": [
                    {
                        "doc_ids": np.array(rr["doc_ids"]),
                        "scores": np.array(rr["scores"])
                    }
                    for rr in res.get("rewrite_results", [])
                ]
            })

        return results

    def get_index_size(self) -> int:
        """
        Get the number of documents in the index via server.

        Returns:
            Number of indexed documents
        """
        return asyncio.run(self._get_index_size_async())

    async def _get_index_size_async(self) -> int:
        """Async implementation of get_index_size."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.server_url}/health")
            response.raise_for_status()
            data = response.json()

        return data["index_size"]

    def map_indices_to_ids(self, indices: np.ndarray) -> np.ndarray:
        """
        Map internal indices to document IDs.

        Note: This is handled server-side. If you need ID mapping,
        ensure the server has id_mapping_path configured.

        Args:
            indices: Internal indices, shape (batch, k)

        Returns:
            Document IDs (same as indices if no mapping on server)
        """
        return indices

    def __repr__(self) -> str:
        """String representation of client."""
        return f"FaissClient(server={self.server_url})"
