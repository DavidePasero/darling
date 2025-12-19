from typing import List, Literal, Optional, Tuple
import logging
import numpy as np

import torch
from sentence_transformers import SentenceTransformer
import faiss
import asyncio
import pickle
import os
import time
from openai import AsyncOpenAI

from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class FaissRetriever(BaseRetriever):
    def __init__(
        self,
        faiss_index_path: str,
        embedding_model: str,
        id_mapping_path: Optional[str] = None,
        index_device: str = "cuda",
        device: str = "cuda",
        embedding_mode: str = "local",
        vllm_server_url: Optional[str] = None,
        max_seq_len: int = 512,
        verbose: bool = True
    ):
        super().__init__(id_mapping_path=id_mapping_path, verbose=verbose)
        self.device = device
        self.embedding_model_name = embedding_model
        self.embedding_mode = embedding_mode
        self.vllm_server_url = vllm_server_url
        self.max_seq_len = max_seq_len

        self._setup_embedding_model()
        self._setup_faiss_index(faiss_index_path, index_device)

        if verbose:
            logger.info("FAISS Retriever ready!")

    def _setup_embedding_model(self):
        """Setup embedding model (local or vLLM)."""
        if self.embedding_mode == "vllm":
            self._setup_vllm_embeddings()
        else:
            self._setup_local_embeddings()

    def _setup_vllm_embeddings(self):
        """Setup vLLM embedding server."""
        if not self.vllm_server_url:
            raise ValueError("vllm_server_url required for embedding_mode='vllm'")

        if self.verbose:
            logger.info(f"Using vLLM server: {self.vllm_server_url}")
            logger.info(f"Embedding model: {self.embedding_model_name}")

        self.openai_client = AsyncOpenAI(
            base_url=f"{self.vllm_server_url}/v1",
            api_key="EMPTY"
        )
        self.embedding_model = None
        self.dimension = 1536

    def _setup_local_embeddings(self):
        """Setup local sentence transformer model."""
        if self.verbose:
            logger.info(f"Loading embedding model locally: {self.embedding_model_name}")

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device,
            trust_remote_code=True
        )
        self.embedding_model.eval()
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        if self.verbose:
            logger.info(f"Model dimension: {self.dimension}")

    def _setup_faiss_index(self, faiss_index_path: str, index_device: str):
        """Setup FAISS index from file."""
        if self.verbose:
            logger.info(f"Loading FAISS index: {faiss_index_path}")

        cpu_index = self._load_index_from_file(faiss_index_path)
        self.index = self._move_index_to_device(cpu_index, index_device)

        index_class = self.index.__class__.__name__
        self.supports_nprobe = 'IVF' in index_class or hasattr(self.index, 'nprobe')

        if self.verbose:
            logger.info(f"Index type: {index_class}")
            logger.info(f"Supports nprobe: {self.supports_nprobe}")

    def _load_index_from_file(self, path: str) -> faiss.Index:
        """Load FAISS index from .faiss or .npy file."""
        if path.endswith('.npy'):
            if self.verbose:
                logger.info("Building Flat index from .npy embeddings")
            embeddings = np.load(path)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            return index

        return faiss.read_index(path)

    def _move_index_to_device(self, cpu_index: faiss.Index, device: str) -> faiss.Index:
        """Move FAISS index to specified device."""
        if device != "cuda":
            if self.verbose:
                logger.info(f"Index on CPU: {cpu_index.ntotal} vectors")
            return cpu_index

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

        if self.verbose:
            logger.info("Transferring index to GPU...")

        start = time.time()
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        co.useFloat16LookupTables = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)

        if self.verbose:
            elapsed = time.time() - start
            logger.info(f"Index on GPU: {gpu_index.ntotal} vectors ({elapsed:.2f}s)")

        return gpu_index

    async def _encode_vllm_async(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts using vLLM server via AsyncOpenAI client."""
        # Use max_seq_len directly (vLLM will handle it)
        # Note: truncate_prompt_tokens must be <= max_model_len of the server
        max_tokens = self.max_seq_len

        async def fetch_batch(batch_texts):
            # Use truncate_prompt_tokens for automatic left truncation
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=batch_texts,
                extra_body={"truncate_prompt_tokens": max_tokens}
            )
            return [item.embedding for item in response.data]

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        tasks = [fetch_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        embeddings = np.array([emb for batch_result in results for emb in batch_result], dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        is_query: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings (batched).

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            is_query: Whether these are queries (vs documents)
            show_progress: Show progress bar

        Returns:
            np.ndarray of shape (len(texts), dimension)
        """
        debug_log = os.environ.get("DEBUG_LOG", "0") == "1"

        if debug_log:
            start_time = time.time()
            print(f"[RETRIEVAL DEBUG] Starting embedding encoding for {len(texts)} texts (batch_size={batch_size}, mode={self.embedding_mode})")

        if self.embedding_mode == "vllm":
            result = asyncio.run(self._encode_vllm_async(texts, batch_size))

        else:
            encode_kwargs = {
                "batch_size": batch_size,
                "show_progress_bar": show_progress,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
            }

            # Use query prompt for Qwen models
            if is_query and "Qwen" in self.embedding_model_name:
                encode_kwargs["prompt_name"] = "query"
            elif is_query and "bge" in self.embedding_model_name:
                texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

            embeddings = self.embedding_model.encode(texts, **encode_kwargs)
            result = embeddings.astype("float32")

        if debug_log:
            elapsed = time.time() - start_time
            print(f"[RETRIEVAL DEBUG] Embedding encoding completed in {elapsed:.2f}s ({len(texts)/elapsed:.2f} texts/sec)")

        return result

    async def async_encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        is_query: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings (batched).

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            is_query: Whether these are queries (vs documents)
            show_progress: Show progress bar

        Returns:
            np.ndarray of shape (len(texts), dimension)
        """
        debug_log = os.environ.get("DEBUG_LOG", "0") == "1"

        if debug_log:
            start_time = time.time()
            print(f"[RETRIEVAL DEBUG] Starting embedding encoding for {len(texts)} texts (batch_size={batch_size}, mode={self.embedding_mode})")

        if self.embedding_mode == "vllm":
            result = await self._encode_vllm_async(texts, batch_size)

        else:
            encode_kwargs = {
                "batch_size": batch_size,
                "show_progress_bar": show_progress,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
            }

            # Use query prompt for Qwen models
            if is_query and "Qwen" in self.embedding_model_name:
                encode_kwargs["prompt_name"] = "query"
            elif is_query and "bge" in self.embedding_model_name:
                texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

            embeddings = self.embedding_model.encode(texts, **encode_kwargs)
            result = embeddings.astype("float32")

        if debug_log:
            elapsed = time.time() - start_time
            print(f"[RETRIEVAL DEBUG] Embedding encoding completed in {elapsed:.2f}s ({len(texts)/elapsed:.2f} texts/sec)")

        return result

    def search(
        self,
        queries: List[str],
        k: int = 10,
        nprobe: int = 64,
        batch_size: int = 64,
        search_batch_size: int = 256,
        return_scores: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index for queries.

        Args:
            queries: List of query strings
            k: Number of results to return per query
            nprobe: Number of clusters to search (higher = slower but better)
            batch_size: Batch size for encoding embeddings
            search_batch_size: Batch size for FAISS index search (default: 256)
            return_scores: Whether to return similarity scores

        Returns:
            Tuple of (scores, indices) as np.ndarrays
            - scores: shape (len(queries), k)
            - indices: shape (len(queries), k)
        """

        debug_log = os.environ.get("DEBUG_LOG", "0") == "1"

        if debug_log:
            search_start = time.time()
            print(f"[RETRIEVAL DEBUG] Starting FAISS search for {len(queries)} queries (k={k}, nprobe={nprobe}, search_batch_size={search_batch_size})")

        if nprobe is not None and self.supports_nprobe:
            self.index.nprobe = nprobe
            if debug_log:
                print(f"[RETRIEVAL DEBUG] Set nprobe={nprobe}")

        if debug_log:
            encode_start = time.time()

        query_embeddings = self.encode(
            queries,
            batch_size=batch_size,
            is_query=True
        )

        if debug_log:
            encode_time = time.time() - encode_start
            print(f"[RETRIEVAL DEBUG] Query encoding completed in {encode_time:.2f}s")
            print(f"[RETRIEVAL DEBUG] Query embedding shape: {query_embeddings.shape}")

        if query_embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension ({query_embeddings.shape[1]}) "
                f"does not match index dimension ({self.index.d}). "
                f"Embedding model: {self.embedding_model_name}"
            )

        if debug_log:
            index_search_start = time.time()

        # Batch the FAISS search to control memory usage
        num_queries = query_embeddings.shape[0]
        all_scores = []
        all_indices = []

        for i in range(0, num_queries, search_batch_size):
            batch_embeddings = query_embeddings[i:i + search_batch_size]
            batch_scores, batch_indices = self.index.search(batch_embeddings, k)
            all_scores.append(batch_scores)
            all_indices.append(batch_indices)

        scores = np.vstack(all_scores)
        indices = np.vstack(all_indices)

        if debug_log:
            index_search_time = time.time() - index_search_start
            total_time = time.time() - search_start
            print(f"[RETRIEVAL DEBUG] FAISS index.search() completed in {index_search_time:.2f}s ({len(all_scores)} batches)")
            print(f"[RETRIEVAL DEBUG] Total search time: {total_time:.2f}s (encode: {encode_time:.2f}s, index: {index_search_time:.2f}s)")

        return (scores, indices) if return_scores else indices

    async def async_search(
        self,
        queries: List[str],
        k: int = 10,
        nprobe: int = 64,
        batch_size: int = 64,
        search_batch_size: int = 256,
        return_scores: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index for queries.

        Args:
            queries: List of query strings
            k: Number of results to return per query
            nprobe: Number of clusters to search (higher = slower but better)
            batch_size: Batch size for encoding embeddings
            search_batch_size: Batch size for FAISS index search (default: 256)
            return_scores: Whether to return similarity scores

        Returns:
            Tuple of (scores, indices) as np.ndarrays
            - scores: shape (len(queries), k)
            - indices: shape (len(queries), k)
        """

        debug_log = os.environ.get("DEBUG_LOG", "0") == "1"

        if debug_log:
            search_start = time.time()
            print(f"[RETRIEVAL DEBUG] Starting FAISS search for {len(queries)} queries (k={k}, nprobe={nprobe}, search_batch_size={search_batch_size})")

        if nprobe is not None and self.supports_nprobe:
            self.index.nprobe = nprobe
            if debug_log:
                print(f"[RETRIEVAL DEBUG] Set nprobe={nprobe}")

        if debug_log:
            encode_start = time.time()

        query_embeddings = await self.async_encode(
            queries,
            batch_size=batch_size,
            is_query=True
        )

        if debug_log:
            encode_time = time.time() - encode_start
            print(f"[RETRIEVAL DEBUG] Query encoding completed in {encode_time:.2f}s")
            print(f"[RETRIEVAL DEBUG] Query embedding shape: {query_embeddings.shape}")

        if query_embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension ({query_embeddings.shape[1]}) "
                f"does not match index dimension ({self.index.d}). "
                f"Embedding model: {self.embedding_model_name}"
            )

        if debug_log:
            index_search_start = time.time()

        # Batch the FAISS search to control memory usage
        num_queries = query_embeddings.shape[0]
        all_scores = []
        all_indices = []

        for i in range(0, num_queries, search_batch_size):
            batch_embeddings = query_embeddings[i:i + search_batch_size]
            batch_scores, batch_indices = self.index.search(batch_embeddings, k)
            all_scores.append(batch_scores)
            all_indices.append(batch_indices)

        scores = np.vstack(all_scores)
        indices = np.vstack(all_indices)

        if debug_log:
            index_search_time = time.time() - index_search_start
            total_time = time.time() - search_start
            print(f"[RETRIEVAL DEBUG] FAISS index.search() completed in {index_search_time:.2f}s ({len(all_scores)} batches)")
            print(f"[RETRIEVAL DEBUG] Total search time: {total_time:.2f}s (encode: {encode_time:.2f}s, index: {index_search_time:.2f}s)")

        return (scores, indices) if return_scores else indices

    def map_indices_to_ids(self, indices: np.ndarray) -> np.ndarray:
        """
        Map FAISS indices to real document IDs using NumPy vectorization.
        This is O(1) compared to O(N*K) of the loop version.
        """
        if self.id_mapping is None:
            return indices

        # Convert mapping to numpy array if it isn't already
        # We assume self.id_mapping is a list or array of IDs corresponding to index positions 0..N
        if not isinstance(self.id_mapping, np.ndarray):
            mapping_arr = np.array(self.id_mapping)
        else:
            mapping_arr = self.id_mapping
            
        # Handle -1 (Faiss returns -1 if not enough neighbors found)
        # We create a safe mask
        valid_mask = indices != -1
        
        # Create output array filled with a placeholder (e.g. -1 or empty string)
        # Assuming IDs are strings based on typical BEIR datasets
        doc_ids = np.empty(indices.shape, dtype=object) 
        doc_ids.fill("-1") # Default value

        # Fancy indexing: only map valid indices
        # valid_indices are the actual integers from FAISS
        valid_indices = indices[valid_mask]
        
        # Use numpy array indexing to get all IDs at once
        try:
            doc_ids[valid_mask] = mapping_arr[valid_indices]
        except IndexError as e:
            print(f"Error mapping indices: {e}. Check if id_mapping length matches index size.")
            
        return doc_ids

    def retrieve_batch(
            self,
            query_rewrites: List[List[str]],
            k: int = 10,
            mode: Literal["union", "intersection", "first"] = "union",
            nprobe: int = 64,
            batch_size: int = 128,
            search_batch_size: int = 256
    ):

        debug_log = os.environ.get("DEBUG_LOG", "0") == "1"

        if debug_log:
            batch_start = time.time()
            total_rewrites = sum(len(rw) for rw in query_rewrites)
            print(f"[RETRIEVAL DEBUG] retrieve_batch called: {len(query_rewrites)} queries, {total_rewrites} total rewrites, mode={mode}")

        flat_rewrites = []
        mapping = []

        for qi, rewrites in enumerate(query_rewrites):
            for ri, r in enumerate(rewrites):
                flat_rewrites.append(r)
                mapping.append((qi, ri))

        if len(flat_rewrites) == 0:
            return []

        if debug_log:
            print(f"[RETRIEVAL DEBUG] Flattened to {len(flat_rewrites)} queries for batch search")

        scores_flat, index_flat = self.search(
            flat_rewrites,
            k=k,
            nprobe=nprobe,
            batch_size=batch_size,
            search_batch_size=search_batch_size
        )

        if debug_log:
            mapping_start = time.time()
            print(f"[RETRIEVAL DEBUG] Starting index->ID mapping")

        doc_ids_flat = self.map_indices_to_ids(index_flat)

        if debug_log:
            mapping_time = time.time() - mapping_start
            print(f"[RETRIEVAL DEBUG] Index->ID mapping completed in {mapping_time:.2f}s")

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

                doc_to_scores = {}
                for doc_id in common:
                    vals = []
                    for r in per_rw:
                        if doc_id in r["doc_ids"]:
                            idx = np.where(r["doc_ids"] == doc_id)[0][0]
                            vals.append(r["scores"][idx])
                    doc_to_scores[doc_id] = np.mean(vals)

                sorted_docs = sorted(doc_to_scores.items(), key=lambda x: x[1], reverse=True)
                merged_doc_ids = np.array([d for d, _ in sorted_docs[:k]])
                merged_scores = np.array([s for _, s in sorted_docs[:k]])

            else:
                doc_to_scores = {}
                for r in per_rw:
                    for doc_id, score in zip(r["doc_ids"], r["scores"]):
                        if doc_id not in doc_to_scores:
                            doc_to_scores[doc_id] = []
                        doc_to_scores[doc_id].append(score)

                doc_mean = {doc: np.mean(vals) for doc, vals in doc_to_scores.items()}
                sorted_docs = sorted(doc_mean.items(), key=lambda x: x[1], reverse=True)

                merged_doc_ids = np.array([d for d, _ in sorted_docs[:k]])
                merged_scores = np.array([s for _, s in sorted_docs[:k]])

            results.append({
                "doc_ids": merged_doc_ids,
                "scores": merged_scores,
                "rewrite_results": per_rw
            })

        if debug_log:
            total_time = time.time() - batch_start
            print(f"[RETRIEVAL DEBUG] retrieve_batch completed in {total_time:.2f}s")

        return results
    
    async def async_retrieve_batch(
            self,
            query_rewrites: List[List[str]],
            k: int = 10,
            mode: Literal["union", "intersection", "first"] = "union",
            nprobe: int = 64,
            batch_size: int = 128,
            search_batch_size: int = 256
    ):

        debug_log = os.environ.get("DEBUG_LOG", "0") == "1"

        if debug_log:
            batch_start = time.time()
            total_rewrites = sum(len(rw) for rw in query_rewrites)
            print(f"[RETRIEVAL DEBUG] retrieve_batch called: {len(query_rewrites)} queries, {total_rewrites} total rewrites, mode={mode}")

        flat_rewrites = []
        mapping = []

        for qi, rewrites in enumerate(query_rewrites):
            for ri, r in enumerate(rewrites):
                flat_rewrites.append(r)
                mapping.append((qi, ri))

        if len(flat_rewrites) == 0:
            return []

        if debug_log:
            print(f"[RETRIEVAL DEBUG] Flattened to {len(flat_rewrites)} queries for batch search")

        scores_flat, index_flat = await self.async_search(
            flat_rewrites,
            k=k,
            nprobe=nprobe,
            batch_size=batch_size,
            search_batch_size=search_batch_size
        )

        if debug_log:
            mapping_start = time.time()
            print(f"[RETRIEVAL DEBUG] Starting index->ID mapping")

        doc_ids_flat = self.map_indices_to_ids(index_flat)

        if debug_log:
            mapping_time = time.time() - mapping_start
            print(f"[RETRIEVAL DEBUG] Index->ID mapping completed in {mapping_time:.2f}s")

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

                doc_to_scores = {}
                for doc_id in common:
                    vals = []
                    for r in per_rw:
                        if doc_id in r["doc_ids"]:
                            idx = np.where(r["doc_ids"] == doc_id)[0][0]
                            vals.append(r["scores"][idx])
                    doc_to_scores[doc_id] = np.mean(vals)

                sorted_docs = sorted(doc_to_scores.items(), key=lambda x: x[1], reverse=True)
                merged_doc_ids = np.array([d for d, _ in sorted_docs[:k]])
                merged_scores = np.array([s for _, s in sorted_docs[:k]])

            else:
                doc_to_scores = {}
                for r in per_rw:
                    for doc_id, score in zip(r["doc_ids"], r["scores"]):
                        if doc_id not in doc_to_scores:
                            doc_to_scores[doc_id] = []
                        doc_to_scores[doc_id].append(score)

                doc_mean = {doc: np.mean(vals) for doc, vals in doc_to_scores.items()}
                sorted_docs = sorted(doc_mean.items(), key=lambda x: x[1], reverse=True)

                merged_doc_ids = np.array([d for d, _ in sorted_docs[:k]])
                merged_scores = np.array([s for _, s in sorted_docs[:k]])

            results.append({
                "doc_ids": merged_doc_ids,
                "scores": merged_scores,
                "rewrite_results": per_rw
            })

        if debug_log:
            total_time = time.time() - batch_start
            print(f"[RETRIEVAL DEBUG] retrieve_batch completed in {total_time:.2f}s")

        return results

    def get_index_size(self) -> int:
        """Get the number of documents in the FAISS index."""
        return self.index.ntotal


Retriever = FaissRetriever
