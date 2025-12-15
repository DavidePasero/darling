from typing import List, Literal, Optional, Tuple
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
        """
        Initialize retriever with FAISS index and embedding model.

        Args:
            faiss_index_path: Path to FAISS index file (.faiss) or embeddings (.npy)
            embedding_model: HuggingFace model name for embeddings
            id_mapping_path: Path to pickle or text file with document ID mapping
            index_device: Device for FAISS index ('cuda' or 'cpu')
            device: Device for embedding model ('cuda' or 'cpu')
            embedding_mode: 'local' or 'vllm'
            vllm_server_url: URL for vLLM server (required if embedding_mode='vllm')
            max_seq_len: Maximum sequence length for embeddings (default 512)
            verbose: Print initialization messages
        """
        super().__init__(id_mapping_path=id_mapping_path, verbose=verbose)

        self.device = device
        self.embedding_model_name = embedding_model
        self.embedding_mode = embedding_mode
        self.vllm_server_url = vllm_server_url
        self.id_mapping_path = id_mapping_path
        self.max_seq_len = max_seq_len

        # Load ID mapping if provided
        self.id_mapping = None
        if self.id_mapping_path and os.path.exists(self.id_mapping_path):
            if verbose:
                print(f"Loading ID mapping: {self.id_mapping_path}")
            
            try:
                # Try pickle first
                with open(self.id_mapping_path, 'rb') as f:
                    self.id_mapping = pickle.load(f)
            except Exception:
                # Fallbck to text file (line-separated IDs)
                if verbose:
                    print(f"Pickle load failed, trying text mode for: {self.id_mapping_path}")
                with open(self.id_mapping_path, 'r') as f:
                    self.id_mapping = [line.strip() for line in f]
            
            if verbose:
                print(f"Loaded {len(self.id_mapping)} document IDs")

        if embedding_mode == "vllm":
            if vllm_server_url is None:
                raise ValueError("vllm_server_url required for embedding_mode='vllm'")

            if verbose:
                print(f"Using vLLM server: {vllm_server_url}")
                print(f"Embedding model: {embedding_model}")

            # Initialize AsyncOpenAI client pointing to vLLM server
            self.openai_client = AsyncOpenAI(
                base_url=f"{vllm_server_url}/v1",
                api_key="EMPTY"  # vLLM doesn't need a real API key
            )
            self.embedding_model = None
            self.dimension = 1536

        else:
            if verbose:
                print(f"Loading embedding model locally: {embedding_model}")

            self.embedding_model = SentenceTransformer(
                embedding_model,
                device=device,
                trust_remote_code=True
            )
            self.embedding_model.eval()
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()

            if verbose:
                print(f"Model dimension: {self.dimension}")

        if verbose:
            print(f"Loading FAISS index: {faiss_index_path}")

        if faiss_index_path.endswith('.npy'):
            if verbose:
                print("Detected .npy file, building Flat index from embeddings...")
            embeddings = np.load(faiss_index_path)
            dimension = embeddings.shape[1]
            cpu_index = faiss.IndexFlatIP(dimension)
            cpu_index.add(embeddings)
        else:
            cpu_index = faiss.read_index(faiss_index_path)

        # Move index to GPU if requested
        if index_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() returned False")

            if verbose:
                print(f"Transferring index to GPU (this may take a minute for large indices)...")

            transfer_start = time.time()
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            co.useFloat16LookupTables = True
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)

            if verbose:
                transfer_time = time.time() - transfer_start
                print(f"Index on GPU: {self.index.ntotal} vectors (transfer took {transfer_time:.2f}s)")
        
        else:
            self.index = cpu_index
            if verbose:
                print(f"Index on CPU: {self.index.ntotal} vectors")

        # Detect index type and whether it supports nprobe
        index_class = self.index.__class__.__name__
        self.supports_nprobe = 'IVF' in index_class or hasattr(self.index, 'nprobe')

        if verbose:
            print(f"Index type: {index_class}")
            print(f"Supports nprobe: {self.supports_nprobe}")
            print(f"FAISS Retriever ready!\n")

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

    def search(
        self,
        queries: List[str],
        k: int = 10,
        nprobe: int = 64,
        batch_size: int = 64,
        return_scores: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index for queries.

        Args:
            queries: List of query strings
            k: Number of results to return per query
            nprobe: Number of clusters to search (higher = slower but better)
            batch_size: Batch size for encoding
            return_scores: Whether to return similarity scores

        Returns:
            Tuple of (scores, indices) as np.ndarrays
            - scores: shape (len(queries), k)
            - indices: shape (len(queries), k)
        """
        debug_log = os.environ.get("DEBUG_LOG", "0") == "1"

        if debug_log:
            search_start = time.time()
            print(f"[RETRIEVAL DEBUG] Starting FAISS search for {len(queries)} queries (k={k}, nprobe={nprobe})")

        if nprobe is not None and self.supports_nprobe:
            self.index.nprobe = nprobe
            if debug_log:
                print(f"[RETRIEVAL DEBUG] Set nprobe={nprobe}")

        # Encode queries
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

        scores, indices = self.index.search(query_embeddings, k)

        if debug_log:
            index_search_time = time.time() - index_search_start
            total_time = time.time() - search_start
            print(f"[RETRIEVAL DEBUG] FAISS index.search() completed in {index_search_time:.2f}s")
            print(f"[RETRIEVAL DEBUG] Total search time: {total_time:.2f}s (encode: {encode_time:.2f}s, index: {index_search_time:.2f}s)")

        return (scores, indices) if return_scores else indices

    def map_indices_to_ids(self, indices: np.ndarray) -> np.ndarray:
        """
        Map FAISS indices to real document IDs.

        Args:
            indices: FAISS indices, shape (batch, k)

        Returns:
            Document IDs, same shape as input
        """
        if self.id_mapping is None:
            return indices

        # Vectorized mapping
        doc_ids = np.zeros_like(indices)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                idx = indices[i, j]
                if idx >= 0 and idx < len(self.id_mapping):
                    doc_ids[i, j] = self.id_mapping[idx]
                else:
                    doc_ids[i, j] = -1

        return doc_ids

    def retrieve_batch(
            self,
            query_rewrites: List[List[str]],
            k: int = 10,
            mode: Literal["union", "intersection", "first"] = "union",
            nprobe: int = 64,
            batch_size: int = 128
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
            batch_size=batch_size
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
