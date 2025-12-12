from typing import List, Literal, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import asyncio
import aiohttp
import pickle
import os

from base_retriever import BaseRetriever


class FaissRetriever(BaseRetriever):

    def __init__(
        self,
        dataset: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        id_mapping_path: Optional[str] = None,
        index_device: str = "cpu",
        device: str = "cuda",
        embedding_mode: str = "local",
        vllm_server_url: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize retriever with FAISS index and embedding model.

        Args:
            dataset: Preset dataset name ('fiqua' or 'msmarco')
            faiss_index_path: Path to FAISS index file (.faiss) or embeddings (.npy)
            embedding_model: HuggingFace model name for embeddings
            id_mapping_path: Path to pickle or text file with document ID mapping
            index_device: Device for FAISS index ('cuda' or 'cpu')
            device: Device for embedding model ('cuda' or 'cpu')
            embedding_mode: 'local' or 'vllm'
            vllm_server_url: URL for vLLM server (required if embedding_mode='vllm')
            verbose: Print initialization messages
        """
        super().__init__(id_mapping_path=id_mapping_path, verbose=verbose)

        # Handle dataset presets
        if dataset:
            if dataset.lower() == "msmarco":
                faiss_index_path = faiss_index_path or "/home/scur1900/scratch_shared/msmarco/faiss/index"
                id_mapping_path = id_mapping_path or "/home/scur1900/scratch_shared/msmarco/faiss/docid"
                embedding_model = embedding_model or "BAAI/bge-base-en-v1.5"
            elif dataset.lower() == "fiqua":
                faiss_index_path = faiss_index_path or "/home/scur1900/scratch_shared/fiqa/faiss_index/embeddings_cache.npy"
                id_mapping_path = id_mapping_path or "/home/scur1900/scratch_shared/fiqa/bm25_index/id_mapping.pkl"
                embedding_model = embedding_model or "Qwen/Qwen3-Embedding-0.6B"
            else:
                if verbose:
                    print(f"Warning: Unknown dataset '{dataset}'. Please specify paths manually.")

        # Set defaults if not provided and not set by dataset
        embedding_model = embedding_model or "Qwen/Qwen3-Embedding-0.6B"
        
        if not faiss_index_path:
            raise ValueError("faiss_index_path must be provided or set via 'dataset' argument")

        self.device = device
        self.embedding_model_name = embedding_model
        self.embedding_mode = embedding_mode
        self.vllm_server_url = vllm_server_url
        self.id_mapping_path = id_mapping_path

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

        if index_device == "cuda" and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            co.useFloat16LookupTables = True
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
            if verbose:
                print(f"Index on GPU: {self.index.ntotal} vectors")
        else:
            self.index = cpu_index
            if verbose:
                print(f"Index on CPU: {self.index.ntotal} vectors")

        if self.verbose:
            print(f"FAISS Retriever ready!\n")

    async def _encode_vllm_async(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        async def fetch_batch(session, batch_texts):
            async with session.post(
                f"{self.vllm_server_url}/v1/embed",
                json={"model": self.embedding_model_name, "input": batch_texts},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return [item["embedding"] for item in data["data"]]

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_batch(session, batch) for batch in batches]
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
        if self.embedding_mode == "vllm":
            return asyncio.run(self._encode_vllm_async(texts, batch_size))
            
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
            return embeddings.astype("float32")

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
        # Set nprobe parameter
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        elif hasattr(faiss, 'GpuParameterSpace') and hasattr(self.index, 'getResources'):
             # For GPU index check if GpuParameterSpace exists (faiss-gpu) and index is GPU backed
            faiss.GpuParameterSpace().set_index_parameter(self.index, "nprobe", nprobe)

        # Encode queries
        query_embeddings = self.encode(
            queries,
            batch_size=batch_size,
            is_query=True
        )

        # Search
        scores, indices = self.index.search(query_embeddings, k)

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
        flat_rewrites = []
        mapping = []

        for qi, rewrites in enumerate(query_rewrites):
            for ri, r in enumerate(rewrites):
                flat_rewrites.append(r)
                mapping.append((qi, ri))

        if len(flat_rewrites) == 0:
            return []

        scores_flat, index_flat = self.search(
            flat_rewrites,
            k=k,
            nprobe=nprobe,
            batch_size=batch_size
        )

        doc_ids_flat = self.map_indices_to_ids(index_flat)

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

        return results

    def get_index_size(self) -> int:
        """Get the number of documents in the FAISS index."""
        return self.index.ntotal


# Backward compatibility: allow importing as "Retriever"
Retriever = FaissRetriever


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Testing Retriever Class")
    print("=" * 80)

    # This assumes you have a FAISS index and ID mapping
    # Update these paths to your actual files
    retriever = Retriever(
        faiss_index_path="path/to/msmarco.faiss",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        id_mapping_path="path/to/id_mapping.pkl"
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