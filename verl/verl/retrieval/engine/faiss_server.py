"""
FastAPI server for FAISS retrieval with low-latency batch search.
Designed to run on a separate GPU from Ray training processes.
"""
from typing import List, Optional
import os
import logging
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from verl.retrieval.engine.retriever import FaissRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    """Request model for batch search."""
    queries: List[str]
    k: int = 10
    nprobe: Optional[int] = 64
    batch_size: int = 64
    search_batch_size: int = 256


class SearchResponse(BaseModel):
    """Response model for batch search."""
    scores: List[List[float]]
    indices: List[List[int]]


class RetrieveBatchRequest(BaseModel):
    """Request model for retrieve_batch with query rewrites."""
    query_rewrites: List[List[str]]
    k: int = 10
    mode: str = "union"
    nprobe: Optional[int] = 64
    batch_size: int = 128
    search_batch_size: int = 256


class RetrieveBatchResponse(BaseModel):
    """Response model for retrieve_batch."""
    results: List[dict]


app = FastAPI(title="FAISS Retrieval Server")

retriever: Optional[FaissRetriever] = None


@app.on_event("startup")
async def startup_event():
    """Initialize FAISS retriever on startup."""
    global retriever

    faiss_index_path = os.environ.get("FAISS_INDEX_PATH")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    id_mapping_path = os.environ.get("ID_MAPPING_PATH")
    device = os.environ.get("DEVICE", "cuda")
    index_device = os.environ.get("INDEX_DEVICE", "cuda")
    embedding_mode = os.environ.get("EMBEDDING_MODE", "local")
    vllm_server_url = os.environ.get("VLLM_SERVER_URL")
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "512"))

    if not faiss_index_path:
        raise RuntimeError("FAISS_INDEX_PATH environment variable is required")

    logger.info("=" * 80)
    logger.info("Initializing FAISS Retrieval Server")
    logger.info("=" * 80)
    logger.info(f"FAISS Index: {faiss_index_path}")
    logger.info(f"Embedding Model: {embedding_model}")
    logger.info(f"Embedding Mode: {embedding_mode}")
    logger.info(f"ID Mapping: {id_mapping_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Index Device: {index_device}")
    if embedding_mode == "vllm":
        logger.info(f"vLLM Server URL: {vllm_server_url}")
    logger.info("=" * 80)

    retriever = FaissRetriever(
        faiss_index_path=faiss_index_path,
        embedding_model=embedding_model,
        id_mapping_path=id_mapping_path,
        device=device,
        index_device=index_device,
        embedding_mode=embedding_mode,
        vllm_server_url=vllm_server_url,
        max_seq_len=max_seq_len,
        verbose=True
    )

    logger.info("FAISS Retrieval Server is ready!")
    logger.info("=" * 80)


@app.get("/health")
async def health():
    """Health check endpoint."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return {
        "status": "healthy",
        "index_size": retriever.get_index_size()
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Batch search endpoint.

    Args:
        queries: List of query strings
        k: Number of results per query
        nprobe: Number of clusters to probe (FAISS IVF indices)
        batch_size: Encoding batch size
        search_batch_size: FAISS search batch size

    Returns:
        scores and indices as nested lists
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    scores, indices = await retriever.async_search(
        queries=request.queries,
        k=request.k,
        nprobe=request.nprobe,
        batch_size=request.batch_size,
        search_batch_size=request.search_batch_size,
        return_scores=True
    )

    return SearchResponse(
        scores=scores.tolist(),
        indices=indices.tolist()
    )


@app.post("/retrieve_batch", response_model=RetrieveBatchResponse)
async def retrieve_batch(request: RetrieveBatchRequest):
    """
    Retrieve batch endpoint with query rewrites.

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
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    results = retriever.async_retrieve_batch(
        query_rewrites=request.query_rewrites,
        k=request.k,
        mode=request.mode,
        nprobe=request.nprobe,
        batch_size=request.batch_size,
        search_batch_size=request.search_batch_size
    )

    serialized_results = []
    for res in results:
        serialized_results.append({
            "doc_ids": res["doc_ids"].tolist() if hasattr(res["doc_ids"], "tolist") else res["doc_ids"],
            "scores": res["scores"].tolist() if hasattr(res["scores"], "tolist") else res["scores"],
            "rewrite_results": [
                {
                    "doc_ids": rr["doc_ids"].tolist() if hasattr(rr["doc_ids"], "tolist") else rr["doc_ids"],
                    "scores": rr["scores"].tolist() if hasattr(rr["scores"], "tolist") else rr["scores"]
                }
                for rr in res.get("rewrite_results", [])
            ]
        })

    return RetrieveBatchResponse(results=serialized_results)


def main():
    """Main entry point for running the server."""
    port = int(os.environ.get("PORT", "8002"))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
