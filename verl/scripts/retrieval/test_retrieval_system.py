#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import time
import requests
import json
from typing import List, Dict, Any

verl_path = Path(__file__).parent.parent
sys.path.insert(0, str(verl_path))

import numpy as np
from verl.retrieval.engine.retriever import FaissRetriever
from verl.retrieval.engine.bm25_retriever import Bm25Retriever
from verl.retrieval.engine.document_dataset import BeirAdapter
from verl.utils.prompt_extension.rewrite_prompt_extenders import MinimalRewritePromptExtender

# --- Logic ---

def generate_rewrites_batch(
    queries: List[str],
    port: int,
    prompt_extender: MinimalRewritePromptExtender,
    n: int = 1,
    model_name: str = "darling_retrieval_bm25"
) -> List[List[str]]:
    """
    Generates N rewrites for a batch of queries using a vLLM server and a PromptExtender.
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    batch_rewrites = []
    
    for query in queries:
        # Use the class to build the prompt messages
        messages = prompt_extender.extend_prompt(query)
        
        payload = {
            "model": model_name,
            "messages": messages,
            "n": n,  # Request N independent completions
            "temperature": 0.7, 
            "max_tokens": 512
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            current_rewrites = []
            for choice in data.get('choices', []):
                # The prompt explicitly asks for "ONLY the rewritten query"
                # so we take the content directly.
                content = choice['message']['content'].strip()
                if content:
                    current_rewrites.append(content)
            
            # Fallback if model output was empty
            if not current_rewrites:
                current_rewrites = [query]
                
            batch_rewrites.append(current_rewrites)
            
        except Exception as e:
            print(f"Error calling LLM for query '{query}': {e}")
            batch_rewrites.append([query])

    return batch_rewrites


def test_retrieval(
    beir_dataset_path: str,
    retriever_type: str = "faiss",
    faiss_index_path: str = None,
    faiss_id_mapping_path: str = None,
    bm25_index_path: str = None,
    bm25_id_mapping_path: str = None,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "cuda",
    k: int = 10,
    nprobe: int = 64,
    use_vllm: bool = False,
    vllm_port: int = 8000,
    num_rewrites: int = 1,
    model_name: str = "darling_retrieval_bm25",
    search_batch_size=128
):
    print(f"Testing {retriever_type.upper()} retrieval system")
    print(f"Dataset: {beir_dataset_path}")
    print(f"K: {k}")
    
    # Initialize the prompt extender
    prompt_extender = MinimalRewritePromptExtender()
    
    if use_vllm:
        print(f"Mode: LLM Query Rewriting")
        print(f"  - Port: {vllm_port}")
        print(f"  - Model: {model_name}")
        print(f"  - N Rewrites: {num_rewrites}")
        print(f"  - Prompt Strategy: {prompt_extender._NAME}")
    else:
        print("Mode: Standard Retrieval (Original Queries)")
        
    print("=" * 80)

    if 'fiqa' in beir_dataset_path: 
        adapter = BeirAdapter(data_path=beir_dataset_path, split="train")
    else:
        adapter = BeirAdapter(data_path=beir_dataset_path, split="dev")
    doc_dataset = adapter.to_unified()

    print(f"Loaded {len(doc_dataset.queries)} queries")
    print(f"Loaded {len(doc_dataset.corpus)} documents")
    print(f"Loaded {len(doc_dataset.qrels)} qrels")

    if retriever_type == "faiss":
        if faiss_index_path is None:
            faiss_index_path = f"{beir_dataset_path}/faiss_index/faiss_index.faiss"
        if faiss_id_mapping_path is None:
            faiss_id_mapping_path = f"{beir_dataset_path}/faiss_index/id_mapping.pkl"

        print(f"FAISS index: {faiss_index_path}")
        print(f"ID mapping: {faiss_id_mapping_path}")

        retriever = FaissRetriever(
            faiss_index_path=faiss_index_path,
            embedding_model=embedding_model,
            id_mapping_path=faiss_id_mapping_path,
            device=device,
            max_seq_len=512,
            verbose=True,
            index_device=device,
        )
    else:
        if bm25_index_path is None:
            bm25_index_path = f"{beir_dataset_path}/bm25_index/index"
        if bm25_id_mapping_path is None:
            bm25_id_mapping_path = f"{beir_dataset_path}/bm25_index/id_mapping.pkl"

        print(f"BM25 index: {bm25_index_path}")
        print(f"ID mapping: {bm25_id_mapping_path}")

        retriever = Bm25Retriever(
            index_path=bm25_index_path,
            id_mapping_path=bm25_id_mapping_path,
            verbose=True
        )
        nprobe = None

    print("\n" + "=" * 80)
    print("Testing retrieval")
    print("=" * 80)

    sample_query_ids = list(doc_dataset.qrels.keys())
    print(f"Testing on {len(sample_query_ids)} queries")
    sample_queries = [doc_dataset.queries[qid] for qid in sample_query_ids]

    start = time.time()

    all_retrieved_ids = []

    for i in range(0, len(sample_queries), search_batch_size):
        batch_queries = sample_queries[i:i+search_batch_size]
        
        if use_vllm:
            print(f"Processing batch {i} with vLLM...")
            
            # Generate Rewrites
            batch_rewrites = generate_rewrites_batch(
                batch_queries, 
                port=vllm_port, 
                prompt_extender=prompt_extender, # Pass the instance here
                n=num_rewrites, 
                model_name=model_name
            )
            
            # Retrieve using retrieve_batch
            batch_results = retriever.retrieve_batch(
                query_rewrites=batch_rewrites,
                k=k,
                mode="union"
            )

            # 3. Extract IDs
            for res in batch_results:
                ids = res['doc_ids']
                if hasattr(ids, 'tolist'):
                    ids = ids.tolist()
                all_retrieved_ids.append(ids)
                
        else:
            # Standard Workflow
            if retriever_type == "faiss":
                scores, indices = retriever.search(batch_queries, k=k, nprobe=nprobe)
                batch_ids = retriever.map_indices_to_ids(indices).tolist()
            else:
                scores, indices = retriever.search(batch_queries, k=k)
                if retriever.id_mapping is not None:
                     batch_ids = retriever.map_indices_to_ids(indices).tolist()
                else:
                     batch_ids = indices.tolist()
            
            all_retrieved_ids.extend(batch_ids)

        if i % (search_batch_size * 5) == 0:
            print(f"Processed {min(i+len(batch_queries), len(sample_queries))}/{len(sample_queries)} queries")

    print(f"Search took {time.time() - start:.4f} seconds")

    # Compute metrics
    print("Computing metrics...")
    
    metrics = ["ndcg", "recall", "precision", "hit"]
    results = {}
    
    for metric in metrics:
        rewards = doc_dataset.compute_rewards_batch(
            query_uids=sample_query_ids,
            retrieved_doc_ids_batch=all_retrieved_ids,
            method=metric,
            k=k
        )
        results[metric] = sum(rewards) / len(rewards)

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Mean NDCG@{k}:      {results['ndcg']:.4f}")
    print(f"Mean Recall@{k}:    {results['recall']:.4f}")
    print(f"Mean Precision@{k}: {results['precision']:.4f}")
    print(f"Mean Hit@{k}:       {results['hit']:.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test retrieval system")
    parser.add_argument("--beir-dataset", required=True, help="Path to BEIR dataset directory")
    parser.add_argument("--retriever-type", default="faiss", choices=["faiss", "bm25"], help="Retriever type")

    # Retriever arguments
    parser.add_argument("--faiss-index", help="Path to FAISS index file")
    parser.add_argument("--faiss-id-mapping", help="Path to FAISS ID mapping file")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--bm25-index", help="Path to BM25 index directory")
    parser.add_argument("--bm25-id-mapping", help="Path to BM25 ID mapping file")

    # LLM arguments
    parser.add_argument("--use-vllm", action="store_true", help="Enable LLM query rewriting")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--num-rewrites", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="darling_retrieval")

    # Common arguments
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--search_batch_size", type=int, default=128)

    args = parser.parse_args()

    print(f"Using top-k: {args.k}")
    print("*"*80)
    
    if args.use_vllm and not args.vllm_port:
         print("Warning: --use-vllm set but no port provided. Using 8000.")

    test_retrieval(
        beir_dataset_path=args.beir_dataset,
        retriever_type=args.retriever_type,
        faiss_index_path=args.faiss_index,
        faiss_id_mapping_path=args.faiss_id_mapping,
        bm25_index_path=args.bm25_index,
        bm25_id_mapping_path=args.bm25_id_mapping,
        embedding_model=args.embedding_model,
        device=args.device,
        k=args.k,
        nprobe=args.nprobe,
        use_vllm=args.use_vllm,
        vllm_port=args.vllm_port,
        num_rewrites=args.num_rewrites,
        model_name=args.model_name,
        search_batch_size=args.search_batch_size
    )

if __name__ == "__main__":
    main()