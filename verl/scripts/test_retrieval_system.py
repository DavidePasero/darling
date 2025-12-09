#!/usr/bin/env python3
"""
Test Retrieval System - Standalone Test

This script tests the retrieval infrastructure WITHOUT any RL training:
1. Load FAISS index
2. Load embedding model
3. Test query encoding
4. Test retrieval
5. Compute sample rewards (NDCG, Recall)

Use this to verify your FAISS setup before running training.
"""

import sys
import argparse
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.retrieval.engine.retriever import Retriever
from verl.retrieval.engine.document_dataset import BeirAdapter


def test_retrieval_system(
    beir_dataset_path: str,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "cuda"
):
    """
    Test the complete retrieval system.

    Args:
        beir_dataset_path: Path to BEIR dataset directory
        embedding_model: Embedding model name
        device: cuda or cpu
    """
    print("=" * 80)
    print("TESTING RETRIEVAL SYSTEM")
    print("=" * 80)

    # Paths
    faiss_index_path = f"{beir_dataset_path}/faiss_index.faiss"
    id_mapping_path = f"{beir_dataset_path}/id_mapping.pkl"

    # Step 1: Initialize Retriever
    print("\n" + "=" * 80)
    print("Step 1: Initialize Retriever")
    print("=" * 80)

    retriever = Retriever(
        faiss_index_path=faiss_index_path,
        embedding_model=embedding_model,
        id_mapping_path=id_mapping_path,
        device=device,
        verbose=True
    )

    # Step 2: Initialize Document Dataset
    print("\n" + "=" * 80)
    print("Step 2: Initialize Document Dataset (UnifiedDataset via BeirAdapter)")
    print("=" * 80)

    adapter = BeirAdapter(data_path=beir_dataset_path, split="train")
    doc_dataset = adapter.to_unified()

    print(f"Loaded {len(doc_dataset.queries)} queries")
    print(f"Loaded {len(doc_dataset.corpus)} documents")
    print(f"Loaded {len(doc_dataset.qrels)} query relevance labels")

    # Step 3: Test single query retrieval
    print("\n" + "=" * 80)
    print("Step 3: Test Single Query Retrieval")
    print("=" * 80)

    test_queries = [
        "what is the capital of france",
        "deep learning neural networks",
        "machine learning algorithms"
    ]

    print(f"\nTesting {len(test_queries)} queries...")
    scores, indices = retriever.search(test_queries, k=5, nprobe=64)
    doc_ids = retriever.map_indices_to_ids(indices)

    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: '{query}'")
        print(f"  Top-5 Doc IDs: {doc_ids[i]}")
        print(f"  Scores: {scores[i]}")

    # Step 4: Test multiple rewrites per query
    print("\n" + "=" * 80)
    print("Step 4: Test Multiple Rewrites (Query Rewriting Scenario)")
    print("=" * 80)

    query_rewrites = [
        ["capital of france", "france capital city", "paris location"],
        ["deep learning", "neural networks", "ML algorithms"]
    ]

    print("\nTesting UNION mode (combine all retrieved docs)...")
    results = retriever.retrieve_batch(query_rewrites, k=10, mode="union")

    for i, result in enumerate(results):
        print(f"\nOriginal Query {i+1} Rewrites: {query_rewrites[i]}")
        print(f"  Merged Top-10 Doc IDs: {result['doc_ids'][:10]}")
        print(f"  Merged Scores: {result['scores'][:10]}")
        print(f"  Number of unique docs found: {len(result['doc_ids'])}")

    print("\nTesting INTERSECTION mode (only common docs)...")
    results = retriever.retrieve_batch(query_rewrites, k=10, mode="intersection")

    for i, result in enumerate(results):
        print(f"\nOriginal Query {i+1} Common Docs:")
        print(f"  Doc IDs: {result['doc_ids'][:10]}")
        print(f"  Number of common docs: {len(result['doc_ids'])}")

    # Step 5: Test reward computation
    print("\n" + "=" * 80)
    print("Step 5: Test Reward Computation with Real Queries")
    print("=" * 80)

    # Get sample queries from dataset
    sample_query_ids = list(doc_dataset.queries.keys())[:3]
    if sample_query_ids:
        print(f"\nTesting with {len(sample_query_ids)} real queries from dataset")

        for query_id in sample_query_ids:
            query_text = doc_dataset.queries[query_id]
            relevant_docs = doc_dataset.get_relevant_docs(query_id)

            if not relevant_docs:
                continue

            # Retrieve for this query
            scores, indices = retriever.search([query_text], k=10, nprobe=64)
            retrieved_doc_ids = retriever.map_indices_to_ids(indices)[0]

            # Compute metrics
            ndcg = doc_dataset.compute_ndcg(query_id, retrieved_doc_ids, k=10)
            recall = doc_dataset.compute_recall(query_id, retrieved_doc_ids, k=10)
            precision = doc_dataset.compute_precision(query_id, retrieved_doc_ids, k=10)
            hit = doc_dataset.compute_hit(query_id, retrieved_doc_ids, k=10)

            print(f"\n" + "-" * 60)
            print(f"Query ID: {query_id}")
            print(f"Query: {query_text[:100]}...")
            print(f"Relevant docs: {len(relevant_docs)}")
            print(f"Retrieved docs: {retrieved_doc_ids[:5]}...")
            print(f"\nMetrics:")
            print(f"  NDCG@10:     {ndcg:.4f}")
            print(f"  Recall@10:   {recall:.4f}")
            print(f"  Precision@10: {precision:.4f}")
            print(f"  Hit@10:      {hit:.4f}")

        # Test batch computation
        print(f"\n" + "-" * 60)
        print("Testing batch reward computation...")

        test_query_ids = sample_query_ids[:3]
        test_queries = [doc_dataset.queries[qid] for qid in test_query_ids]

        # Batch retrieval
        scores, indices = retriever.search(test_queries, k=10, nprobe=64)
        retrieved_doc_ids_batch = retriever.map_indices_to_ids(indices).tolist()

        # Batch reward computation
        ndcg_rewards = doc_dataset.compute_rewards_batch(
            query_uids=test_query_ids,
            retrieved_doc_ids_batch=retrieved_doc_ids_batch,
            method="ndcg",
            k=10
        )

        recall_rewards = doc_dataset.compute_rewards_batch(
            query_uids=test_query_ids,
            retrieved_doc_ids_batch=retrieved_doc_ids_batch,
            method="recall",
            k=10
        )

        print(f"\nBatch Results ({len(test_query_ids)} queries):")
        for i, qid in enumerate(test_query_ids):
            print(f"  Query {qid}: NDCG={ndcg_rewards[i]:.4f}, Recall={recall_rewards[i]:.4f}")

    else:
        print("\nNo queries found in dataset.")

    # Summary
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nYour retrieval system is working correctly!")
    print("You can now proceed to training with:")
    print("  bash scripts/train_retrieval_beir.sh")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test Retrieval System")
    parser.add_argument(
        "--beir-dataset",
        required=True,
        help="Path to BEIR dataset directory (should contain faiss_index.faiss and id_mapping.pkl)"
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model name (default: Qwen/Qwen3-Embedding-0.6B)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )

    args = parser.parse_args()

    test_retrieval_system(
        beir_dataset_path=args.beir_dataset,
        embedding_model=args.embedding_model,
        device=args.device
    )


if __name__ == "__main__":
    main()