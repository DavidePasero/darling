#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

retrieval_path = Path(__file__).parent.parent / "verl" / "retrieval"
sys.path.insert(0, str(retrieval_path.parent))

from retrieval.engine.retriever import FaissRetriever
from retrieval.engine.bm25_retriever import Bm25Retriever
from retrieval.engine.document_dataset import BeirAdapter


def test_retrieval(
    beir_dataset_path: str,
    retriever_type: str = "faiss",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "cuda",
    k: int = 10
):
    print(f"Testing {retriever_type.upper()} retrieval system")
    print(f"Dataset: {beir_dataset_path}")
    print(f"K: {k}")
    print("=" * 80)

    adapter = BeirAdapter(data_path=beir_dataset_path, split="train")
    doc_dataset = adapter.to_unified()
    
    print(f"Loaded {len(doc_dataset.queries)} queries")
    print(f"Loaded {len(doc_dataset.corpus)} documents")
    print(f"Loaded {len(doc_dataset.qrels)} qrels")

    if retriever_type == "faiss":
        faiss_index_path = f"{beir_dataset_path}/faiss_index.faiss"
        id_mapping_path = f"{beir_dataset_path}/id_mapping.pkl"
        
        retriever = FaissRetriever(
            faiss_index_path=faiss_index_path,
            embedding_model=embedding_model,
            id_mapping_path=id_mapping_path,
            device=device,
            verbose=True
        )
        nprobe = 64
    else:
        bm25_index_path = f"{beir_dataset_path}/bm25_index"
        id_mapping_path = f"{beir_dataset_path}/id_mapping.pkl"
        
        retriever = Bm25Retriever(
            index_path=bm25_index_path,
            id_mapping_path=id_mapping_path,
            verbose=True
        )
        nprobe = None

    print("\n" + "=" * 80)
    print("Testing retrieval and reward computation")
    print("=" * 80)

    sample_query_ids = list(doc_dataset.qrels.keys())[:10]
    sample_queries = [doc_dataset.queries[qid] for qid in sample_query_ids]

    if retriever_type == "faiss":
        scores, indices = retriever.search(sample_queries, k=k, nprobe=nprobe)
    else:
        scores, indices = retriever.search(sample_queries, k=k)
    
    retrieved_doc_ids_batch = retriever.map_indices_to_ids(indices).tolist()

    ndcg_rewards = doc_dataset.compute_rewards_batch(
        query_uids=sample_query_ids,
        retrieved_doc_ids_batch=retrieved_doc_ids_batch,
        method="ndcg",
        k=k
    )

    recall_rewards = doc_dataset.compute_rewards_batch(
        query_uids=sample_query_ids,
        retrieved_doc_ids_batch=retrieved_doc_ids_batch,
        method="recall",
        k=k
    )

    precision_rewards = doc_dataset.compute_rewards_batch(
        query_uids=sample_query_ids,
        retrieved_doc_ids_batch=retrieved_doc_ids_batch,
        method="precision",
        k=k
    )

    hit_rewards = doc_dataset.compute_rewards_batch(
        query_uids=sample_query_ids,
        retrieved_doc_ids_batch=retrieved_doc_ids_batch,
        method="hit",
        k=k
    )

    print(f"\nResults for {len(sample_query_ids)} queries:")
    print("-" * 80)
    for i, qid in enumerate(sample_query_ids):
        query_text = doc_dataset.queries[qid]
        relevant_docs = doc_dataset.get_relevant_docs(qid)
        retrieved_docs = retrieved_doc_ids_batch[i]
        
        print(f"\nQuery {i+1}: {qid}")
        print(f"  Text: {query_text[:80]}...")
        print(f"  Relevant docs: {len(relevant_docs)}")
        print(f"  Retrieved: {retrieved_docs[:5]}...")
        print(f"  NDCG@{k}:      {ndcg_rewards[i]:.4f}")
        print(f"  Recall@{k}:    {recall_rewards[i]:.4f}")
        print(f"  Precision@{k}: {precision_rewards[i]:.4f}")
        print(f"  Hit@{k}:       {hit_rewards[i]:.4f}")

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Mean NDCG@{k}:      {sum(ndcg_rewards)/len(ndcg_rewards):.4f}")
    print(f"Mean Recall@{k}:    {sum(recall_rewards)/len(recall_rewards):.4f}")
    print(f"Mean Precision@{k}: {sum(precision_rewards)/len(precision_rewards):.4f}")
    print(f"Mean Hit@{k}:       {sum(hit_rewards)/len(hit_rewards):.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test retrieval system")
    parser.add_argument("--beir-dataset", required=True, help="Path to BEIR dataset")
    parser.add_argument("--retriever-type", default="faiss", choices=["faiss", "bm25"], help="Retriever type")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--k", type=int, default=10, help="Top-k retrieval")

    args = parser.parse_args()

    test_retrieval(
        beir_dataset_path=args.beir_dataset,
        retriever_type=args.retriever_type,
        embedding_model=args.embedding_model,
        device=args.device,
        k=args.k
    )


if __name__ == "__main__":
    main()