import time
import requests
import numpy as np
import random
import string
from typing import List


class RemoteRerankerClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.score_url = f"{base_url}/score"

    def compute_reranker_score_batch(
        self,
        query_uids: List[str],
        retrieved_doc_ids_batch: List[List[str]],
        queries_dict: dict,
        corpus_dict: dict,
        k: int = 10,
    ) -> List[float]:
        # 1. Prepare pairs
        all_pairs_q = []
        all_pairs_d = []
        doc_counts = []

        for query_id, docs in zip(query_uids, retrieved_doc_ids_batch):
            docs_k = docs[:k]
            query_text = queries_dict[query_id]

            for doc in docs_k:
                all_pairs_q.append(query_text)
                all_pairs_d.append(corpus_dict[doc])

            doc_counts.append(len(docs_k))

        if not all_pairs_q:
            return [0.0] * len(query_uids)

        # 2. Batch Inference
        batch_size = 128  # Safe batch size for HTTP/GPU
        all_scores = []

        print(f"DEBUG: Sending {len(all_pairs_q)} pairs to server in batches of {batch_size}...")
        t_start = time.time()

        for i in range(0, len(all_pairs_q), batch_size):
            chunk_q = all_pairs_q[i : i + batch_size]
            chunk_d = all_pairs_d[i : i + batch_size]

            payload = {"text_1": chunk_q, "text_2": chunk_d}

            try:
                resp = requests.post(self.score_url, json=payload)
                resp.raise_for_status()
                data = resp.json().get("data", [])

                # Extract scores and ensure they are floats
                chunk_scores = [float(item["score"]) for item in data]

                # Safety check: if server returns fewer scores than requested
                if len(chunk_scores) != len(chunk_q):
                    print(f"Warning: Mismatch in response length. Expected {len(chunk_q)}, got {len(chunk_scores)}")
                    chunk_scores += [0.0] * (len(chunk_q) - len(chunk_scores))

                all_scores.extend(chunk_scores)
            except Exception as e:
                print(f"Server Error at batch {i}: {e}")
                all_scores.extend([0.0] * len(chunk_q))

        t_end = time.time()
        total_time = t_end - t_start
        print(f"DEBUG: Inference finished. Total time: {total_time:.2f}s")
        print(f"DEBUG: Throughput: {len(all_pairs_q) / total_time:.2f} pairs/sec")

        # 3. Reconstruct Rewards
        rewards = []
        start_idx = 0
        discounts_cache = 1.0 / np.log2(np.arange(k) + 2)

        for count in doc_counts:
            if count == 0:
                rewards.append(0.0)
                continue

            scores = np.array(all_scores[start_idx : start_idx + count])

            # Compute discounted sum (DCG-like)
            discounted_scores = scores * discounts_cache[:count]
            rewards.append(float(np.sum(discounted_scores)))
            start_idx += count

        return rewards


def generate_random_text(length=50):
    """Generates a random string of fixed words to simulate real token load."""
    words = ["machine", "learning", "data", "science", "python", "model", "training", "inference", "gpu", "latency"]
    return " ".join(random.choices(words, k=length))


if __name__ == "__main__":
    # --- Configuration ---
    NUM_QUERIES = 1000  # Increased from 10 to 1000
    TOP_K = 10  # Docs per query
    SERVER_URL = "http://localhost:8000/v1"

    print(f"--- Generating Synthetic Data ({NUM_QUERIES} queries x {TOP_K} docs)... ---")

    # Generate Queries (length ~20 words)
    queries = {f"q{i}": f"Query {i}: {generate_random_text(20)}" for i in range(NUM_QUERIES)}

    # Generate Corpus (length ~100 words to mimic real passages)
    corpus = {}
    retrieved_doc_ids_batch = []

    for i in range(NUM_QUERIES):
        doc_ids_for_this_query = []
        for j in range(TOP_K):
            doc_id = f"d{i}_{j}"
            # Unique doc content to prevent aggressive caching (if any)
            corpus[doc_id] = f"Doc {doc_id}: {generate_random_text(100)}"
            doc_ids_for_this_query.append(doc_id)
        retrieved_doc_ids_batch.append(doc_ids_for_this_query)

    query_uids = list(queries.keys())

    print(f"--- Data Generation Complete. Total Pairs: {NUM_QUERIES * TOP_K} ---")

    # --- Initialize Client ---
    client = RemoteRerankerClient(base_url=SERVER_URL)

    # --- Benchmark Run ---
    print("\n--- Starting Benchmark ---")

    start_time = time.time()

    rewards = client.compute_reranker_score_batch(query_uids, retrieved_doc_ids_batch, queries, corpus, k=TOP_K)

    total_time = time.time() - start_time
    print(f"\n===========================================")
    print(f"Total Benchmark Time: {total_time:.4f}s")
    print(f"Pairs Processed:      {NUM_QUERIES * TOP_K}")
    print(f"Avg Latency per Pair: {(total_time / (NUM_QUERIES * TOP_K)) * 1000:.2f} ms")
    print(f"Sample Rewards (First 5): {rewards[:5]}")
    print(f"===========================================")
