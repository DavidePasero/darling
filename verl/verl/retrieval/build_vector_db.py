import os

import numpy as np
from dataset_utils import get_ms_marco_dataset, get_scifact_dataset
from engine.vectordatabase import FaissDB
from reward.reward_manager import RewardManager
from tqdm import tqdm

# Global variable to hold the mapping from FAISS Index ID (int) -> MS MARCO Passage ID (int)
# We need this because FAISS returns 0..N, but MS MARCO IDs are specific integers.
ID_MAPPING = []


def build_vector_db(corpus: list[str], model_name="Qwen/Qwen3-Embedding-8B", index_path="msmarco.index"):
    """
    Creates a FaissDB object with standard configuration.
    """

    print("\n🏗️  Initializing Vector DB...")
    db = FaissDB(model_name=model_name)

    if os.path.exists(index_path):
        print(f"   Loading existing index from {index_path}...")
        db.load_index(index_path)
        # NOTE: In a real app, you must also save/load the ID_MAPPING pickle!
        # For this script, we assume we just re-generated corpus_ids or loaded them.
    else:
        print(f"   Building new index for {len(corpus)} documents...")
        # nlist=4096 is standard for MS MARCO
        db.build_index(corpus, nlist=4096, m=32)
        db.save_index(index_path)

    return db


def evaluate(db, queries: list[str], qrels: dict, k=10):
    print(f"\n📊 Starting Evaluation on {len(queries)} queries (k={k})...")

    eval_scorer = RewardManager(method="ndcg", k=k)
    all_ndcg_scores = []
    batch_size = 32

    for i in tqdm(range(0, len(queries), batch_size)):
        batch_q = queries[i : i + batch_size]
        _, indices_batch = db.search(batch_q, k=k)

        for j, indices in enumerate(indices_batch):
            query_text = batch_q[j]
            if query_text not in qrels:
                continue

            gold_pids = qrels[query_text]

            score = eval_scorer.compute_rewards(indices_batch=[indices], labels=gold_pids, id_mapping=ID_MAPPING)[0]
            all_ndcg_scores.append(score)

    print("\n📈 Results:")
    print(f"   Mean NDCG@{k}: {np.mean(all_ndcg_scores):.4f}")


def compute_reward(
    rewritten_queries: list[str],
    labels: list[int],
    db: FaissDB,
    reward_manager: RewardManager,
) -> list[float]:
    """
    The GLUE function: Connects FaissDB (Retrieval) to RewardManager (Scoring).
    """
    # 1. Retrieval Step (Using FaissDB)
    # We use the database to find the 'k' most similar documents for the rewritten queries
    scores, indices = db.search(rewritten_queries, k=reward_manager.k)

    # 2. Scoring Step (Using RewardManager)
    # The manager doesn't need the DB, it just needs the results (indices) and the truth (labels)
    rewards = reward_manager.compute_rewards(
        indices_batch=indices,
        labels=labels,
        id_mapping=ID_MAPPING,
    )

    return rewards


if __name__ == "__main__":
    # 1. Setup Data (Using BeIR/scifact which is Safe/Standard)
    corpus, corpus_ids, queries, qrels = get_scifact_dataset()

    # Set Globals for lookups
    ID_MAPPING = corpus_ids

    # 2. Setup DB
    # Using a different index name so we don't mix up with MS MARCO
    db = build_vector_db(corpus, model_name="Qwen/Qwen3-Embedding-0.6B", index_path="scifact.index")

    # 3. Setup Reward Manager
    reward_mgr = RewardManager(method="ndcg", k=10)

    # 4. RL Loop Simulation
    print("\n🧪 Testing Reward Signal (RL Step)...")
    if queries:
        original_q = queries[0]
        if original_q in qrels:
            labels = qrels[original_q]

            rewrites = [original_q, original_q + " details", "completely irrelevant garbage"]

            rewards = compute_reward(rewrites, labels, db, reward_mgr)

            for r_q, score in zip(rewrites, rewards):
                print(f"   Rewrite: '{r_q}' -> Reward: {score:.4f}")
        else:
            print(f"Skipping RL test: No qrels for first query '{original_q}'")

    # 5. Full Evaluation
    evaluate(db, queries, qrels)
