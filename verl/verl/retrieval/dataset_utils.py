import os
from functools import partial

from datasets import load_dataset
from tqdm import tqdm


def _format_batch(batch, docid_key_name="docid"):
    return {
        "formatted_text": [f"{t} {txt}".strip() for t, txt in zip(batch["title"], batch["text"])],
        "docid_int": [int(did) for did in batch[docid_key_name]],
    }


def _format_query_batch(batch):
    return {
        "query_text": batch["query"],
        # Extract list of docids for each query in the batch
        "relevant_pids": [[int(p["docid"]) for p in passages] for passages in batch["positive_passages"]],
    }


def get_ms_marco_dataset(split="train", sample_size=None):
    """
    Parses MS MARCO to create a retrieval-ready dataset.

    Returns:
        corpus (List[str]): All 8.8M passages for the Vector DB.
        corpus_ids (List[int]): Corresponding IDs for the passages.
        queries (List[str]): The evaluation queries.
        qrels (Dict[str, List[int]]): Ground truth {query_text: [relevant_doc_ids]}.
    """
    print("📦 Loading MS MARCO Corpus (Tevatron/msmarco-passage-corpus)...")
    corpus_data = load_dataset("Tevatron/msmarco-passage-corpus", split="train")
    print(len(corpus_data))
    if sample_size:
        print(f"⚠️  Sampling first {sample_size} documents for testing...")
        corpus_data = corpus_data.select(range(sample_size))

    print("   Processing corpus text...")

    # num_proc uses available CPU cores to process chunks in parallel
    processed_corpus = corpus_data.map(
        partial(_format_batch, docid_key_name="docid"),
        batched=True,
        remove_columns=corpus_data.column_names,
        num_proc=min(os.cpu_count(), 8),
        desc="Formatting Corpus",  # Adds progress bar
    )

    # Accessing columns from the processed HF dataset is efficient
    corpus = processed_corpus["formatted_text"]
    corpus_ids = processed_corpus["docid_int"]

    print(f"📦 Loading MS MARCO Queries & Qrels ({split})...")
    q_data = load_dataset("Tevatron/msmarco-passage", split=split)

    if sample_size:
        q_data = q_data.select(range(min(len(q_data), 100)))

    print("   Processing queries and labels...")

    # Re-enabled multiprocessing for queries as requested
    processed_queries = q_data.map(
        _format_query_batch,
        batched=True,
        remove_columns=q_data.column_names,
        num_proc=min(os.cpu_count(), 8),
        desc="Processing Queries",
    )

    queries = processed_queries["query_text"]
    # Zip queries with their relevant pids to create the qrels dictionary
    qrels = dict(zip(queries, processed_queries["relevant_pids"]))

    return corpus, corpus_ids, queries, qrels


def get_scifact_dataset(split="train", sample_size=None):
    """
    Parses SciFact using the BeIR/scifact dataset (standard format, no remote code needed).
    BeIR splits data into 'corpus', 'queries', and 'qrels'.
    """
    print("📦 Loading SciFact Corpus (BeIR/scifact)...")

    # Load Corpus (Docs)
    # BeIR datasets don't use scripts, so they are safe (Parquet/JSONL)
    corpus_data = load_dataset("BeIR/scifact", "corpus", split="corpus")

    if sample_size:
        print(f"⚠️  Sampling first {sample_size} documents for testing...")
        corpus_data = corpus_data.select(range(sample_size))

    print("   Processing corpus text...")

    processed_corpus = corpus_data.map(
        partial(_format_batch, docid_key_name="_id"),
        batched=True,
        remove_columns=corpus_data.column_names,
        desc="Formatting Corpus",
        num_proc=min(os.cpu_count(), 4),
    )

    corpus = processed_corpus["formatted_text"]
    corpus_ids = processed_corpus["docid_int"]

    # Load Queries
    print("📦 Loading SciFact Queries (BeIR/scifact)...")
    q_data = load_dataset("BeIR/scifact", "queries", split="queries")
    print(f"   Loaded {len(q_data)} queries.")

    # Load Qrels (Ground Truth)
    # BeIR stores qrels in a separate config
    print("📦 Loading SciFact Qrels...")
    qrels_data = load_dataset("BeIR/scifact-qrels", split=split)
    print(f"   Loaded {len(qrels_data)} qrel entries.")

    # Build a quick lookup for qrels: query_id (str) -> list of relevant doc_ids (int)
    # BeIR qrels columns: 'query-id', 'corpus-id', 'score'
    qrels_lookup = {}
    for item in tqdm(qrels_data, desc="Indexing Qrels"):
        qid = int(item["query-id"])
        did = item["corpus-id"]
        if qid not in qrels_lookup:
            qrels_lookup[qid] = []
        qrels_lookup[qid].append(did)

    print("   Processing queries and labels...")
    queries = []
    qrels = {}  # Map query_text -> list of relevant doc_ids

    # We need to map query_id -> query_text to build the final qrels dict
    # We iterate through the queries dataset
    if sample_size:
        q_data = q_data.select(range(min(len(q_data), sample_size)))

    for item in tqdm(q_data, desc="Processing Queries"):
        qid = int(item["_id"])
        query_text = item["text"]

        # Only add queries that have ground truth labels in this split
        if qid in qrels_lookup:
            queries.append(query_text)
            qrels[query_text] = qrels_lookup[qid]

    return corpus, corpus_ids, queries, qrels
