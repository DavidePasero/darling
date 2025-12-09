#!/usr/bin/env python3
"""
Build FAISS Index from BEIR Dataset

This script:
1. Loads corpus from BEIR format (corpus.jsonl via BeirAdapter)
2. Encodes documents using embedding model
3. Builds FAISS IVF index
4. Saves index and ID mapping

Usage:
    python build_beir_faiss_index.py \
        --beir-dataset datasets/fiqa \
        --embedding-model Qwen/Qwen3-Embedding-0.6B \
        --nlist 4096
"""

import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.retrieval.engine.document_dataset import BeirAdapter


def load_beir_corpus(beir_dataset_path: str):
    """
    Load corpus from BEIR dataset using BeirAdapter.

    Args:
        beir_dataset_path: Path to BEIR dataset directory

    Returns:
        corpus_texts: List of document texts
        corpus_ids: List of document IDs
    """
    print(f"Loading BEIR corpus from: {beir_dataset_path}")

    adapter = BeirAdapter(data_path=beir_dataset_path, split="train")
    dataset = adapter.to_unified()

    corpus_ids = list(dataset.corpus.keys())
    corpus_texts = [dataset.corpus[doc_id] for doc_id in corpus_ids]

    print(f"Loaded {len(corpus_texts)} documents")
    return corpus_texts, corpus_ids


def encode_corpus(corpus_texts, embedding_model_name: str, batch_size: int = 128, device: str = "cuda"):
    """
    Encode corpus using sentence transformer.

    Args:
        corpus_texts: List of document texts
        embedding_model_name: HuggingFace embedding model
        batch_size: Encoding batch size
        device: Device for encoding

    Returns:
        embeddings: numpy array of shape (num_docs, embedding_dim)
    """
    print(f"\nLoading embedding model: {embedding_model_name}")

    model = SentenceTransformer(
        embedding_model_name,
        device=device,
        trust_remote_code=True
    )
    model.eval()

    print(f"Encoding {len(corpus_texts)} documents with batch size {batch_size}...")

    embeddings = []
    for i in tqdm(range(0, len(corpus_texts), batch_size)):
        batch = corpus_texts[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings).astype('float32')
    print(f"Encoded embeddings shape: {embeddings.shape}")

    return embeddings


def build_faiss_index(embeddings, nlist: int = 4096, m: int = 32, use_gpu: bool = True):
    """
    Build FAISS IVF-PQ index.

    Args:
        embeddings: numpy array of embeddings
        nlist: Number of IVF clusters
        m: Number of PQ subquantizers
        use_gpu: Use GPU for training

    Returns:
        index: FAISS index
    """
    dimension = embeddings.shape[1]
    num_docs = embeddings.shape[0]

    print(f"\nBuilding FAISS index:")
    print(f"  Dimension: {dimension}")
    print(f"  Num docs: {num_docs}")
    print(f"  IVF nlist: {nlist}")
    print(f"  PQ m: {m}")

    # Create IVF-PQ index
    quantizer = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)

    # Train index
    print("Training FAISS index...")
    if use_gpu and torch.cuda.is_available():
        print("Using GPU for training")
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.train(embeddings)
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        print("Using CPU for training")
        index.train(embeddings)

    # Add vectors
    print("Adding vectors to index...")
    index.add(embeddings)

    print(f"Index built with {index.ntotal} vectors")

    return index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS Index from BEIR Dataset")
    parser.add_argument(
        "--beir-dataset",
        default="datasets/fiqa",
        help="Path to BEIR dataset directory (contains corpus.jsonl, queries.jsonl, qrels/)"
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace embedding model (default: Qwen/Qwen3-Embedding-0.6B)"
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="Number of IVF clusters (default: 4096)"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=32,
        help="Number of PQ subquantizers (default: 32)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Encoding batch size (default: 128)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for encoding (default: cuda)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (defaults to same as beir-dataset)"
    )

    args = parser.parse_args()

    # Default output dir to beir dataset dir
    output_dir = Path(args.output_dir if args.output_dir else args.beir_dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "faiss_index.faiss"
    id_mapping_path = output_dir / "id_mapping.pkl"

    print("=" * 80)
    print("BUILD BEIR FAISS INDEX")
    print("=" * 80)

    # Step 1: Load corpus
    corpus_texts, corpus_ids = load_beir_corpus(args.beir_dataset)

    # Step 2: Encode corpus
    embeddings = encode_corpus(
        corpus_texts,
        args.embedding_model,
        batch_size=args.batch_size,
        device=args.device
    )

    # Step 3: Build FAISS index
    use_gpu = args.device == "cuda" and torch.cuda.is_available()
    index = build_faiss_index(
        embeddings,
        nlist=args.nlist,
        m=args.m,
        use_gpu=use_gpu
    )

    # Step 4: Save index and mapping
    print(f"\nSaving index to: {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"Saving ID mapping to: {id_mapping_path}")
    with open(id_mapping_path, 'wb') as f:
        pickle.dump(corpus_ids, f)

    print("\n" + "=" * 80)
    print("✅ FAISS Index Build Complete!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  Index: {index_path}")
    print(f"  ID Mapping: {id_mapping_path}")
    print(f"\nYou can now test with:")
    print(f"  python scripts/test_retrieval_system.py \\")
    print(f"    --beir-dataset {output_dir} \\")
    print(f"    --embedding-model {args.embedding_model}")
    print("=" * 80)


if __name__ == "__main__":
    main()