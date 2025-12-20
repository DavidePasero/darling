#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.retrieval.engine.index_builder import IndexBuilder


def main():
    parser = argparse.ArgumentParser(description="Build Search Index (FAISS or BM25) from BEIR Dataset")
    
    # Mode selection
    parser.add_argument(
        "--index-type",
        type=str,
        default="faiss",
        choices=["faiss", "bm25"],
        help="Type of index to build"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to include in the index (prioritizing relevant ones)"
    )

    # General args
    parser.add_argument("--beir-dataset", default="datasets/msmarco", help="Path to BEIR dataset directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    
    # FAISS specific (H100 Defaults)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B", help="HF embedding model")
    parser.add_argument("--nlist", type=int, default=4096, help="IVF clusters (Recommended: 4096)")
    parser.add_argument("--m", type=int, default=32, help="PQ subquantizers")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (High for H100)")
    parser.add_argument("--device", default="cuda", help="Device")
    
    # BM25 specific
    parser.add_argument("--bm25-threads", type=int, default=12, help="Threads for Pyserini")

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        out_path = Path(args.output_dir)
    else:
        # Default: create subfolder inside the dataset folder
        base = Path(args.beir_dataset)
        out_path = base / f"{args.index_type}_index"

    builder = IndexBuilder(
        index_type=args.index_type,
        output_dir=out_path,
        verbose=True
    )

    builder.build_from_beir(
        beir_dataset_path=args.beir_dataset,
        embedding_model=args.embedding_model,
        faiss_nlist=args.nlist,
        faiss_m=args.m,
        batch_size=args.batch_size,
        bm25_threads=args.bm25_threads,
        device=args.device,
        max_docs=args.max_docs,
    )

if __name__ == "__main__":
    main()