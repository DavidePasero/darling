#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.retrieval.engine.index_builder import IndexBuilder


def main():
    parser = argparse.ArgumentParser(description="Build BM25 Index from BEIR Dataset using Pyserini")
    parser.add_argument(
        "--beir-dataset",
        default="datasets/fiqa",
        help="Path to BEIR dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for index (defaults to beir-dataset/bm25_index)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads for indexing"
    )

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.beir_dataset) / "bm25_index"

    builder = IndexBuilder(
        index_type="bm25",
        output_dir=output_dir,
        verbose=True
    )

    builder.build_from_beir(
        beir_dataset_path=args.beir_dataset,
        bm25_threads=args.threads
    )


if __name__ == "__main__":
    main()