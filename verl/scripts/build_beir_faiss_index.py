#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.retrieval.engine.index_builder import IndexBuilder


def main():
    parser = argparse.ArgumentParser(description="Build FAISS Index from BEIR Dataset")
    parser.add_argument(
        "--beir-dataset",
        default="datasets/fiqa",
        help="Path to BEIR dataset directory"
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace embedding model"
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=512,
        help="Number of IVF clusters"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=32,
        help="Number of PQ subquantizers"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Encoding batch size"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device for encoding"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (defaults to same as beir-dataset)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir if args.output_dir else args.beir_dataset)

    builder = IndexBuilder(
        index_type="faiss",
        output_dir=output_dir,
        verbose=True
    )

    builder.build_from_beir(
        beir_dataset_path=args.beir_dataset,
        embedding_model=args.embedding_model,
        faiss_nlist=args.nlist,
        faiss_m=args.m,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()