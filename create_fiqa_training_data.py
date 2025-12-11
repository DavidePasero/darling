#!/usr/bin/env python3
"""
Create training data (parquet files) from FiQA BEIR dataset.
This ensures UIDs match the BEIR query IDs.
"""
import json
import pandas as pd
from pathlib import Path

def create_fiqa_training_data(
    fiqa_path="/home/lukas/Projects/darling/verl/datasets/fiqa",
    output_dir="/home/lukas/Projects/darling/verl/datasets/fiqa",
    split="train"
):
    """
    Create training parquet from FiQA BEIR dataset.
    
    Args:
        fiqa_path: Path to FiQA dataset directory
        output_dir: Where to save the parquet files
        split: Which split to use ('train', 'dev', or 'test')
    """
    
    fiqa_path = Path(fiqa_path)
    output_dir = Path(output_dir)
    
    print("=" * 80)
    print(f"Creating FiQA Training Data - {split.upper()} split")
    print("=" * 80)
    
    # Load queries
    queries_file = fiqa_path / "queries.jsonl"
    qrels_file = fiqa_path / "qrels" / f"{split}.tsv"
    
    print(f"\n1. Loading queries from: {queries_file}")
    queries = {}
    with open(queries_file, "r") as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']
    
    print(f"   ✓ Loaded {len(queries)} queries")
    
    # Load qrels to filter queries with labels
    print(f"\n2. Loading qrels from: {qrels_file}")
    valid_qids = set()
    with open(qrels_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                valid_qids.add(parts[0])
    
    print(f"   ✓ Found {len(valid_qids)} queries with relevance labels")
    
    # Create training data
    print(f"\n3. Creating training data...")
    data = []
    for qid in valid_qids:
        if qid in queries:
            data.append({
                'uid': qid,  # CRITICAL: This matches BEIR query IDs
                'prompt': queries[qid],
                'data_source': 'fiqa_retrieval'
            })
    
    print(f"   ✓ Created {len(data)} training samples")
    
    # Save as parquet
    df = pd.DataFrame(data)
    output_file = output_dir / f"fiqa_{split}.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"\n4. Saved to: {output_file}")
    
    # Show statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"{'='*80}")
    print(f"Total queries in FiQA:           {len(queries)}")
    print(f"Queries with labels ({split}):   {len(valid_qids)}")
    print(f"Training samples created:        {len(data)}")
    print(f"Output file:                     {output_file}")
    print(f"{'='*80}")
    
    # Show samples
    print(f"\n📋 Sample training data:")
    for i, row in df.head(3).iterrows():
        print(f"\n  Sample {i+1}:")
        print(f"    UID:    {row['uid']}")
        print(f"    Prompt: {row['prompt'][:80]}...")
    
    print(f"\n✅ SUCCESS! Training data created.")
    print(f"\nNext steps:")
    print(f"  1. Use this file in your training script:")
    print(f"     data.train_files={output_file}")
    print(f"  2. Make sure your BEIR dataset path points to: {fiqa_path}")
    print(f"     +reward_model.beir_dataset_path={fiqa_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create FiQA training data")
    parser.add_argument(
        "--fiqa_path",
        default="/home/lukas/Projects/darling/verl/datasets/fiqa",
        help="Path to FiQA dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        default="/home/lukas/Projects/darling/verl/datasets/fiqa",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "dev", "test"],
        help="Which split to create"
    )
    
    args = parser.parse_args()
    
    # Create training data
    df_train = create_fiqa_training_data(
        fiqa_path=args.fiqa_path,
        output_dir=args.output_dir,
        split=args.split
    )
    
    # Optionally create dev/test splits too
    if args.split == "train":
        print(f"\n\n{'='*80}")
        print("Creating DEV split as well...")
        print(f"{'='*80}")
        df_dev = create_fiqa_training_data(
            fiqa_path=args.fiqa_path,
            output_dir=args.output_dir,
            split="dev"
        )
