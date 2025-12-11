#!/usr/bin/env python3
"""
Quick diagnostic script to check if training data UIDs match BEIR dataset query IDs.
Updated for FiQA dataset.
"""
import json
import pandas as pd
import sys
from pathlib import Path

def check_uid_match(parquet_file, beir_queries_file):
    """Check if UIDs in parquet match query IDs in BEIR dataset."""
    
    print("=" * 80)
    print("UID MISMATCH DIAGNOSTIC TOOL - FiQA Dataset")
    print("=" * 80)
    
    # Load training data UIDs
    print(f"\n1. Loading training data from: {parquet_file}")
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"❌ ERROR: Could not load parquet file: {e}")
        return False
    
    print(f"   Columns in parquet: {df.columns.tolist()}")
    
    if 'uid' not in df.columns:
        print(f"❌ ERROR: 'uid' column not found in parquet file!")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    
    training_uids = set(df['uid'].unique())
    print(f"✓ Loaded {len(training_uids)} unique UIDs from training data")
    print(f"  Sample UIDs:")
    for uid in list(training_uids)[:5]:
        print(f"    - {uid}")
    
    # Load BEIR query IDs with text
    print(f"\n2. Loading BEIR queries from: {beir_queries_file}")
    try:
        beir_queries = {}
        with open(beir_queries_file, "r") as f:
            for line in f:
                q = json.loads(line)
                beir_queries[q['_id']] = q['text']
    except Exception as e:
        print(f"❌ ERROR: Could not load BEIR queries: {e}")
        return False
    
    beir_qids = set(beir_queries.keys())
    print(f"✓ Loaded {len(beir_qids)} query IDs from BEIR dataset")
    print(f"  Sample query IDs with text:")
    for qid in list(beir_qids)[:5]:
        print(f"    - {qid}: {beir_queries[qid][:60]}...")
    
    # Check overlap
    print(f"\n3. Checking UID overlap...")
    overlap = training_uids & beir_qids
    missing = training_uids - beir_qids
    extra = beir_qids - training_uids
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}")
    print(f"Training data UIDs:     {len(training_uids)}")
    print(f"BEIR query IDs:         {len(beir_qids)}")
    print(f"Matching UIDs:          {len(overlap)} ({100*len(overlap)/len(training_uids):.1f}%)")
    print(f"Missing from BEIR:      {len(missing)}")
    print(f"Extra in BEIR:          {len(extra)}")
    print(f"{'='*80}")
    
    if len(missing) > 0:
        print(f"\n❌ PROBLEM: {len(missing)} UIDs from training data are NOT in BEIR dataset!")
        print(f"\nMissing UIDs (first 10):")
        for uid in list(missing)[:10]:
            # Try to show the prompt from training data
            sample = df[df['uid'] == uid].iloc[0]
            prompt = sample.get('prompt', 'N/A')[:60] if 'prompt' in df.columns else 'N/A'
            print(f"  - {uid}: {prompt}...")
        
        print(f"\n💡 SOLUTIONS:")
        print(f"  1. Create training data from BEIR queries (recommended)")
        print(f"     python create_fiqa_training_data.py")
        print(f"  2. Use a different BEIR dataset that matches your training data")
        print(f"  3. Filter your training data to only include matching UIDs")
        return False
    else:
        print(f"\n✅ SUCCESS: All training UIDs match BEIR query IDs!")
        print(f"   Your data is correctly configured.")
        
        # Show some matching examples
        print(f"\n📋 Sample matching queries:")
        for uid in list(overlap)[:3]:
            sample = df[df['uid'] == uid].iloc[0]
            prompt = sample.get('prompt', 'N/A')[:60] if 'prompt' in df.columns else 'N/A'
            beir_text = beir_queries[uid][:60]
            print(f"\n  UID: {uid}")
            print(f"  Training prompt: {prompt}...")
            print(f"  BEIR query:      {beir_text}...")
        
        return True


if __name__ == "__main__":
    # Default paths for FiQA dataset
    default_beir_queries = "/home/lukas/Projects/darling/verl/datasets/fiqa/queries.jsonl"
    
    if len(sys.argv) == 1:
        print("Usage: python check_uid_match.py <parquet_file> [beir_queries.jsonl]")
        print("\nExample:")
        print(f"  python check_uid_match.py /path/to/train.parquet")
        print(f"  python check_uid_match.py /path/to/train.parquet {default_beir_queries}")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    beir_queries = sys.argv[2] if len(sys.argv) > 2 else default_beir_queries
    
    if not Path(parquet_file).exists():
        print(f"❌ ERROR: Parquet file not found: {parquet_file}")
        sys.exit(1)
    
    if not Path(beir_queries).exists():
        print(f"❌ ERROR: BEIR queries file not found: {beir_queries}")
        print(f"   Using default: {default_beir_queries}")
        if not Path(default_beir_queries).exists():
            print(f"   Default also not found!")
            sys.exit(1)
        beir_queries = default_beir_queries
    
    success = check_uid_match(parquet_file, beir_queries)
    sys.exit(0 if success else 1)

