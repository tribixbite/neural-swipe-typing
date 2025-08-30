#!/usr/bin/env python3
"""
Process synthetic trace data to combine with existing real data.
1. Combine all synthetic trace files into one file
2. Convert synthetic format to match existing data format  
3. Combine and shuffle both datasets
4. Split into train/val/test files
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Set
from glob import glob

def convert_synthetic_to_real_format(synthetic_entry: Dict) -> Dict:
    """
    Convert synthetic trace format to real data format.
    
    Synthetic format:
    {
      "word_seq": {"time": [...], "x": [...], "y": [...]},
      "word": "...",
      "std_dev": "...",
      "timestamp": ...
    }
    
    Real format:
    {
      "curve": {"x": [...], "y": [...], "t": [...], "grid_name": "qwerty_english"},
      "word": "..."
    }
    """
    word_seq = synthetic_entry["word_seq"]
    
    # Convert normalized coordinates to integer coordinates
    # Multiply x by 360, y by 215 (as specified in requirements)
    x_coords = [int(x * 360) for x in word_seq["x"]]
    y_coords = [int(y * 215) for y in word_seq["y"]]
    
    # Convert time to integer milliseconds (assuming time is in seconds)
    # Start from 0 for first timestamp
    t_coords = []
    if word_seq["time"]:
        start_time = word_seq["time"][0]
        t_coords = [int((t - start_time) * 1000) for t in word_seq["time"]]
    
    converted_entry = {
        "curve": {
            "x": x_coords,
            "y": y_coords,
            "t": t_coords,
            "grid_name": "qwerty_english"
        },
        "word": synthetic_entry["word"]
    }
    
    return converted_entry

def combine_synthetic_files(synthetic_dir: str, output_file: str) -> int:
    """Combine all synthetic trace batch files into one file"""
    print(f"Combining synthetic files from {synthetic_dir}...")
    
    batch_files = sorted(glob(os.path.join(synthetic_dir, "synthetic_traces_batch_*.jsonl")))
    total_traces = 0
    
    with open(output_file, 'w') as outf:
        for batch_file in batch_files:
            print(f"Processing {batch_file}...")
            with open(batch_file, 'r') as inf:
                for line in inf:
                    line = line.strip()
                    if line:
                        outf.write(line + '\n')
                        total_traces += 1
    
    print(f"Combined {total_traces} synthetic traces into {output_file}")
    return total_traces

def convert_synthetic_format(input_file: str, output_file: str) -> int:
    """Convert synthetic traces from API format to real data format"""
    print(f"Converting synthetic format from {input_file} to {output_file}...")
    
    converted_count = 0
    
    with open(input_file, 'r') as inf, open(output_file, 'w') as outf:
        for line in inf:
            line = line.strip()
            if line:
                synthetic_entry = json.loads(line)
                converted_entry = convert_synthetic_to_real_format(synthetic_entry)
                outf.write(json.dumps(converted_entry) + '\n')
                converted_count += 1
    
    print(f"Converted {converted_count} synthetic traces")
    return converted_count

def load_existing_data(file_path: str) -> List[Dict]:
    """Load existing real data"""
    print(f"Loading existing data from {file_path}...")
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} existing traces")
    return data

def combine_and_shuffle_datasets(existing_files: List[str], synthetic_file: str, output_file: str) -> int:
    """Combine existing and synthetic datasets, then shuffle"""
    print("Combining and shuffling datasets...")
    
    all_data = []
    
    # Load existing data from all files
    for file_path in existing_files:
        if os.path.exists(file_path):
            all_data.extend(load_existing_data(file_path))
    
    # Load synthetic data
    synthetic_data = load_existing_data(synthetic_file)
    all_data.extend(synthetic_data)
    
    print(f"Total combined data: {len(all_data)} traces")
    
    # Shuffle the combined dataset
    random.shuffle(all_data)
    
    # Write shuffled data
    with open(output_file, 'w') as f:
        for entry in all_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Wrote shuffled combined dataset to {output_file}")
    return len(all_data)

def split_dataset(input_file: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split combined dataset into train/val/test files"""
    print(f"Splitting dataset from {input_file}...")
    
    # Load all data
    data = load_existing_data(input_file)
    total_count = len(data)
    
    # Calculate split sizes
    train_size = int(total_count * train_ratio)
    val_size = int(total_count * val_ratio)
    test_size = total_count - train_size - val_size
    
    print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Write split files
    splits = [
        ('train', train_data),
        ('val', val_data), 
        ('test', test_data)
    ]
    
    for split_name, split_data in splits:
        output_file = os.path.join(output_dir, f"combined_english_swipes_{split_name}.jsonl")
        with open(output_file, 'w') as f:
            for entry in split_data:
                f.write(json.dumps(entry) + '\n')
        print(f"Wrote {len(split_data)} {split_name} samples to {output_file}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # File paths
    synthetic_dir = "data/synthetic_traces"
    output_dir = "data/combined_dataset"
    
    # Intermediate files
    combined_synthetic = os.path.join(output_dir, "all_synthetic_traces.jsonl")
    converted_synthetic = os.path.join(output_dir, "converted_synthetic_traces.jsonl")
    final_combined = os.path.join(output_dir, "all_combined_traces.jsonl")
    
    # Existing data files
    existing_files = [
        "data/raw_converted_english_swipes_train.jsonl",
        "data/raw_converted_english_swipes_val.jsonl", 
        "data/raw_converted_english_swipes_test.jsonl"
    ]
    
    print("=== Processing Synthetic Dataset ===")
    
    # Step 1: Combine all synthetic files
    total_synthetic = combine_synthetic_files(synthetic_dir, combined_synthetic)
    
    # Step 2: Convert synthetic format to match real data
    converted_count = convert_synthetic_format(combined_synthetic, converted_synthetic)
    
    # Step 3: Combine with existing data and shuffle
    total_combined = combine_and_shuffle_datasets(existing_files, converted_synthetic, final_combined)
    
    # Step 4: Split into train/val/test
    split_dataset(final_combined, output_dir)
    
    print("\n=== Summary ===")
    print(f"Synthetic traces processed: {total_synthetic}")
    print(f"Total combined traces: {total_combined}")
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    print(f"- {output_dir}/combined_english_swipes_train.jsonl")
    print(f"- {output_dir}/combined_english_swipes_val.jsonl")
    print(f"- {output_dir}/combined_english_swipes_test.jsonl")

if __name__ == "__main__":
    main()