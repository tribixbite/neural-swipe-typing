#!/usr/bin/env python3
"""Create small subset of filtered English dataset for testing."""

import json
import os
from pathlib import Path

def create_subset(input_path, output_path, n_samples=1000):
    """Create a small subset of the dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for i, line in enumerate(infile):
            if i >= n_samples:
                break
            outfile.write(line)
    
    print(f"Created subset with {min(i, n_samples)} samples at {output_path}")

if __name__ == "__main__":
    data_dir = Path("data/data_preprocessed")
    
    # Create small subsets for testing
    create_subset(
        data_dir / "english_filtered_train.jsonl",
        data_dir / "english_subset_train.jsonl",
        n_samples=800
    )
    
    create_subset(
        data_dir / "english_filtered_valid.jsonl", 
        data_dir / "english_subset_valid.jsonl",
        n_samples=100
    )
    
    create_subset(
        data_dir / "english_filtered_test.jsonl",
        data_dir / "english_subset_test.jsonl", 
        n_samples=100
    )