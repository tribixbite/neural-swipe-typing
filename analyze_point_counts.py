#!/usr/bin/env python3
"""
Analyze point count distribution in the optimized dataset.
"""

import json
from collections import Counter

def analyze_point_counts(file_path):
    """Count traces by number of points."""
    
    point_count_distribution = Counter()
    more_than_150 = 0
    more_than_250 = 0
    total_traces = 0
    max_points = 0
    
    print(f"Analyzing {file_path}...")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                print(f"  Processed {i:,} traces...")
            
            data = json.loads(line)
            num_points = len(data['points'])
            
            total_traces += 1
            point_count_distribution[num_points] += 1
            
            if num_points > 150:
                more_than_150 += 1
            if num_points > 250:
                more_than_250 += 1
            
            if num_points > max_points:
                max_points = num_points
    
    print(f"\nTotal traces analyzed: {total_traces:,}")
    print(f"\n--- RESULTS ---")
    print(f"Traces with > 250 points: {more_than_250:,} ({100*more_than_250/total_traces:.2f}%)")
    print(f"Traces with > 150 points: {more_than_150:,} ({100*more_than_150/total_traces:.2f}%)")
    print(f"Maximum points in any trace: {max_points}")
    
    # Show distribution around these thresholds
    print("\nPoint count distribution (140-260 range):")
    for points in range(140, min(261, max_points + 1)):
        count = point_count_distribution[points]
        if count > 0:
            print(f"  {points:3d} points: {count:6,} traces")
    
    # Show the tail of the distribution
    if max_points > 260:
        print(f"\nTraces with 260+ points:")
        for points in sorted([k for k in point_count_distribution.keys() if k > 260]):
            count = point_count_distribution[points]
            print(f"  {points:3d} points: {count:6,} traces")

if __name__ == "__main__":
    analyze_point_counts("data/futo/train_optimized.jsonl")