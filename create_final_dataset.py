#!/usr/bin/env python3
"""
Create final training dataset filtered to ≤200 points per trace.
"""

import json
from collections import Counter
from pathlib import Path

def create_final_dataset(input_file, output_file, max_points=200):
    """Filter dataset to traces with ≤ max_points."""
    
    # Statistics
    total_input = 0
    total_output = 0
    filtered_too_long = 0
    
    # Track word distribution
    word_counts = Counter()
    point_distribution = Counter()
    
    print(f"Filtering {input_file} to traces with ≤{max_points} points...")
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for i, line in enumerate(infile):
            total_input += 1
            
            if i % 100000 == 0 and i > 0:
                print(f"  Processed {i:,} traces, kept {total_output:,} ({100*total_output/i:.1f}%)")
            
            data = json.loads(line)
            num_points = len(data['points'])
            
            # Filter by point count
            if num_points > max_points:
                filtered_too_long += 1
                continue
            
            # Write to output
            outfile.write(line)  # Write original line to preserve formatting
            total_output += 1
            
            # Track statistics
            word_counts[data['word']] += 1
            point_distribution[num_points] += 1
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FINAL DATASET CREATED")
    print("="*60)
    
    print(f"\nInput traces: {total_input:,}")
    print(f"Output traces: {total_output:,} ({100*total_output/total_input:.1f}%)")
    print(f"Filtered (>{max_points} points): {filtered_too_long:,} ({100*filtered_too_long/total_input:.1f}%)")
    
    print(f"\nUnique words: {len(word_counts):,}")
    
    # Top words
    print("\nTop 20 most common words:")
    for word, count in word_counts.most_common(20):
        print(f"  {word:15} {count:6,}")
    
    # Point distribution summary
    print("\nPoint distribution summary:")
    ranges = [(1, 10), (11, 25), (26, 50), (51, 75), (76, 100), 
              (101, 125), (126, 150), (151, 175), (176, 200)]
    
    for start, end in ranges:
        count = sum(point_distribution[i] for i in range(start, end+1))
        if count > 0:
            pct = 100 * count / total_output
            print(f"  {start:3d}-{end:3d} points: {count:7,} ({pct:5.1f}%)")
    
    # Save statistics
    stats = {
        'input_file': str(input_file),
        'output_file': str(output_file),
        'max_points': max_points,
        'total_input': total_input,
        'total_output': total_output,
        'retention_rate': 100 * total_output / total_input,
        'filtered_too_long': filtered_too_long,
        'unique_words': len(word_counts),
        'most_common_words': word_counts.most_common(100)
    }
    
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_file}")
    
    # Show sample output
    print("\n--- SAMPLE OUTPUT (first 3 records) ---")
    with open(output_file, 'r') as f:
        for i in range(3):
            line = f.readline()
            if line:
                data = json.loads(line)
                print(f"\nRecord {i+1}:")
                print(f"  Word: '{data['word']}'")
                print(f"  Points: {len(data['points'])}")
                if len(data['points']) >= 3:
                    print(f"  Duration: {data['points'][-1]['t']}ms")
                    print(f"  First point: x={data['points'][0]['x']:.4f}, y={data['points'][0]['y']:.4f}")
                    print(f"  Last point: x={data['points'][-1]['x']:.4f}, y={data['points'][-1]['y']:.4f}")

def main():
    input_file = "data/futo/train_optimized.jsonl"
    output_file = "data/futo/train_final.jsonl"
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    create_final_dataset(input_file, output_file, max_points=200)
    
    print(f"\nFinal dataset saved to: {output_file}")

if __name__ == "__main__":
    main()