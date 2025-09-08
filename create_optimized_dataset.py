#!/usr/bin/env python3
"""
Create optimized training dataset with quality improvements and minimal fields.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

# def remove_duplicate_points(points):
#     """Remove consecutive duplicate points."""
#     if len(points) <= 1:
#         return points
    
#     cleaned = [points[0]]
#     for p in points[1:]:
#         if p['x'] != cleaned[-1]['x'] or p['y'] != cleaned[-1]['y']:
#             cleaned.append(p)
#     return cleaned

def optimize_point(point, precision=12):
    """Optimize a single point by reducing precision and removing timestamp."""
    return {
        'x': round(point['x'], precision),
        'y': round(point['y'], precision),
        't': point['t']  # Keep timestamp for now, we'll compute relative time later
    }

def create_optimized_record(data):
    """Create optimized training record with only essential fields."""
    
    word = data['word']
    points = data['data']
    
    # Remove duplicate consecutive points
    # points = remove_duplicate_points(points)
    
    # Quality checks
    if len(points) < 3:
        return None  # Too short
    
    if len(points) > 400:  # Reduce from 50 to 400 to keep more data
        return None  # Too long
    
    # Check duration
    if len(points) >= 2:
        duration = points[-1]['t'] - points[0]['t']
        if duration < 80 or duration > 3000:
            return None  # Unrealistic duration
    
    # Convert to relative timestamps (milliseconds from start)
    start_time = points[0]['t']
    optimized_points = []
    
    for p in points:
        opt_point = {
            'x': round(p['x'], 12),
            'y': round(p['y'], 12),
            't': p['t'] - start_time  # Relative time from start
        }
        optimized_points.append(opt_point)
    
    # Create minimal record for training
    return {
        'word': word,
        'points': optimized_points
    }

def process_dataset(input_file, output_file, min_word_frequency=1):
    """Process dataset with all optimizations."""
    
    print("Phase 1: Counting word frequencies...")
    word_counts = Counter()
    
    # First pass: count word frequencies
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0:
                print(f"  Counted {i} lines...")
            data = json.loads(line)
            word_counts[data['word']] += 1
    
    print(f"Found {len(word_counts)} unique words")
    
    # Filter words by frequency
    valid_words = {word for word, count in word_counts.items() 
                   if count >= min_word_frequency}
    print(f"Keeping {len(valid_words)} words with frequency >= {min_word_frequency}")
    
    # Statistics
    stats = {
        'total_input': 0,
        'total_output': 0,
        'filtered_low_freq': 0,
        'filtered_too_short': 0,
        'filtered_too_long': 0,
        'filtered_bad_duration': 0,
        'word_distribution': defaultdict(int),
        'point_count_distribution': defaultdict(int)
    }
    
    print("\nPhase 2: Processing and optimizing data...")
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for i, line in enumerate(infile):
            stats['total_input'] += 1
            
            if i % 50000 == 0:
                print(f"  Processed {i} lines, kept {stats['total_output']}...")
            
            data = json.loads(line)
            
            # Check word frequency
            if data['word'] not in valid_words:
                stats['filtered_low_freq'] += 1
                continue
            
            # Create optimized record
            optimized = create_optimized_record(data)
            
            if optimized is None:
                # Track why it was filtered
                points = data['data']
                # points = remove_duplicate_points(data['data'])
                if len(points) < 3:
                    stats['filtered_too_short'] += 1
                elif len(points) > 400:
                    stats['filtered_too_long'] += 1
                else:
                    stats['filtered_bad_duration'] += 1
                continue
            
            # Write optimized record
            outfile.write(json.dumps(optimized, separators=(',', ':')) + '\n')
            stats['total_output'] += 1
            stats['word_distribution'][optimized['word']] += 1
            stats['point_count_distribution'][len(optimized['points'])] += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    print(f"\nInput records: {stats['total_input']:,}")
    print(f"Output records: {stats['total_output']:,} ({100*stats['total_output']/stats['total_input']:.1f}%)")
    
    print("\nFiltering breakdown:")
    print(f"  Low frequency words: {stats['filtered_low_freq']:,}")
    print(f"  Too short (<3 points): {stats['filtered_too_short']:,}")
    print(f"  Too long (>400 points): {stats['filtered_too_long']:,}")
    print(f"  Bad duration: {stats['filtered_bad_duration']:,}")
    
    print(f"\nUnique words in output: {len(stats['word_distribution'])}")
    
    # Top words
    print("\nTop 20 most common words:")
    word_ranking = sorted(stats['word_distribution'].items(), key=lambda x: x[1], reverse=True)
    for word, count in word_ranking[:20]:
        print(f"  {word:15} {count:6,}")
    
    # Point distribution
    print("\nPoint count distribution:")
    for points in sorted(stats['point_count_distribution'].keys())[:10]:
        count = stats['point_count_distribution'][points]
        pct = 100 * count / stats['total_output']
        print(f"  {points:3d} points: {count:6,} ({pct:.1f}%)")
    
    # Save stats
    stats_file = output_file.replace('.jsonl', '_stats.json')
    stats_for_json = {
        'total_input': stats['total_input'],
        'total_output': stats['total_output'],
        'retention_rate': 100 * stats['total_output'] / stats['total_input'],
        'filtered_low_freq': stats['filtered_low_freq'],
        'filtered_too_short': stats['filtered_too_short'],
        'filtered_too_long': stats['filtered_too_long'],
        'filtered_bad_duration': stats['filtered_bad_duration'],
        'unique_words': len(stats['word_distribution']),
        'min_word_frequency': min_word_frequency
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_for_json, f, indent=2)
    
    print(f"\nStats saved to: {stats_file}")
    
    # Sample output
    print("\n--- SAMPLE OUTPUT ---")
    with open(output_file, 'r') as f:
        for i in range(3):
            line = f.readline()
            if line:
                sample = json.loads(line)
                print(f"\nSample {i+1}:")
                print(f"  Word: '{sample['word']}'")
                print(f"  Points: {len(sample['points'])}")
                print(f"  First 3 points:")
                for j, p in enumerate(sample['points'][:3]):
                    print(f"    {j}: x={p['x']:.4f}, y={p['y']:.4f}, t={p['t']}ms")

def main():
    input_file = "data/futo/train_filtered.jsonl"
    output_file = "data/futo/train_optimized.jsonl"
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    process_dataset(input_file, output_file, min_word_frequency=1)
    
    print(f"\nOptimized dataset saved to: {output_file}")

if __name__ == "__main__":
    main()