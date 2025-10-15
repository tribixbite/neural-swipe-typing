#!/usr/bin/env python3
"""
Analyze words with traces > 200 points and their distribution below 200 points.
"""

import json
from collections import Counter, defaultdict

def analyze_200_point_threshold(file_path):
    """Analyze words around the 200 point threshold."""
    
    # Track words by point count categories
    words_above_200 = Counter()
    words_below_200 = Counter()
    
    # Track all instances of each word
    word_point_distribution = defaultdict(lambda: {'above_200': 0, 'below_200': 0, 'point_counts': []})
    
    total_traces = 0
    traces_above_200 = 0
    
    print(f"Analyzing {file_path}...")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                print(f"  Processed {i:,} traces...")
            
            data = json.loads(line)
            word = data['word']
            num_points = len(data['points'])
            
            total_traces += 1
            
            # Track distribution
            word_point_distribution[word]['point_counts'].append(num_points)
            
            if num_points > 200:
                traces_above_200 += 1
                words_above_200[word] += 1
                word_point_distribution[word]['above_200'] += 1
            else:
                words_below_200[word] += 1
                word_point_distribution[word]['below_200'] += 1
    
    print(f"\nTotal traces analyzed: {total_traces:,}")
    print(f"Traces with > 200 points: {traces_above_200:,} ({100*traces_above_200/total_traces:.2f}%)")
    print(f"Traces with ≤ 200 points: {total_traces - traces_above_200:,} ({100*(total_traces - traces_above_200)/total_traces:.2f}%)")
    
    # Find words that appear both above and below 200 points
    words_with_both = []
    for word, counts in word_point_distribution.items():
        if counts['above_200'] > 0 and counts['below_200'] > 0:
            words_with_both.append((word, counts['above_200'], counts['below_200']))
    
    words_with_both.sort(key=lambda x: x[1], reverse=True)  # Sort by above_200 count
    
    print(f"\n--- WORDS WITH TRACES BOTH ABOVE AND BELOW 200 POINTS ---")
    print(f"Found {len(words_with_both)} words appearing on both sides of threshold")
    print("\nTop 30 words with traces both >200 and ≤200 points:")
    print(f"{'Word':<20} {'> 200 pts':>10} {'≤ 200 pts':>10} {'Ratio >200':>12}")
    print("-" * 55)
    
    for word, above, below in words_with_both[:30]:
        total = above + below
        ratio = above / total
        print(f"{word:<20} {above:>10} {below:>10} {ratio:>11.1%}")
    
    # Words ONLY above 200 points
    words_only_above = []
    for word, counts in word_point_distribution.items():
        if counts['above_200'] > 0 and counts['below_200'] == 0:
            words_only_above.append((word, counts['above_200']))
    
    words_only_above.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n--- WORDS APPEARING ONLY WITH >200 POINTS ---")
    print(f"Found {len(words_only_above)} words that ONLY appear with >200 points")
    if words_only_above:
        print("\nTop words (or all if < 20):")
        for word, count in words_only_above[:20]:
            avg_points = sum(word_point_distribution[word]['point_counts']) / len(word_point_distribution[word]['point_counts'])
            print(f"  {word:<20} {count:>5} instances, avg points: {avg_points:.1f}")
    
    # Most common words above 200 points
    print(f"\n--- MOST COMMON WORDS WITH >200 POINTS ---")
    print("Top 20 words by frequency above 200 points:")
    for word, count in words_above_200.most_common(20):
        below_count = words_below_200.get(word, 0)
        total = count + below_count
        print(f"  {word:<20} {count:>5} times (also appears {below_count:>5} times ≤200 pts)")
    
    # Analyze word lengths for traces > 200 points
    word_lengths_above_200 = Counter()
    for word, count in words_above_200.items():
        word_lengths_above_200[len(word)] += count
    
    print(f"\n--- WORD LENGTH DISTRIBUTION FOR >200 POINT TRACES ---")
    for length in sorted(word_lengths_above_200.keys()):
        count = word_lengths_above_200[length]
        print(f"  {length:2d} chars: {count:>5} traces")
    
    # Sample some specific cases
    print(f"\n--- SAMPLE ANALYSIS ---")
    sample_words = ['the', 'and', 'that', 'with', 'from', 'which', 'their', 'through', 'between', 'different']
    print("Common words and their point distributions:")
    for word in sample_words:
        if word in word_point_distribution:
            counts = word_point_distribution[word]
            if counts['point_counts']:
                points = counts['point_counts']
                avg = sum(points) / len(points)
                min_pts = min(points)
                max_pts = max(points)
                print(f"  '{word}':")
                print(f"    Total instances: {len(points)}")
                print(f"    > 200 points: {counts['above_200']}")
                print(f"    ≤ 200 points: {counts['below_200']}")
                print(f"    Point range: {min_pts}-{max_pts}, avg: {avg:.1f}")

if __name__ == "__main__":
    analyze_200_point_threshold("data/futo/train_optimized.jsonl")