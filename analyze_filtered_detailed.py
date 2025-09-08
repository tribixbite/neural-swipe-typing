#!/usr/bin/env python3
"""Detailed analysis of the filtered dataset to identify further improvements."""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

def analyze_dataset(file_path, sample_size=10000):
    """Perform comprehensive analysis of the dataset."""
    
    # Statistics collectors
    word_counter = Counter()
    word_lengths = Counter()
    swipe_point_counts = []
    swipe_durations = []
    canvas_dimensions = Counter()
    distance_values = []
    
    # Quality metrics
    very_short_swipes = 0  # < 3 points
    very_long_swipes = 0   # > 50 points
    duplicate_points = 0
    suspiciously_fast = 0  # < 100ms
    suspiciously_slow = 0  # > 5000ms
    
    # Word quality
    two_char_words = set()
    three_char_words = set()
    long_words = set()  # > 10 chars
    
    print(f"Analyzing first {sample_size} samples from {file_path}...")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            
            if i % 1000 == 0:
                print(f"  Processed {i} samples...")
            
            data = json.loads(line)
            word = data['word']
            points = data['data']
            
            # Word statistics
            word_counter[word] += 1
            word_lengths[len(word)] += 1
            
            if len(word) == 2:
                two_char_words.add(word)
            elif len(word) == 3:
                three_char_words.add(word)
            elif len(word) > 10:
                long_words.add(word)
            
            # Swipe statistics
            num_points = len(points)
            swipe_point_counts.append(num_points)
            
            if num_points < 3:
                very_short_swipes += 1
            elif num_points > 50:
                very_long_swipes += 1
            
            # Check for duplicate consecutive points
            for j in range(1, len(points)):
                if (points[j]['x'] == points[j-1]['x'] and 
                    points[j]['y'] == points[j-1]['y']):
                    duplicate_points += 1
            
            # Duration analysis
            if len(points) >= 2:
                duration = points[-1]['t'] - points[0]['t']
                swipe_durations.append(duration)
                
                if duration < 100:
                    suspiciously_fast += 1
                elif duration > 5000:
                    suspiciously_slow += 1
            
            # Canvas dimensions
            canvas_key = f"{data['canvas_width']}x{data['canvas_height']}"
            canvas_dimensions[canvas_key] += 1
            
            # Distance values
            if 'distance' in data:
                distance_values.append(data['distance'])
    
    # Print analysis results
    print("\n" + "="*60)
    print("DATASET ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nSamples analyzed: {min(i+1, sample_size)}")
    
    print("\n--- WORD STATISTICS ---")
    print(f"Unique words: {len(word_counter)}")
    print(f"Two-char words: {len(two_char_words)} unique")
    print(f"Three-char words: {len(three_char_words)} unique")
    print(f"Long words (>10 chars): {len(long_words)} unique")
    
    print("\nWord length distribution:")
    for length in sorted(word_lengths.keys())[:15]:
        count = word_lengths[length]
        print(f"  {length:2d} chars: {count:5d} ({100*count/sum(word_lengths.values()):.1f}%)")
    
    print("\n--- SWIPE QUALITY METRICS ---")
    print(f"Average points per swipe: {statistics.mean(swipe_point_counts):.1f}")
    print(f"Median points per swipe: {statistics.median(swipe_point_counts):.0f}")
    print(f"Min/Max points: {min(swipe_point_counts)}/{max(swipe_point_counts)}")
    
    print(f"\nProblematic swipes:")
    print(f"  Very short (<3 points): {very_short_swipes}")
    print(f"  Very long (>50 points): {very_long_swipes}")
    print(f"  Has duplicate points: {duplicate_points}")
    
    if swipe_durations:
        print(f"\nSwipe duration (ms):")
        print(f"  Average: {statistics.mean(swipe_durations):.0f}")
        print(f"  Median: {statistics.median(swipe_durations):.0f}")
        print(f"  Min/Max: {min(swipe_durations):.0f}/{max(swipe_durations):.0f}")
        print(f"  Suspiciously fast (<100ms): {suspiciously_fast}")
        print(f"  Suspiciously slow (>5s): {suspiciously_slow}")
    
    print("\n--- CANVAS DIMENSIONS ---")
    print("Top 5 most common canvas sizes:")
    for dims, count in canvas_dimensions.most_common(5):
        print(f"  {dims}: {count} ({100*count/sum(canvas_dimensions.values()):.1f}%)")
    
    if distance_values:
        print(f"\n--- DISTANCE VALUES ---")
        print(f"Average distance: {statistics.mean(distance_values):.2f}")
        print(f"Median distance: {statistics.median(distance_values):.2f}")
        print(f"Min/Max: {min(distance_values):.2f}/{max(distance_values):.2f}")
    
    print("\n--- SAMPLE DATA ---")
    print("First 3 complete samples:")
    with open(file_path, 'r') as f:
        for i in range(3):
            data = json.loads(f.readline())
            print(f"\nSample {i+1}:")
            print(f"  Word: '{data['word']}'")
            print(f"  Points: {len(data['data'])}")
            print(f"  Duration: {data['data'][-1]['t'] - data['data'][0]['t']}ms")
            print(f"  First point: x={data['data'][0]['x']:.6f}, y={data['data'][0]['y']:.6f}")
            print(f"  Last point: x={data['data'][-1]['x']:.6f}, y={data['data'][-1]['y']:.6f}")
            
            # Check decimal precision
            x_decimals = [len(str(p['x']).split('.')[-1]) for p in data['data'][:3]]
            y_decimals = [len(str(p['y']).split('.')[-1]) for p in data['data'][:3]]
            print(f"  X decimal places (first 3): {x_decimals}")
            print(f"  Y decimal places (first 3): {y_decimals}")
    
    print("\n--- RECOMMENDATIONS ---")
    print("1. Remove swipes with < 3 points (too short for meaningful gesture)")
    print("2. Remove swipes with > 50 points (likely errors or held touches)")
    print("3. Filter duration: 100ms < duration < 3000ms (realistic swipe speeds)")
    print("4. Remove duplicate consecutive points")
    print("5. Reduce coordinate precision to 4 decimal places (sufficient for mobile)")
    print("6. Consider normalizing coordinates to [0,1] range")
    print("7. Add minimum word frequency threshold (e.g., word appears 5+ times)")

if __name__ == "__main__":
    analyze_dataset("data/futo/train_filtered.jsonl", sample_size=10000)