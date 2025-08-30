#!/usr/bin/env python3
"""
Comprehensive analysis of the combined dataset for mobile deployment optimization.
Analyzes data quality, statistics, and characteristics relevant for real-time inference.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import statistics

def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def analyze_coordinates(data: List[Dict]) -> Dict:
    """Analyze coordinate ranges and distributions"""
    x_coords = []
    y_coords = []
    
    for entry in data:
        curve = entry['curve']
        x_coords.extend(curve['x'])
        y_coords.extend(curve['y'])
    
    return {
        'x_min': min(x_coords), 'x_max': max(x_coords),
        'y_min': min(y_coords), 'y_max': max(y_coords),
        'x_mean': statistics.mean(x_coords), 'x_std': statistics.stdev(x_coords),
        'y_mean': statistics.mean(y_coords), 'y_std': statistics.stdev(y_coords)
    }

def analyze_trajectories(data: List[Dict]) -> Dict:
    """Analyze trajectory characteristics"""
    trace_lengths = []
    velocities = []
    word_lengths = []
    
    for entry in data:
        curve = entry['curve']
        x, y, t = curve['x'], curve['y'], curve['t']
        word = entry['word']
        
        trace_lengths.append(len(x))
        word_lengths.append(len(word))
        
        # Calculate velocities
        for i in range(1, len(x)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            dt = max(t[i] - t[i-1], 1)  # Avoid division by zero
            
            velocity = np.sqrt(dx**2 + dy**2) / dt * 1000  # pixels per second
            velocities.append(velocity)
    
    return {
        'trace_length_mean': statistics.mean(trace_lengths),
        'trace_length_median': statistics.median(trace_lengths),
        'trace_length_min': min(trace_lengths),
        'trace_length_max': max(trace_lengths),
        'velocity_mean': statistics.mean(velocities) if velocities else 0,
        'velocity_median': statistics.median(velocities) if velocities else 0,
        'velocity_max': max(velocities) if velocities else 0,
        'word_length_mean': statistics.mean(word_lengths),
        'word_length_distribution': Counter(word_lengths)
    }

def analyze_vocabulary(data: List[Dict]) -> Dict:
    """Analyze vocabulary characteristics"""
    word_counts = Counter()
    char_counts = Counter()
    
    for entry in data:
        word = entry['word'].lower()
        word_counts[word] += 1
        for char in word:
            char_counts[char] += 1
    
    return {
        'total_unique_words': len(word_counts),
        'total_word_instances': sum(word_counts.values()),
        'most_common_words': word_counts.most_common(20),
        'char_distribution': dict(char_counts.most_common()),
        'words_by_frequency_buckets': {
            '1': len([w for w, c in word_counts.items() if c == 1]),
            '2-5': len([w for w, c in word_counts.items() if 2 <= c <= 5]),
            '6-20': len([w for w, c in word_counts.items() if 6 <= c <= 20]),
            '21-100': len([w for w, c in word_counts.items() if 21 <= c <= 100]),
            '100+': len([w for w, c in word_counts.items() if c > 100])
        }
    }

def analyze_data_quality(data: List[Dict]) -> Dict:
    """Analyze data quality issues"""
    issues = []
    invalid_coordinates = 0
    invalid_timing = 0
    invalid_words = 0
    
    for i, entry in enumerate(data):
        curve = entry['curve']
        word = entry['word']
        x, y, t = curve['x'], curve['y'], curve['t']
        
        # Check coordinate validity
        if any(coord < 0 for coord in x) or any(coord < 0 for coord in y):
            invalid_coordinates += 1
            
        if max(x) > 400 or max(y) > 250:  # Beyond expected keyboard bounds
            invalid_coordinates += 1
            
        # Check timing validity
        if len(set([len(x), len(y), len(t)])) > 1:  # Mismatched array lengths
            issues.append(f"Entry {i}: Mismatched array lengths")
            
        if t != sorted(t):  # Non-monotonic timestamps
            invalid_timing += 1
            
        # Check word validity
        if not word or not word.isalpha():
            invalid_words += 1
            
        if len(word) < 2 or len(word) > 20:
            issues.append(f"Entry {i}: Unusual word length: '{word}'")
    
    return {
        'total_entries': len(data),
        'invalid_coordinates': invalid_coordinates,
        'invalid_timing': invalid_timing, 
        'invalid_words': invalid_words,
        'specific_issues': issues[:10],  # First 10 issues
        'data_quality_score': 1 - (invalid_coordinates + invalid_timing + invalid_words) / (len(data) * 3)
    }

def generate_mobile_optimization_profile(data: List[Dict]) -> Dict:
    """Generate optimization profile for mobile deployment"""
    coord_stats = analyze_coordinates(data)
    traj_stats = analyze_trajectories(data)
    vocab_stats = analyze_vocabulary(data)
    quality_stats = analyze_data_quality(data)
    
    # Mobile-specific analysis
    short_words = [entry for entry in data if len(entry['word']) <= 5]
    common_words = vocab_stats['most_common_words'][:3000]  # Top 3k words
    
    profile = {
        'dataset_size': len(data),
        'coordinate_bounds': {
            'x_range': [coord_stats['x_min'], coord_stats['x_max']],
            'y_range': [coord_stats['y_min'], coord_stats['y_max']],
            'requires_normalization': coord_stats['x_max'] > 360 or coord_stats['y_max'] > 215
        },
        'trajectory_characteristics': {
            'avg_points_per_trace': traj_stats['trace_length_mean'],
            'max_points_per_trace': traj_stats['trace_length_max'],
            'avg_velocity_pps': traj_stats['velocity_mean'],
            'max_velocity_pps': traj_stats['velocity_max']
        },
        'vocabulary_optimization': {
            'total_unique_words': vocab_stats['total_unique_words'],
            'top_3k_coverage': sum(count for word, count in common_words) / vocab_stats['total_word_instances'],
            'recommended_vocab_size': 3000,
            'word_length_distribution': dict(traj_stats['word_length_distribution'])
        },
        'mobile_performance_estimate': {
            'avg_inference_points': traj_stats['trace_length_mean'],
            'max_inference_points': min(traj_stats['trace_length_max'], 150),  # Cap for mobile
            'estimated_latency_ms': traj_stats['trace_length_mean'] * 0.5,  # Rough estimate
            'memory_requirements_mb': (traj_stats['trace_length_mean'] * 128 * 4) / (1024 * 1024)  # Float32
        },
        'data_quality': quality_stats
    }
    
    return profile

def main():
    print("=== Combined Dataset Analysis ===")
    
    # Load datasets
    datasets = {
        'train': 'data/combined_dataset/combined_english_swipes_train.jsonl',
        'val': 'data/combined_dataset/combined_english_swipes_val.jsonl', 
        'test': 'data/combined_dataset/combined_english_swipes_test.jsonl'
    }
    
    all_data = []
    for split, path in datasets.items():
        print(f"Loading {split} data from {path}")
        data = load_dataset(path)
        all_data.extend(data)
        print(f"Loaded {len(data)} samples")
    
    print(f"\nTotal combined dataset: {len(all_data)} samples")
    
    # Generate comprehensive analysis
    profile = generate_mobile_optimization_profile(all_data)
    
    # Print results
    print("\n=== COORDINATE ANALYSIS ===")
    coord_bounds = profile['coordinate_bounds']
    print(f"X range: {coord_bounds['x_range']}")
    print(f"Y range: {coord_bounds['y_range']}")
    print(f"Requires normalization: {coord_bounds['requires_normalization']}")
    
    print("\n=== TRAJECTORY CHARACTERISTICS ===")
    traj = profile['trajectory_characteristics']
    print(f"Average points per trace: {traj['avg_points_per_trace']:.1f}")
    print(f"Maximum points per trace: {traj['max_points_per_trace']}")
    print(f"Average velocity: {traj['avg_velocity_pps']:.1f} pixels/second")
    print(f"Maximum velocity: {traj['max_velocity_pps']:.1f} pixels/second")
    
    print("\n=== VOCABULARY OPTIMIZATION ===")
    vocab = profile['vocabulary_optimization']
    print(f"Total unique words: {vocab['total_unique_words']:,}")
    print(f"Top 3k word coverage: {vocab['top_3k_coverage']:.3f}")
    print(f"Recommended vocab size: {vocab['recommended_vocab_size']:,}")
    print("Word length distribution:")
    for length, count in sorted(vocab['word_length_distribution'].items()):
        print(f"  {length} chars: {count:,} words ({count/len(all_data)*100:.1f}%)")
    
    print("\n=== MOBILE PERFORMANCE ESTIMATES ===")
    mobile = profile['mobile_performance_estimate']
    print(f"Average inference points: {mobile['avg_inference_points']:.1f}")
    print(f"Max inference points (capped): {mobile['max_inference_points']}")
    print(f"Estimated latency: {mobile['estimated_latency_ms']:.1f}ms")
    print(f"Estimated memory per inference: {mobile['memory_requirements_mb']:.2f}MB")
    
    print("\n=== DATA QUALITY ===")
    quality = profile['data_quality']
    print(f"Data quality score: {quality['data_quality_score']:.3f}")
    print(f"Invalid coordinates: {quality['invalid_coordinates']:,}")
    print(f"Invalid timing: {quality['invalid_timing']:,}")
    print(f"Invalid words: {quality['invalid_words']:,}")
    
    # Save profile for mobile optimization
    with open('mobile_optimization_profile.json', 'w') as f:
        json.dump(profile, f, indent=2)
    
    print(f"\nProfile saved to mobile_optimization_profile.json")
    
    return profile

if __name__ == "__main__":
    main()