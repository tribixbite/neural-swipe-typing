#!/usr/bin/env python3
"""
Process raw .log files from swipetraces/ folder to generate proper train/val/test datasets.
Handles per-file keyboard layout normalization, sentence splitting, and vocabulary filtering.
"""

import os
import json
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import re
import statistics

def load_english_vocab(vocab_path: str) -> Set[str]:
    """Load English vocabulary from file"""
    vocab = set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word and len(word) >= 3:  # Filter words less than 3 characters
                vocab.add(word)
    return vocab

def normalize_coordinates(x: float, y: float, source_width: int, source_height: int, 
                        target_width: int = 360, target_height: int = 215) -> Tuple[float, float]:
    """Normalize coordinates from source keyboard to target keyboard dimensions"""
    # Convert to 0-1 normalized coordinates
    norm_x = x / source_width
    norm_y = y / source_height
    
    # Convert to target dimensions
    target_x = norm_x * target_width
    target_y = norm_y * target_height
    
    # Clamp to bounds
    target_x = max(0, min(target_x, target_width))
    target_y = max(0, min(target_y, target_height))
    
    return target_x, target_y

def validate_touch_sequence(sequence: List[List[str]]) -> bool:
    """Validate that touch sequence starts with touchstart and ends with touchend"""
    if not sequence:
        return False
    first_event = sequence[0][4]  # event field
    last_event = sequence[-1][4]
    return first_event == 'touchstart' and last_event == 'touchend'

def has_error_flag(parts: List[str]) -> bool:
    """Check if line has error flag in column 12 or 13"""
    return (len(parts) >= 12 and parts[11] == "1") or (len(parts) >= 13 and parts[12] == "1")

def validate_coordinate_arrays(trajectory: Dict) -> bool:
    """Validate that x, y, t arrays have equal length and sufficient data points"""
    x_len = len(trajectory['x'])
    y_len = len(trajectory['y'])
    t_len = len(trajectory['t'])
    
    # Arrays must have equal length and at least 3 points
    return x_len == y_len == t_len >= 3

def calculate_trajectory_similarity(traj1: Dict, traj2: Dict) -> float:
    """Calculate similarity between two trajectories using DTW-like distance"""
    x1, y1 = np.array(traj1['x']), np.array(traj1['y'])
    x2, y2 = np.array(traj2['x']), np.array(traj2['y'])
    
    # Normalize to 0-1 range for comparison
    if len(x1) == 0 or len(x2) == 0:
        return 0.0
        
    # Simple point-to-point distance if same length, otherwise interpolate
    if len(x1) == len(x2):
        distances = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return 1.0 / (1.0 + np.mean(distances))  # Similarity score 0-1
    else:
        # Resample to same length for comparison
        min_len = min(len(x1), len(x2))
        x1_resample = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(x1)), x1)
        y1_resample = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(y1)), y1)
        x2_resample = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(x2)), x2)
        y2_resample = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(y2)), y2)
        
        distances = np.sqrt((x1_resample - x2_resample)**2 + (y1_resample - y2_resample)**2)
        return 1.0 / (1.0 + np.mean(distances))

def extract_swipe_trajectory(log_lines: List[str], sentence: str, word: str, 
                           source_width: int, source_height: int) -> Optional[Dict]:
    """Extract a single swipe trajectory for a word from log lines"""
    trajectory = {
        'x': [],
        'y': [],
        't': [],
        'grid_name': 'qwerty_english'
    }
    
    word_lines = []
    for line in log_lines:
        parts = line.strip().split()
        if len(parts) >= 12 and parts[10] == word and parts[0] == sentence:
            # Filter out error-flagged lines
            if not has_error_flag(parts):
                word_lines.append(parts)
    
    if not word_lines:
        return None
    
    # Group by continuous touch sequences (touchstart -> touchmove* -> touchend)
    sequences = []
    current_seq = []
    
    for parts in word_lines:
        event = parts[4]
        if event == 'touchstart':
            if current_seq:  # End previous sequence
                sequences.append(current_seq)
            current_seq = [parts]
        elif event in ['touchmove', 'touchend'] and current_seq:
            current_seq.append(parts)
        
        if event == 'touchend' and current_seq:
            sequences.append(current_seq)
            current_seq = []
    
    if current_seq:
        sequences.append(current_seq)
    
    # Use the longest valid sequence (main swipe)
    if not sequences:
        return None
    
    # Filter sequences that have proper touchstart/touchend
    valid_sequences = [seq for seq in sequences if validate_touch_sequence(seq)]
    if not valid_sequences:
        return None
        
    main_seq = max(valid_sequences, key=len)
    
    # Extract coordinates and timestamps
    timestamps = []
    coords = []
    
    for parts in main_seq:
        timestamp = int(parts[1])
        x_pos = float(parts[5])
        y_pos = float(parts[6])
        timestamps.append(timestamp)
        coords.append((x_pos, y_pos))
    
    # Sort by timestamp to ensure chronological order
    sorted_data = sorted(zip(timestamps, coords))
    timestamps, coords = zip(*sorted_data)
    
    # Calculate relative timestamps starting from 0
    base_timestamp = timestamps[0]
    for i, (timestamp, (x_pos, y_pos)) in enumerate(zip(timestamps, coords)):
        rel_timestamp = max(0, timestamp - base_timestamp)  # Ensure non-negative
        
        # Normalize coordinates to target keyboard size
        norm_x, norm_y = normalize_coordinates(x_pos, y_pos, source_width, source_height)
        
        trajectory['x'].append(norm_x)
        trajectory['y'].append(norm_y)
        trajectory['t'].append(rel_timestamp)  # Non-negative relative timestamp in ms
    
    # Validate coordinate arrays
    if not validate_coordinate_arrays(trajectory):
        return None
        
    return trajectory

def process_log_file(file_path: str, vocab: Set[str]) -> List[Dict]:
    """Process a single log file and extract valid word trajectories"""
    print(f"  Processing {os.path.basename(file_path)}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines or not lines[0].startswith('sentence'):
        print(f"    ⚠️ Invalid format, skipping")
        return []
    
    # Parse header to get keyboard dimensions
    data_lines = lines[1:]  # Skip header
    if not data_lines:
        return []
    
    # Get keyboard dimensions from first line (space-separated)
    first_parts = data_lines[0].strip().split()
    if len(first_parts) < 4:
        return []
    
    keyb_width = int(first_parts[2])
    keyb_height = int(first_parts[3])
    
    print(f"    Keyboard: {keyb_width}x{keyb_height}")
    
    # Group by sentence and extract words
    sentence_data = defaultdict(list)
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 12:
            sentence = parts[0]
            sentence_data[sentence].append(line)
    
    trajectories = []
    
    for sentence, sentence_lines in sentence_data.items():
        # Split sentence into words
        words = sentence.replace('_', ' ').split()
        
        for word in words:
            word_clean = word.lower().strip()
            
            # Filter: must be in vocabulary and >= 3 characters
            if word_clean in vocab and len(word_clean) >= 3:
                trajectory = extract_swipe_trajectory(
                    sentence_lines, sentence, word, keyb_width, keyb_height)
                
                if trajectory:
                    sample = {
                        'curve': trajectory,
                        'word': word_clean
                    }
                    trajectories.append(sample)
    
    print(f"    Extracted {len(trajectories)} valid trajectories")
    return trajectories

def split_train_val_test(samples: List[Dict], train_ratio: float = 0.7, 
                        val_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples into train/validation/test sets"""
    random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    return train_samples, val_samples, test_samples

def analyze_keyboard_layout(samples: List[Dict]) -> Dict:
    """Analyze keyboard coordinate usage to understand layout"""
    all_x = []
    all_y = []
    
    for sample in samples:
        curve = sample['curve']
        all_x.extend(curve['x'])
        all_y.extend(curve['y'])
    
    if not all_x or not all_y:
        return {}
    
    x_stats = {
        'min': min(all_x),
        'max': max(all_x),
        'mean': statistics.mean(all_x),
        'median': statistics.median(all_x)
    }
    
    y_stats = {
        'min': min(all_y),
        'max': max(all_y),
        'mean': statistics.mean(all_y),
        'median': statistics.median(all_y)
    }
    
    # Analyze Y coordinate distribution to find keyboard rows
    y_bins = np.histogram(all_y, bins=20)[0]
    y_bin_edges = np.histogram(all_y, bins=20)[1]
    
    # Find peaks in Y distribution (likely keyboard rows)
    peak_indices = []
    for i in range(1, len(y_bins) - 1):
        if y_bins[i] > y_bins[i-1] and y_bins[i] > y_bins[i+1] and y_bins[i] > np.mean(y_bins):
            peak_indices.append(i)
    
    keyboard_rows = [y_bin_edges[i] for i in peak_indices]
    
    return {
        'x_stats': x_stats,
        'y_stats': y_stats,
        'total_coordinates': len(all_x),
        'keyboard_rows_detected': len(keyboard_rows),
        'keyboard_row_positions': keyboard_rows,
        'y_usage_percentage': (y_stats['max'] - y_stats['min']) / 215 * 100  # Assuming 215 target height
    }

def analyze_word_trajectories(samples: List[Dict]) -> Dict:
    """Analyze trajectory patterns for duplicate words"""
    word_trajectories = defaultdict(list)
    
    # Group trajectories by word
    for sample in samples:
        word = sample['word']
        word_trajectories[word].append(sample['curve'])
    
    # Analyze words with multiple trajectories
    similarity_analysis = {}
    trajectory_stats = {}
    
    for word, trajectories in word_trajectories.items():
        trajectory_stats[word] = {
            'count': len(trajectories),
            'avg_length': statistics.mean([len(t['x']) for t in trajectories]),
            'length_variance': statistics.variance([len(t['x']) for t in trajectories]) if len(trajectories) > 1 else 0
        }
        
        # Calculate pairwise similarities for words with multiple trajectories
        if len(trajectories) > 1:
            similarities = []
            for i in range(len(trajectories)):
                for j in range(i + 1, len(trajectories)):
                    sim = calculate_trajectory_similarity(trajectories[i], trajectories[j])
                    similarities.append(sim)
            
            if similarities:
                similarity_analysis[word] = {
                    'mean_similarity': statistics.mean(similarities),
                    'min_similarity': min(similarities),
                    'max_similarity': max(similarities),
                    'trajectory_count': len(trajectories)
                }
    
    return {
        'word_counts': {word: stats['count'] for word, stats in trajectory_stats.items()},
        'trajectory_stats': trajectory_stats,
        'similarity_analysis': similarity_analysis,
        'words_with_multiple_trajectories': len([w for w, t in word_trajectories.items() if len(t) > 1]),
        'total_unique_words': len(word_trajectories)
    }

def validate_data_quality(samples: List[Dict]) -> Dict:
    """Run comprehensive data quality checks"""
    issues = []
    stats = {
        'total_samples': len(samples),
        'coordinate_validation_passed': 0,
        'trajectory_length_stats': [],
        'coordinate_range_issues': 0
    }
    
    for i, sample in enumerate(samples):
        curve = sample['curve']
        word = sample['word']
        
        # Check coordinate array consistency
        if not validate_coordinate_arrays(curve):
            issues.append(f"Sample {i} (word: {word}): Coordinate arrays have inconsistent lengths")
            continue
        
        stats['coordinate_validation_passed'] += 1
        stats['trajectory_length_stats'].append(len(curve['x']))
        
        # Check for reasonable coordinate ranges (0-360, 0-215 for normalized)
        x_vals = curve['x']
        y_vals = curve['y']
        
        if any(x < 0 or x > 360 for x in x_vals) or any(y < 0 or y > 215 for y in y_vals):
            stats['coordinate_range_issues'] += 1
            issues.append(f"Sample {i} (word: {word}): Coordinates outside expected range")
    
    if stats['trajectory_length_stats']:
        stats['avg_trajectory_length'] = statistics.mean(stats['trajectory_length_stats'])
        stats['min_trajectory_length'] = min(stats['trajectory_length_stats'])
        stats['max_trajectory_length'] = max(stats['trajectory_length_stats'])
    
    return {
        'stats': stats,
        'issues': issues[:20],  # Limit to first 20 issues
        'total_issues': len(issues)
    }

def save_dataset(samples: List[Dict], output_path: str):
    """Save dataset to JSONL format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

def main():
    # Configuration
    DATA_DIR = "data"
    SWIPETRACES_DIR = os.path.join(DATA_DIR, "swipetraces")
    VOCAB_PATH = os.path.join(DATA_DIR, "data_preprocessed", "english_vocab.txt")
    
    RANDOM_SEED = 42
    MAX_FILES = None  # Set to a number to limit files for testing, None for all
    
    print("🔄 Processing Raw Swipe Log Files")
    print("=" * 50)
    
    # Set random seed
    random.seed(RANDOM_SEED)
    
    # Load vocabulary
    print("📚 Loading English vocabulary...")
    if not os.path.exists(VOCAB_PATH):
        print(f"❌ Vocabulary file not found: {VOCAB_PATH}")
        return
    
    vocab = load_english_vocab(VOCAB_PATH)
    print(f"  Loaded {len(vocab)} vocabulary words")
    
    # Get log files
    log_files = [f for f in os.listdir(SWIPETRACES_DIR) if f.endswith('.log')]
    if MAX_FILES:
        log_files = log_files[:MAX_FILES]
    
    print(f"📁 Found {len(log_files)} log files")
    
    # Process all files
    all_samples = []
    file_stats = {}
    
    for i, filename in enumerate(log_files):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(log_files)} files")
        
        file_path = os.path.join(SWIPETRACES_DIR, filename)
        samples = process_log_file(file_path, vocab)
        
        file_stats[filename] = len(samples)
        all_samples.extend(samples)
    
    print(f"\n📊 Processing Results:")
    print(f"  Total files processed: {len(log_files)}")
    print(f"  Total valid samples: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("❌ No valid samples found!")
        return
    
    # Show word distribution
    word_counts = Counter(sample['word'] for sample in all_samples)
    print(f"  Unique words: {len(word_counts)}")
    print(f"  Most common words: {dict(word_counts.most_common(10))}")
    
    # Run comprehensive data analysis
    print(f"\n🔍 Running Data Quality Analysis...")
    quality_analysis = validate_data_quality(all_samples)
    print(f"  Samples passing validation: {quality_analysis['stats']['coordinate_validation_passed']}/{quality_analysis['stats']['total_samples']}")
    print(f"  Average trajectory length: {quality_analysis['stats'].get('avg_trajectory_length', 0):.1f} points")
    print(f"  Coordinate range issues: {quality_analysis['stats']['coordinate_range_issues']}")
    
    if quality_analysis['total_issues'] > 0:
        print(f"  ⚠️ Found {quality_analysis['total_issues']} data quality issues")
        for issue in quality_analysis['issues'][:5]:  # Show first 5
            print(f"    - {issue}")
    
    # Analyze keyboard layout usage
    print(f"\n⌨️ Keyboard Layout Analysis...")
    keyboard_analysis = analyze_keyboard_layout(all_samples)
    if keyboard_analysis:
        print(f"  X coordinate range: {keyboard_analysis['x_stats']['min']:.1f} - {keyboard_analysis['x_stats']['max']:.1f}")
        print(f"  Y coordinate range: {keyboard_analysis['y_stats']['min']:.1f} - {keyboard_analysis['y_stats']['max']:.1f}")
        print(f"  Y space utilization: {keyboard_analysis['y_usage_percentage']:.1f}%")
        print(f"  Detected keyboard rows: {keyboard_analysis['keyboard_rows_detected']}")
        if keyboard_analysis['keyboard_row_positions']:
            print(f"  Row positions (Y): {[f'{y:.1f}' for y in keyboard_analysis['keyboard_row_positions']]}")
    
    # Analyze word trajectory patterns
    print(f"\n📈 Trajectory Pattern Analysis...")
    trajectory_analysis = analyze_word_trajectories(all_samples)
    print(f"  Words with multiple trajectories: {trajectory_analysis['words_with_multiple_trajectories']}")
    print(f"  Total unique words: {trajectory_analysis['total_unique_words']}")
    
    # Show similarity analysis for common words
    high_count_words = [(word, count) for word, count in word_counts.most_common(10) if count > 1]
    if high_count_words and trajectory_analysis['similarity_analysis']:
        print(f"  Trajectory similarity for repeated words:")
        for word, count in high_count_words[:5]:
            if word in trajectory_analysis['similarity_analysis']:
                sim = trajectory_analysis['similarity_analysis'][word]
                print(f"    '{word}' ({count} samples): avg similarity {sim['mean_similarity']:.3f}")
    
    # Split into train/val/test
    print(f"\n🔀 Splitting into train/validation/test sets...")
    train_samples, val_samples, test_samples = split_train_val_test(all_samples)
    
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    
    # Save datasets
    print(f"\n💾 Saving datasets...")
    
    train_path = os.path.join(DATA_DIR, "raw_converted_english_swipes_train.jsonl")
    val_path = os.path.join(DATA_DIR, "raw_converted_english_swipes_val.jsonl")
    test_path = os.path.join(DATA_DIR, "raw_converted_english_swipes_test.jsonl")
    
    save_dataset(train_samples, train_path)
    save_dataset(val_samples, val_path)
    save_dataset(test_samples, test_path)
    
    # Save comprehensive statistics
    stats = {
        "total_files_processed": len(log_files),
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "unique_words": len(word_counts),
        "avg_samples_per_word": len(all_samples) / len(word_counts) if word_counts else 0,
        "most_common_words": dict(word_counts.most_common(20)),
        "data_quality": quality_analysis['stats'],
        "keyboard_analysis": keyboard_analysis,
        "trajectory_analysis": {
            "words_with_multiple_trajectories": trajectory_analysis['words_with_multiple_trajectories'],
            "total_unique_words": trajectory_analysis['total_unique_words'],
            "similarity_stats": {
                word: sim for word, sim in trajectory_analysis['similarity_analysis'].items() 
                if word in [w for w, c in word_counts.most_common(20)]
            }
        }
    }
    
    stats_path = os.path.join(DATA_DIR, "raw_english_dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Processing complete!")
    print(f"  Datasets saved to: raw_converted_english_swipes_*.jsonl")
    print(f"  Statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()