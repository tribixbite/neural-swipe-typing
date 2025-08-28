#!/usr/bin/env python3
"""
Process raw .log files from swipetraces/ folder to generate proper train/val/test datasets.
Handles per-file keyboard layout normalization, sentence splitting, and vocabulary filtering.
"""

import os
import json
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import re

def load_english_vocab(vocab_path: str) -> Set[str]:
    """Load English vocabulary from file"""
    vocab = set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word and len(word) > 1:  # Filter single character words
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

def extract_swipe_trajectory(log_lines: List[str], sentence: str, word: str, 
                           source_width: int, source_height: int) -> Dict:
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
    
    # Use the longest sequence (main swipe)
    if not sequences:
        return None
    
    main_seq = max(sequences, key=len)
    
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
    
    return trajectory if len(trajectory['x']) >= 3 else None

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
            
            # Filter: must be in vocabulary and > 1 character
            if word_clean in vocab and len(word_clean) > 1:
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
    
    # Save statistics
    stats = {
        "total_files_processed": len(log_files),
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "unique_words": len(word_counts),
        "avg_samples_per_word": len(all_samples) / len(word_counts) if word_counts else 0,
        "most_common_words": dict(word_counts.most_common(20))
    }
    
    stats_path = os.path.join(DATA_DIR, "raw_english_dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Processing complete!")
    print(f"  Datasets saved to: raw_converted_english_swipes_*.jsonl")
    print(f"  Statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()