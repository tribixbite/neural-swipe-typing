#!/usr/bin/env python3
"""
Convert English swipelogs to JSONL format for training.

This script processes raw swipelog files (touch events) and converts them 
to the expected training format with x, y, t coordinates and target words.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm


def parse_swipelog_line(line: str) -> Optional[Dict]:
    """Parse a single line from swipelog file."""
    parts = line.strip().split()
    if len(parts) < 12:  # Header or malformed line
        return None
    
    try:
        return {
            'sentence': parts[0],
            'timestamp': int(parts[1]),
            'keyb_width': int(parts[2]),
            'keyb_height': int(parts[3]),
            'event': parts[4],
            'x_pos': float(parts[5]),
            'y_pos': float(parts[6]),
            'x_radius': float(parts[7]),
            'y_radius': float(parts[8]),
            'angle': float(parts[9]),
            'word': parts[10],
            'is_err': int(parts[11]) if len(parts) > 11 else 0,
            'is_duplicate': int(parts[12]) if len(parts) > 12 else 0
        }
    except (ValueError, IndexError):
        return None


def group_events_by_swipe(events: List[Dict]) -> List[List[Dict]]:
    """
    Group touch events into individual swipes.
    A swipe starts with touchstart and ends with touchend.
    """
    swipes = []
    current_swipe = []
    
    for event in events:
        if event['event'] == 'touchstart':
            if current_swipe:  # Save previous swipe if exists
                swipes.append(current_swipe)
            current_swipe = [event]
        elif event['event'] in ['touchmove', 'touchend']:
            if current_swipe:  # Only add if we have a started swipe
                current_swipe.append(event)
                if event['event'] == 'touchend':
                    swipes.append(current_swipe)
                    current_swipe = []
    
    # Handle case where last swipe didn't end properly
    if current_swipe:
        swipes.append(current_swipe)
    
    return swipes


def extract_swipe_coordinates(swipe_events: List[Dict]) -> Tuple[List[int], List[int], List[int]]:
    """
    Extract x, y, t coordinates from a swipe's touch events.
    """
    x_coords = []
    y_coords = []
    timestamps = []
    
    if not swipe_events:
        return [], [], []
    
    # Get the starting timestamp
    start_time = swipe_events[0]['timestamp']
    
    for event in swipe_events:
        # Convert to integer coordinates
        x_coords.append(int(event['x_pos']))
        y_coords.append(int(event['y_pos']))
        # Convert to relative time in milliseconds
        timestamps.append(event['timestamp'] - start_time)
    
    return x_coords, y_coords, timestamps


def process_log_file(log_path: Path, keyboard_layout: str = "qwerty_english",
                     filter_errors: bool = True) -> List[Dict]:
    """
    Process a single log file and extract swipe data.
    """
    swipe_data = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header
    if lines and lines[0].startswith('sentence'):
        lines = lines[1:]
    
    # Parse all events
    events = []
    for line in lines:
        event = parse_swipelog_line(line)
        if event:
            events.append(event)
    
    # Group events by word
    word_events = defaultdict(list)
    for event in events:
        # Skip duplicates and optionally errors
        if event['is_duplicate']:
            continue
        if filter_errors and event['is_err']:
            continue
        
        word_key = (event['sentence'], event['word'])
        word_events[word_key].append(event)
    
    # Process each word's swipes
    for (sentence, word), events in word_events.items():
        swipes = group_events_by_swipe(events)
        
        for swipe_events in swipes:
            if len(swipe_events) < 2:  # Need at least start and end
                continue
            
            x_coords, y_coords, timestamps = extract_swipe_coordinates(swipe_events)
            
            if len(x_coords) < 2:  # Need at least 2 points
                continue
            
            swipe_entry = {
                "word": word,
                "curve": {
                    "x": x_coords,
                    "y": y_coords,
                    "t": timestamps,
                    "grid_name": keyboard_layout
                }
            }
            swipe_data.append(swipe_entry)
    
    return swipe_data


def process_all_logs(input_dir: Path, output_path: Path, 
                     keyboard_layout: str = "qwerty_english",
                     filter_errors: bool = True,
                     max_files: Optional[int] = None) -> None:
    """
    Process all log files in directory and save as JSONL.
    """
    log_files = list(input_dir.glob("*.log"))
    
    if max_files:
        log_files = log_files[:max_files]
    
    print(f"Found {len(log_files)} log files to process")
    
    all_swipes = []
    
    for log_file in tqdm(log_files, desc="Processing log files"):
        swipes = process_log_file(log_file, keyboard_layout, filter_errors)
        all_swipes.extend(swipes)
    
    print(f"Extracted {len(all_swipes)} swipes total")
    
    # Write to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for swipe in all_swipes:
            json.dump(swipe, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    print(f"Saved to {output_path}")


def split_dataset(input_path: Path, train_ratio: float = 0.8, 
                  val_ratio: float = 0.1) -> None:
    """
    Split JSONL dataset into train/val/test sets.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    n_total = len(lines)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Shuffle for randomness (optional, using a fixed seed for reproducibility)
    import random
    random.seed(42)
    random.shuffle(lines)
    
    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train + n_val]
    test_lines = lines[n_train + n_val:]
    
    # Save splits
    base_path = input_path.parent
    stem = input_path.stem
    
    splits = {
        f"{stem}_train.jsonl": train_lines,
        f"{stem}_valid.jsonl": val_lines,
        f"{stem}_test.jsonl": test_lines
    }
    
    for filename, data in splits.items():
        output_path = base_path / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(data)
        print(f"Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert swipelogs to JSONL format")
    parser.add_argument("--input-dir", type=Path, default="data/swipelogs/",
                        help="Directory containing swipelog files")
    parser.add_argument("--output-path", type=Path, 
                        default="data/data_preprocessed/english.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--keyboard-layout", type=str, default="qwerty_english",
                        help="Keyboard layout name")
    parser.add_argument("--filter-errors", action="store_true", default=True,
                        help="Filter out erroneous swipes")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    parser.add_argument("--split", action="store_true",
                        help="Split dataset into train/val/test")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation set ratio")
    
    args = parser.parse_args()
    
    # Convert logs to JSONL
    process_all_logs(
        args.input_dir,
        args.output_path,
        args.keyboard_layout,
        args.filter_errors,
        args.max_files
    )
    
    # Optionally split dataset
    if args.split:
        split_dataset(args.output_path, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()