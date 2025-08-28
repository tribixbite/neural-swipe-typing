#!/usr/bin/env python3
"""
Data cleaning script to fix coordinate outliers and other issues.
"""

import json
import os
from typing import Dict, List, Tuple


def clean_coordinates(x_coords: List[float], y_coords: List[float], 
                     keyboard_width: int = 360, keyboard_height: int = 215) -> Tuple[List[float], List[float], bool]:
    """Clean coordinate outliers by clamping to keyboard bounds"""
    cleaned_x = [max(0, min(x, keyboard_width)) for x in x_coords]
    cleaned_y = [max(0, min(y, keyboard_height)) for y in y_coords]
    
    # Check if any coordinates were modified
    modified = (cleaned_x != x_coords) or (cleaned_y != y_coords)
    
    return cleaned_x, cleaned_y, modified


def calculate_velocity_outliers(x_coords: List[float], y_coords: List[float], 
                               timestamps: List[float], max_velocity: float = 2.0) -> bool:
    """Check if trajectory contains velocity outliers"""
    if len(x_coords) < 2:
        return False
    
    for i in range(len(x_coords) - 1):
        dt = timestamps[i+1] - timestamps[i]
        if dt <= 0:
            continue
        
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[i+1] - y_coords[i]
        
        vx = abs(dx / dt)
        vy = abs(dy / dt)
        
        if vx > max_velocity or vy > max_velocity:
            return True
    
    return False


def clean_dataset_file(input_path: str, output_path: str, min_seq_length: int = 5, 
                      max_velocity: float = 2.0) -> Dict[str, int]:
    """Clean a dataset file and return statistics"""
    stats = {
        'total_samples': 0,
        'coord_outliers_fixed': 0,
        'velocity_outliers_removed': 0,
        'short_sequences_removed': 0,
        'samples_kept': 0
    }
    
    print(f"🧹 Cleaning {os.path.basename(input_path)}...")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            stats['total_samples'] += 1
            data = json.loads(line.strip())
            
            curve = data['curve']
            x_coords = curve['x']
            y_coords = curve['y']
            timestamps = curve['t']
            
            # Filter out very short sequences
            if len(x_coords) < min_seq_length:
                stats['short_sequences_removed'] += 1
                continue
            
            # Filter out velocity outliers
            if calculate_velocity_outliers(x_coords, y_coords, timestamps, max_velocity):
                stats['velocity_outliers_removed'] += 1
                continue
            
            # Clean coordinates
            cleaned_x, cleaned_y, coords_modified = clean_coordinates(x_coords, y_coords)
            if coords_modified:
                stats['coord_outliers_fixed'] += 1
                data['curve']['x'] = cleaned_x
                data['curve']['y'] = cleaned_y
            
            # Write cleaned sample
            outfile.write(json.dumps(data) + '\n')
            stats['samples_kept'] += 1
    
    return stats


def main():
    """Clean all dataset files"""
    data_dir = "data"
    files_to_clean = [
        'converted_english_swipes_train.jsonl',
        'converted_english_swipes_val.jsonl', 
        'converted_english_swipes_test.jsonl'
    ]
    
    print("🧹 Cleaning English Swipe Dataset")
    print("=" * 50)
    
    total_stats = {
        'total_samples': 0,
        'coord_outliers_fixed': 0,
        'velocity_outliers_removed': 0,
        'short_sequences_removed': 0,
        'samples_kept': 0
    }
    
    for filename in files_to_clean:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(data_dir, f"clean_{filename}")
        
        if not os.path.exists(input_path):
            print(f"⚠️ {filename} not found, skipping...")
            continue
        
        # Clean the file
        stats = clean_dataset_file(input_path, output_path)
        
        # Update total stats
        for key in total_stats:
            total_stats[key] += stats[key]
        
        # Print file statistics
        kept_pct = (stats['samples_kept'] / stats['total_samples']) * 100 if stats['total_samples'] > 0 else 0
        print(f"  📊 {filename}:")
        print(f"    Original samples: {stats['total_samples']:,}")
        print(f"    Coordinate outliers fixed: {stats['coord_outliers_fixed']:,}")
        print(f"    Velocity outliers removed: {stats['velocity_outliers_removed']:,}")
        print(f"    Short sequences removed: {stats['short_sequences_removed']:,}")
        print(f"    Samples kept: {stats['samples_kept']:,} ({kept_pct:.1f}%)")
        print()
    
    # Print overall statistics
    print("=" * 50)
    print("📈 OVERALL CLEANING RESULTS:")
    overall_kept_pct = (total_stats['samples_kept'] / total_stats['total_samples']) * 100 if total_stats['total_samples'] > 0 else 0
    print(f"  Total samples processed: {total_stats['total_samples']:,}")
    print(f"  Coordinate outliers fixed: {total_stats['coord_outliers_fixed']:,}")
    print(f"  Velocity outliers removed: {total_stats['velocity_outliers_removed']:,}")
    print(f"  Short sequences removed: {total_stats['short_sequences_removed']:,}")
    print(f"  Final samples kept: {total_stats['samples_kept']:,} ({overall_kept_pct:.1f}%)")
    
    samples_removed = total_stats['total_samples'] - total_stats['samples_kept']
    removed_pct = (samples_removed / total_stats['total_samples']) * 100 if total_stats['total_samples'] > 0 else 0
    print(f"  Total samples removed: {samples_removed:,} ({removed_pct:.1f}%)")
    
    print(f"\n✅ Cleaning complete! Use 'clean_converted_*' files for training.")


if __name__ == "__main__":
    main()