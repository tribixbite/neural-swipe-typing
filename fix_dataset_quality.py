#!/usr/bin/env python3
"""
Fix critical data quality issues in the combined dataset before training.
Addresses coordinate bounds, timing issues, and data validation.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import statistics

def fix_coordinate_bounds(x_coords: List[float], y_coords: List[float]) -> Tuple[List[float], List[float]]:
    """Fix coordinate bounds to proper 360x215 keyboard dimensions"""
    fixed_x = []
    fixed_y = []
    
    for x, y in zip(x_coords, y_coords):
        # Clamp coordinates to valid bounds
        fixed_x.append(max(0.0, min(360.0, x)))
        fixed_y.append(max(0.0, min(215.0, y)))
    
    return fixed_x, fixed_y

def fix_timing_sequence(t_coords: List[int]) -> List[int]:
    """Ensure monotonic increasing timestamps"""
    if not t_coords:
        return t_coords
    
    fixed_t = [t_coords[0]]
    
    for i in range(1, len(t_coords)):
        # Ensure monotonic increase with minimum 1ms increment
        next_t = max(t_coords[i], fixed_t[-1] + 1)
        fixed_t.append(next_t)
    
    return fixed_t

def validate_entry(entry: Dict) -> bool:
    """Validate entry has required fields and reasonable values"""
    try:
        curve = entry['curve']
        word = entry['word']
        
        # Check required fields
        if not all(key in curve for key in ['x', 'y', 't', 'grid_name']):
            return False
        
        x, y, t = curve['x'], curve['y'], curve['t']
        
        # Check array lengths match
        if len(set([len(x), len(y), len(t)])) > 1:
            return False
        
        # Check minimum trace length
        if len(x) < 3:
            return False
        
        # Check word validity
        if not word or not word.isalpha() or len(word) < 2:
            return False
        
        return True
    except:
        return False

def clean_dataset(input_file: str, output_file: str) -> Tuple[int, int]:
    """Clean dataset by fixing issues and removing invalid entries"""
    print(f"Cleaning {input_file}...")
    
    valid_entries = 0
    fixed_entries = 0
    
    with open(input_file, 'r') as inf, open(output_file, 'w') as outf:
        for line_num, line in enumerate(inf):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                
                # Validate basic structure
                if not validate_entry(entry):
                    continue
                
                curve = entry['curve']
                x, y, t = curve['x'], curve['y'], curve['t']
                
                # Fix coordinate bounds
                fixed_x, fixed_y = fix_coordinate_bounds(x, y)
                
                # Fix timing sequence
                fixed_t = fix_timing_sequence(t)
                
                # Check if we made fixes
                if fixed_x != x or fixed_y != y or fixed_t != t:
                    fixed_entries += 1
                
                # Update entry with fixed data
                entry['curve']['x'] = fixed_x
                entry['curve']['y'] = fixed_y
                entry['curve']['t'] = fixed_t
                
                # Ensure grid_name is correct
                entry['curve']['grid_name'] = 'qwerty_english'
                
                # Write cleaned entry
                outf.write(json.dumps(entry) + '\n')
                valid_entries += 1
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Processed: {valid_entries} valid entries, {fixed_entries} entries fixed")
    return valid_entries, fixed_entries

def main():
    print("=== Dataset Quality Fixing ===")
    
    # File mappings
    datasets = {
        'train': ('data/combined_dataset/combined_english_swipes_train.jsonl', 
                  'data/combined_dataset/cleaned_english_swipes_train.jsonl'),
        'val': ('data/combined_dataset/combined_english_swipes_val.jsonl',
                'data/combined_dataset/cleaned_english_swipes_val.jsonl'),
        'test': ('data/combined_dataset/combined_english_swipes_test.jsonl',
                 'data/combined_dataset/cleaned_english_swipes_test.jsonl')
    }
    
    total_valid = 0
    total_fixed = 0
    
    # Clean each dataset
    for split, (input_file, output_file) in datasets.items():
        valid, fixed = clean_dataset(input_file, output_file)
        total_valid += valid
        total_fixed += fixed
        print(f"{split}: {valid} valid entries, {fixed} fixed")
    
    print(f"\nTotal: {total_valid} valid entries, {total_fixed} entries fixed")
    print("Cleaned datasets saved with 'cleaned_' prefix")
    
    # Quick validation of cleaned data
    print("\n=== Validation of Cleaned Data ===")
    test_file = 'data/combined_dataset/cleaned_english_swipes_test.jsonl'
    
    with open(test_file, 'r') as f:
        sample_entries = [json.loads(line) for line in f.readlines()[:100]]
    
    # Check coordinate bounds
    all_x = [x for entry in sample_entries for x in entry['curve']['x']]
    all_y = [y for entry in sample_entries for y in entry['curve']['y']]
    
    print(f"Sample X range: [{min(all_x)}, {max(all_x)}]")
    print(f"Sample Y range: [{min(all_y)}, {max(all_y)}]")
    
    # Check timing monotonicity
    timing_issues = 0
    for entry in sample_entries:
        t = entry['curve']['t']
        if t != sorted(t):
            timing_issues += 1
    
    print(f"Timing issues in sample: {timing_issues}/100")
    print("Dataset cleaning completed successfully!")

if __name__ == "__main__":
    main()