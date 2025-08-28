#!/usr/bin/env python3
"""
Convert normalized coordinates (0-1 range) to absolute pixel coordinates.
This script processes the English swipe dataset and converts relative coordinates
to absolute coordinates using standard keyboard dimensions.
"""

import json
import os
from typing import List, Dict, Any

# Standard keyboard dimensions (as suggested by user)
KEYBOARD_WIDTH = 360
KEYBOARD_HEIGHT = 215

def convert_coordinates(input_file: str, output_file: str) -> None:
    """
    Convert normalized coordinates to absolute pixel coordinates.
    
    Args:
        input_file: Path to input JSONL file with normalized coordinates
        output_file: Path to output JSONL file with absolute coordinates
    """
    converted_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            
            # Convert x coordinates (0-1) to absolute pixels
            x_coords = data['curve']['x']
            y_coords = data['curve']['y']
            
            # Convert to absolute integer coordinates
            absolute_x = [int(round(x * KEYBOARD_WIDTH)) for x in x_coords]
            absolute_y = [int(round(y * KEYBOARD_HEIGHT)) for y in y_coords]
            
            # Convert timestamps from seconds to milliseconds
            t_coords = data['curve']['t']
            absolute_t = [int(round(t * 1000)) for t in t_coords]
            
            # Update the data
            data['curve']['x'] = absolute_x
            data['curve']['y'] = absolute_y
            data['curve']['t'] = absolute_t
            
            # Write converted line
            outfile.write(json.dumps(data) + '\n')
            converted_count += 1
    
    print(f"Converted {converted_count} samples from {input_file} to {output_file}")

def main():
    """Convert all English dataset files"""
    data_dir = "/home/will/git/swype/neural-swipe-typing/data"
    
    # Files to convert
    files_to_convert = [
        'english_swipes_train.jsonl',
        'english_swipes_val.jsonl', 
        'english_swipes_test.jsonl'
    ]
    
    for filename in files_to_convert:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(data_dir, f"converted_{filename}")
        
        if os.path.exists(input_path):
            print(f"Converting {filename}...")
            convert_coordinates(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found")
    
    print(f"\nUsed keyboard dimensions: {KEYBOARD_WIDTH}x{KEYBOARD_HEIGHT}")
    print("Conversion complete!")

if __name__ == "__main__":
    main()