#!/usr/bin/env python3
"""
Fix English QWERTY keyboard layout format to include width/height properties.
"""

import json
from pathlib import Path


def fix_english_grid():
    """Add width and height properties to English grid."""
    
    grid_path = Path("data/data_preprocessed/gridname_to_grid_english.json")
    
    with open(grid_path, 'r') as f:
        grids = json.load(f)
    
    # Add width and height to qwerty_english grid
    # Standard mobile keyboard dimensions
    grids["qwerty_english"]["width"] = 360
    grids["qwerty_english"]["height"] = 215
    
    # Save updated file
    with open(grid_path, 'w') as f:
        json.dump(grids, f, ensure_ascii=False, indent=2)
    
    print(f"Fixed English grid format at {grid_path}")
    print(f"Added width: 360, height: 215")


if __name__ == "__main__":
    fix_english_grid()