#!/usr/bin/env python3
"""
Create proper English QWERTY keyboard grid matching expected format.
"""

import json
from pathlib import Path


def create_proper_english_grid():
    """Create English grid in the proper format with keys array."""
    
    # Standard mobile keyboard dimensions
    width = 360
    height = 215
    
    # Key dimensions
    key_w = 32
    key_h = 42
    
    # Starting positions
    x_spacing = 36
    y_spacing = 47
    
    keys = []
    
    # First row - QWERTYUIOP
    row1_x = 18
    row1_y = 57
    for i, char in enumerate("qwertyuiop"):
        keys.append({
            "label": char,
            "hitbox": {
                "x": row1_x + i * x_spacing,
                "y": row1_y,
                "w": key_w,
                "h": key_h
            }
        })
    
    # Second row - ASDFGHJKL
    row2_x = 36  # Slightly indented
    row2_y = row1_y + y_spacing
    for i, char in enumerate("asdfghjkl"):
        keys.append({
            "label": char,
            "hitbox": {
                "x": row2_x + i * x_spacing,
                "y": row2_y,
                "w": key_w,
                "h": key_h
            }
        })
    
    # Third row - ZXCVBNM
    row3_x = 72  # More indented
    row3_y = row2_y + y_spacing
    for i, char in enumerate("zxcvbnm"):
        keys.append({
            "label": char,
            "hitbox": {
                "x": row3_x + i * x_spacing,
                "y": row3_y,
                "w": key_w,
                "h": key_h
            }
        })
    
    # Add punctuation
    keys.append({
        "label": "'",
        "hitbox": {"x": 324, "y": row2_y, "w": key_w, "h": key_h}
    })
    
    keys.append({
        "label": ".",
        "hitbox": {"x": 288, "y": row3_y, "w": key_w, "h": key_h}
    })
    
    keys.append({
        "label": ",",
        "hitbox": {"x": 252, "y": row3_y, "w": key_w, "h": key_h}
    })
    
    keys.append({
        "label": "-",
        "hitbox": {"x": 342, "y": row1_y, "w": key_w, "h": key_h}
    })
    
    # Space bar (wider key)
    keys.append({
        "label": " ",
        "hitbox": {
            "x": 90,
            "y": row3_y + y_spacing,
            "w": 180,
            "h": key_h
        }
    })
    
    # Add special keys (needed for some models)
    keys.append({
        "action": "shift",
        "hitbox": {"x": 0, "y": row3_y, "w": 60, "h": key_h}
    })
    
    keys.append({
        "action": "backspace",
        "hitbox": {"x": 330, "y": row3_y, "w": 30, "h": key_h}
    })
    
    keys.append({
        "action": "enter",
        "hitbox": {"x": 300, "y": row3_y + y_spacing, "w": 60, "h": key_h}
    })
    
    # Create grid structure
    grid = {
        "qwerty_english": {
            "width": width,
            "height": height,
            "keys": keys
        }
    }
    
    return grid


def main():
    """Save the English grid to file."""
    grid = create_proper_english_grid()
    
    output_path = Path("data/data_preprocessed/gridname_to_grid_english.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(grid, f, ensure_ascii=False, indent=2)
    
    print(f"Created proper English grid at {output_path}")
    print(f"Grid has {len(grid['qwerty_english']['keys'])} keys")
    
    # Print summary
    labels = [k['label'] for k in grid['qwerty_english']['keys'] if 'label' in k]
    print(f"Character keys: {sorted(set(labels))}")


if __name__ == "__main__":
    main()