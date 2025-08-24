#!/usr/bin/env python3
"""
Add English QWERTY keyboard layout to gridname_to_grid.json

Standard QWERTY layout for English with proper key positions.
Based on a 360x215 pixel keyboard (standard mobile dimensions).
"""

import json
from pathlib import Path


def create_english_qwerty_layout():
    """
    Create English QWERTY keyboard layout.
    
    Layout based on standard mobile keyboard dimensions (360x215).
    Each key has: character, center position (x, y), and size (width, height).
    """
    
    # Key width and height for standard keys
    key_width = 32
    key_height = 42
    
    # Starting positions and spacing
    x_start = 18
    y_start = 10
    x_spacing = 36
    y_spacing = 47
    
    # Define QWERTY layout
    qwerty_layout = {
        "qwerty_english": {
            # First row - numbers (optional, not included in basic layout)
            
            # Second row - QWERTYUIOP
            "q": {"x": x_start, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            "w": {"x": x_start + x_spacing, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            "e": {"x": x_start + x_spacing * 2, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            "r": {"x": x_start + x_spacing * 3, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            "t": {"x": x_start + x_spacing * 4, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            "y": {"x": x_start + x_spacing * 5, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            "u": {"x": x_start + x_spacing * 6, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            "i": {"x": x_start + x_spacing * 7, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            "o": {"x": x_start + x_spacing * 8, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            "p": {"x": x_start + x_spacing * 9, "y": y_start + y_spacing, "w": key_width, "h": key_height},
            
            # Third row - ASDFGHJKL (slightly indented)
            "a": {"x": x_start + 18, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            "s": {"x": x_start + 18 + x_spacing, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            "d": {"x": x_start + 18 + x_spacing * 2, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            "f": {"x": x_start + 18 + x_spacing * 3, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            "g": {"x": x_start + 18 + x_spacing * 4, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            "h": {"x": x_start + 18 + x_spacing * 5, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            "j": {"x": x_start + 18 + x_spacing * 6, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            "k": {"x": x_start + 18 + x_spacing * 7, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            "l": {"x": x_start + 18 + x_spacing * 8, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            
            # Fourth row - ZXCVBNM (more indented, with shift key)
            "z": {"x": x_start + 54, "y": y_start + y_spacing * 3, "w": key_width, "h": key_height},
            "x": {"x": x_start + 54 + x_spacing, "y": y_start + y_spacing * 3, "w": key_width, "h": key_height},
            "c": {"x": x_start + 54 + x_spacing * 2, "y": y_start + y_spacing * 3, "w": key_width, "h": key_height},
            "v": {"x": x_start + 54 + x_spacing * 3, "y": y_start + y_spacing * 3, "w": key_width, "h": key_height},
            "b": {"x": x_start + 54 + x_spacing * 4, "y": y_start + y_spacing * 3, "w": key_width, "h": key_height},
            "n": {"x": x_start + 54 + x_spacing * 5, "y": y_start + y_spacing * 3, "w": key_width, "h": key_height},
            "m": {"x": x_start + 54 + x_spacing * 6, "y": y_start + y_spacing * 3, "w": key_width, "h": key_height},
            
            # Space bar (bottom row, wide key)
            " ": {"x": 180, "y": y_start + y_spacing * 4, "w": 180, "h": key_height},
            
            # Additional characters for completeness
            "'": {"x": x_start + 18 + x_spacing * 8.5, "y": y_start + y_spacing * 2, "w": key_width, "h": key_height},
            ".": {"x": x_start + 54 + x_spacing * 7, "y": y_start + y_spacing * 3, "w": key_width, "h": key_height},
            ",": {"x": x_start + 54 + x_spacing * 6.5, "y": y_start + y_spacing * 3, "w": key_width, "h": key_height},
            "-": {"x": x_start + x_spacing * 9.5, "y": y_start + y_spacing, "w": key_width, "h": key_height}
        }
    }
    
    return qwerty_layout


def add_to_grid_file(grid_path: Path, new_layout: dict):
    """Add English layout to existing grid file or create new one."""
    
    # Try to load existing file
    if grid_path.exists():
        with open(grid_path, 'r', encoding='utf-8') as f:
            grids = json.load(f)
    else:
        grids = {}
    
    # Add new layout
    grids.update(new_layout)
    
    # Save updated file
    with open(grid_path, 'w', encoding='utf-8') as f:
        json.dump(grids, f, ensure_ascii=False, indent=2)
    
    print(f"Added English QWERTY layout to {grid_path}")


def create_standalone_grid_file():
    """Create a standalone grid file with just English layout for testing."""
    layout = create_english_qwerty_layout()
    
    output_path = Path("data/data_preprocessed/gridname_to_grid_english.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)
    
    print(f"Created standalone English grid file at {output_path}")
    return output_path


def main():
    # Create English QWERTY layout
    layout = create_english_qwerty_layout()
    
    # Try to add to existing grid file
    grid_path = Path("data/data_preprocessed/gridname_to_grid.json")
    
    if grid_path.exists():
        add_to_grid_file(grid_path, layout)
    else:
        print(f"Grid file not found at {grid_path}")
        print("Creating standalone English grid file for testing...")
        create_standalone_grid_file()
        print("\nNote: You'll need to merge this with the main grid file after pulling from DVC")


if __name__ == "__main__":
    main()