#!/usr/bin/env python3
"""
Debug file parsing.
"""

import os

def main():
    file_path = "data/swipetraces/6e6sg1k17tie0fr8coi1kqaj2b.log"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("Header:", repr(lines[0]))
    print("First data line:", repr(lines[1]))
    
    # Try different splitting
    print("\nSplit by tab:")
    parts_tab = lines[1].strip().split('\t')
    print(f"  Parts count: {len(parts_tab)}")
    print(f"  Parts: {parts_tab[:10]}")
    
    print("\nSplit by space:")
    parts_space = lines[1].strip().split(' ')
    print(f"  Parts count: {len(parts_space)}")
    print(f"  Parts: {parts_space[:10]}")
    
    print("\nSplit by whitespace:")
    parts_ws = lines[1].strip().split()
    print(f"  Parts count: {len(parts_ws)}")
    print(f"  Parts: {parts_ws[:10]}")

if __name__ == "__main__":
    main()