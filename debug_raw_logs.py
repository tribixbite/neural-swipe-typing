#!/usr/bin/env python3
"""
Debug version to see what's happening during processing.
"""

import os
import sys
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter

def load_english_vocab(vocab_path: str) -> Set[str]:
    """Load English vocabulary from file"""
    vocab = set()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word and len(word) > 1:  # Filter single character words
                vocab.add(word)
    return vocab

def debug_process_log_file(file_path: str, vocab: Set[str]) -> List[Dict]:
    """Debug version of process_log_file"""
    print(f"\n🔍 DEBUG: Processing {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines or not lines[0].startswith('sentence'):
        print(f"  ❌ Invalid format")
        return []
    
    # Parse header to get keyboard dimensions
    data_lines = lines[1:]  # Skip header
    if not data_lines:
        print(f"  ❌ No data lines")
        return []
    
    # Get keyboard dimensions from first line
    first_parts = data_lines[0].strip().split()
    if len(first_parts) < 4:
        print(f"  ❌ Invalid first line format")
        return []
    
    keyb_width = int(first_parts[2])
    keyb_height = int(first_parts[3])
    print(f"  📐 Keyboard: {keyb_width}x{keyb_height}")
    
    # Group by sentence and extract words
    sentence_data = defaultdict(list)
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 11:
            sentence = parts[0]
            sentence_data[sentence].append(line)
    
    print(f"  📝 Found {len(sentence_data)} sentences")
    
    trajectories = []
    for sentence, sentence_lines in sentence_data.items():
        print(f"    Sentence: '{sentence}'")
        
        # Split sentence into words
        words = sentence.replace('_', ' ').split()
        print(f"      Words: {words}")
        
        for word in words:
            word_clean = word.lower().strip()
            print(f"        Word: '{word_clean}' (len={len(word_clean)}) in_vocab={word_clean in vocab}")
            
            # Filter: must be in vocabulary and > 1 character
            if word_clean in vocab and len(word_clean) > 1:
                print(f"          ✅ Word accepted, looking for trajectory...")
                
                # Look for this word in the sentence lines
                word_lines = []
                for line in sentence_lines:
                    parts = line.strip().split()
                    if len(parts) >= 11:
                        print(f"            Line check: parts[0]='{parts[0]}' sentence='{sentence}' parts[10]='{parts[10]}' word='{word}'")
                        if parts[10] == word and parts[0] == sentence:
                            word_lines.append(parts)
                
                print(f"          Found {len(word_lines)} lines for this word")
                if word_lines:
                    print(f"          Sample line: {word_lines[0][:10]}")  # Show first 10 fields
                    
                    # Try to extract trajectory (simplified)
                    if len(word_lines) >= 3:  # Need at least 3 points
                        print(f"          ✅ Sufficient points for trajectory")
                        trajectories.append({
                            'word': word_clean,
                            'sentence': sentence,
                            'lines': len(word_lines)
                        })
            else:
                print(f"          ❌ Word rejected")
    
    print(f"  📊 Extracted {len(trajectories)} trajectories")
    return trajectories

def main():
    DATA_DIR = "data"
    SWIPETRACES_DIR = os.path.join(DATA_DIR, "swipetraces")
    VOCAB_PATH = os.path.join(DATA_DIR, "data_preprocessed", "english_vocab.txt")
    
    print("🔄 DEBUG: Raw Swipe Log Processing")
    print("=" * 50)
    
    # Load vocabulary
    vocab = load_english_vocab(VOCAB_PATH)
    print(f"📚 Loaded {len(vocab)} vocabulary words")
    print(f"  Sample vocab: {list(vocab)[:10]}")
    
    # Get first log file
    log_files = [f for f in os.listdir(SWIPETRACES_DIR) if f.endswith('.log')]
    test_file = log_files[0]
    
    file_path = os.path.join(SWIPETRACES_DIR, test_file)
    trajectories = debug_process_log_file(file_path, vocab)
    
    print(f"\n✅ Debug completed!")

if __name__ == "__main__":
    main()