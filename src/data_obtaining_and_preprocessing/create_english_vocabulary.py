#!/usr/bin/env python3
"""
Create vocabulary file from English training data.
"""

import json
from pathlib import Path
from collections import Counter


def extract_vocabulary(jsonl_path: Path) -> set:
    """Extract unique characters from dataset."""
    chars = set()
    words = set()
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            word = data['word']
            words.add(word)
            chars.update(word)
    
    print(f"Found {len(words)} unique words")
    print(f"Found {len(chars)} unique characters")
    
    return chars


def create_vocabulary_file(chars: set, output_path: Path):
    """Create vocabulary file with special tokens."""
    # Special tokens
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    
    # Sort characters alphabetically
    sorted_chars = sorted(chars)
    
    # Combine special tokens and characters
    vocab = special_tokens + sorted_chars
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token + '\n')
    
    print(f"Created vocabulary with {len(vocab)} tokens")
    print(f"Saved to {output_path}")


def main():
    # Path to training data
    train_path = Path("data/data_preprocessed/english_full_train.jsonl")
    output_path = Path("data/data_preprocessed/voc_english.txt")
    
    # Extract vocabulary
    chars = extract_vocabulary(train_path)
    
    # Create vocabulary file
    create_vocabulary_file(chars, output_path)
    
    # Display sample
    print("\nCharacters found:", ''.join(sorted(chars)))


if __name__ == "__main__":
    main()