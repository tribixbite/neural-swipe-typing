#!/usr/bin/env python3
"""
Filter converted JSONL datasets to only include proper English words.
Uses en.txt as the reference English wordlist.
"""

import json
import os
from pathlib import Path
from typing import Set, Dict, Tuple


def load_english_wordlist(filepath: str) -> Set[str]:
    """Load English words from en.txt (tab-separated format: word<tab>frequency)."""
    words = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                if parts:
                    word = parts[0].lower()
                    words.add(word)
    return words


def filter_jsonl_dataset(
    input_path: str, 
    output_path: str, 
    valid_words: Set[str],
    stats: Dict[str, int]
) -> Tuple[int, int]:
    """
    Filter a JSONL dataset to only include samples with valid English words.
    
    Returns:
        Tuple of (kept_samples, filtered_samples)
    """
    kept = 0
    filtered = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                sample = json.loads(line.strip())
                word = sample['word'].lower()
                
                if word in valid_words:
                    outfile.write(json.dumps(sample) + '\n')
                    kept += 1
                    stats['words_kept'].add(word)
                else:
                    filtered += 1
                    stats['words_filtered'].add(word)
    
    return kept, filtered


def main():
    # Paths
    data_dir = Path('../../data/data_preprocessed')
    wordlist_path = '../../data/en.txt'
    
    # Load English wordlist
    print("Loading English wordlist from en.txt...")
    valid_words = load_english_wordlist(wordlist_path)
    print(f"Loaded {len(valid_words)} valid English words")
    
    # Define input and output files
    datasets = [
        ('english_full_train.jsonl', 'english_filtered_train.jsonl'),
        ('english_full_valid.jsonl', 'english_filtered_valid.jsonl'),
        ('english_full_test.jsonl', 'english_filtered_test.jsonl')
    ]
    
    # Process each dataset
    total_kept = 0
    total_filtered = 0
    
    for input_file, output_file in datasets:
        input_path = data_dir / input_file
        output_path = data_dir / output_file
        
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping...")
            continue
        
        print(f"\nProcessing {input_file}...")
        
        # Track statistics
        stats = {
            'words_kept': set(),
            'words_filtered': set()
        }
        
        kept, filtered = filter_jsonl_dataset(
            str(input_path),
            str(output_path),
            valid_words,
            stats
        )
        
        total_kept += kept
        total_filtered += filtered
        
        print(f"  Kept: {kept} samples ({len(stats['words_kept'])} unique words)")
        print(f"  Filtered: {filtered} samples ({len(stats['words_filtered'])} unique words)")
        print(f"  Retention rate: {kept / (kept + filtered) * 100:.1f}%")
        
        # Show examples of filtered words
        if stats['words_filtered']:
            filtered_examples = list(stats['words_filtered'])[:10]
            print(f"  Examples of filtered words: {', '.join(filtered_examples)}")
    
    # Overall statistics
    print("\n" + "="*50)
    print("OVERALL STATISTICS")
    print("="*50)
    print(f"Total samples kept: {total_kept}")
    print(f"Total samples filtered: {total_filtered}")
    print(f"Overall retention rate: {total_kept / (total_kept + total_filtered) * 100:.1f}%")
    
    # Create updated vocabulary file for filtered dataset
    print("\nCreating filtered vocabulary file...")
    vocab_chars = set()
    
    for _, output_file in datasets:
        output_path = data_dir / output_file
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    word = sample['word']
                    vocab_chars.update(word)
    
    # Write vocabulary file
    voc_path = data_dir / 'voc_english_filtered.txt'
    with open(voc_path, 'w', encoding='utf-8') as f:
        # Add special tokens
        f.write('<PAD>\n')
        f.write('<BOS>\n')
        f.write('<EOS>\n')
        # Add characters sorted
        for char in sorted(vocab_chars):
            f.write(char + '\n')
    
    print(f"Created vocabulary file with {len(vocab_chars) + 3} tokens (including special tokens)")
    
    # Create word frequency file for filtered dataset
    print("\nCreating word frequency file...")
    word_counts = {}
    
    for _, output_file in datasets:
        output_path = data_dir / output_file
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    word = sample['word']
                    word_counts[word] = word_counts.get(word, 0) + 1
    
    # Write word frequency file
    wordfreq_path = data_dir / 'wordfreq_english_filtered.txt'
    with open(wordfreq_path, 'w', encoding='utf-8') as f:
        for word, count in sorted(word_counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{count:6d} {word}\n")
    
    print(f"Created word frequency file with {len(word_counts)} unique words")


if __name__ == "__main__":
    main()