#!/usr/bin/env python3

import json
import sys
from collections import Counter
from pathlib import Path

def load_vocabulary(vocab_file):
    """Load vocabulary from file"""
    vocab = set()
    try:
        with open(vocab_file, 'r') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    vocab.add(word)
    except FileNotFoundError:
        print(f"Warning: Could not find vocabulary file {vocab_file}")
    return vocab

def analyze_dataset(dataset_file, vocab_file):
    """Analyze the futo dataset"""
    
    # Load vocabulary for validation
    print(f"Loading vocabulary from {vocab_file}...")
    vocabulary = load_vocabulary(vocab_file)
    print(f"Loaded {len(vocabulary)} words from vocabulary")
    
    # Statistics
    total_samples = 0
    word_counter = Counter()
    single_char_words = set()
    two_char_words = set()
    real_words = set()
    fake_words = set()
    word_lengths = Counter()
    
    print(f"\nAnalyzing dataset: {dataset_file}")
    print("This may take a while for large files...")
    
    # Process dataset
    with open(dataset_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} lines...")
            
            try:
                data = json.loads(line)
                if 'word' in data:
                    word = data['word'].lower()
                    word_counter[word] += 1
                    word_lengths[len(word)] += 1
                    
                    # Track single and two character words
                    if len(word) == 1:
                        single_char_words.add(word)
                    elif len(word) == 2:
                        two_char_words.add(word)
                    
                    # Check if it's a real word
                    if word in vocabulary:
                        real_words.add(word)
                    else:
                        fake_words.add(word)
                    
                    total_samples += 1
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse line {line_num}")
            except KeyError:
                print(f"  Warning: No 'word' field in line {line_num}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Unique words: {len(word_counter):,}")
    
    print(f"\nSingle character words: {len(single_char_words)}")
    if single_char_words:
        print(f"  Examples: {sorted(single_char_words)[:20]}")
    
    print(f"\nTwo character words: {len(two_char_words)}")
    if two_char_words:
        print(f"  Examples: {sorted(two_char_words)[:20]}")
    
    print(f"\nReal words (in vocabulary): {len(real_words):,} ({100*len(real_words)/len(word_counter):.1f}%)")
    print(f"Fake/unknown words: {len(fake_words):,} ({100*len(fake_words)/len(word_counter):.1f}%)")
    
    print("\nWord length distribution:")
    for length in sorted(word_lengths.keys())[:15]:
        count = word_lengths[length]
        percentage = 100 * count / total_samples
        print(f"  {length:2d} chars: {count:8,} ({percentage:5.2f}%)")
    
    print("\nMost common words:")
    for word, count in word_counter.most_common(20):
        in_vocab = "✓" if word in vocabulary else "✗"
        print(f"  {in_vocab} '{word}': {count:,}")
    
    print("\nExample fake/unknown words (not in vocabulary):")
    fake_sample = sorted(fake_words)[:30]
    for word in fake_sample:
        print(f"  '{word}'")
    
    # Check for common misspellings or typos
    print("\nPotential typos (1 edit distance from real words):")
    typo_count = 0
    for fake_word in sorted(fake_words)[:100]:  # Check first 100 fake words
        if len(fake_word) > 2:  # Skip very short words
            for real_word in vocabulary:
                if abs(len(fake_word) - len(real_word)) <= 1:
                    if levenshtein_distance(fake_word, real_word) == 1:
                        print(f"  '{fake_word}' -> '{real_word}'")
                        typo_count += 1
                        if typo_count >= 20:
                            break
            if typo_count >= 20:
                break

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

if __name__ == "__main__":
    dataset_file = "data/futo/train.jsonl"
    vocab_file = "exported_models_english/english_vocab.txt"
    
    if not Path(dataset_file).exists():
        print(f"Error: Dataset file not found: {dataset_file}")
        sys.exit(1)
    
    if not Path(vocab_file).exists():
        print(f"Warning: Vocabulary file not found: {vocab_file}")
        print("Will proceed without word validation")
        vocab_file = None
    
    analyze_dataset(dataset_file, vocab_file)