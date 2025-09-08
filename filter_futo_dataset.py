#!/usr/bin/env python3
"""
Filter the FUTO dataset according to specific criteria:
1. Remove potentially_invalid_sentence:true
2. Remove words with punctuation (except apostrophes)
3. Remove single character words
4. Restrict to portrait-primary data
5. Require width > height for keyboard canvas
6. Max keyboard width 900px
7. Filter to valid words in top 400k by frequency
"""

import json
import re
import sys
import wordfreq
import nltk
from pathlib import Path
from typing import Dict, Set

# Get NLTK words for validation
try:
    from nltk.corpus import words
    valid_words = set(w.lower() for w in words.words())
except:
    print('Downloading NLTK words corpus...')
    nltk.download('words', quiet=True)
    from nltk.corpus import words
    valid_words = set(w.lower() for w in words.words())

print(f'Loaded {len(valid_words)} dictionary words for validation')


def build_valid_word_set(max_words=400000):
    """Build set of valid high-frequency words using wordfreq and NLTK."""
    print(f'Building valid word set from top {max_words} frequent words...')
    
    valid_word_set = set()
    count = 0
    
    for w in wordfreq.top_n_list('en', max_words):
        clean = w.replace("'", '').replace('-', '').lower()
        
        # Basic validation - must be alphabetic
        if not re.fullmatch(r'[a-z]{2,20}', clean):
            continue
        if len(clean) < 2:
            continue
            
        freq = wordfreq.word_frequency(w, 'en')
        
        # Quality threshold - adjust based on validation
        # Accept if in NLTK dictionary OR has reasonable frequency
        if clean in valid_words or freq >= 5e-8:
            valid_word_set.add(clean)
            # Also add the version with apostrophe if it exists
            if "'" in w:
                valid_word_set.add(w.lower())
            count += 1
    
    print(f'Built valid word set with {len(valid_word_set)} words')
    return valid_word_set


def is_valid_word(word: str, valid_set: Set[str]) -> bool:
    """
    Check if word meets all criteria:
    - Only letters and apostrophes
    - Not single character
    - In valid word set
    """
    # Check for invalid characters (anything not a letter or apostrophe)
    if not re.match(r"^[a-zA-Z']+$", word):
        return False
    
    # Remove single character words
    if len(word) <= 1:
        return False
    
    # Clean word for lookup (remove apostrophes for base form)
    clean_word = word.lower()
    base_word = clean_word.replace("'", '')
    
    # Check if word or base form is in valid set
    return clean_word in valid_set or base_word in valid_set


def filter_dataset(input_file: str, output_file: str, valid_words: Set[str]):
    """Filter the dataset according to all criteria."""
    
    # Statistics
    total_lines = 0
    filtered_lines = 0
    reasons = {
        'invalid_sentence': 0,
        'invalid_word': 0,
        'single_char': 0,
        'not_portrait': 0,
        'wrong_dimensions': 0,
        'width_too_large': 0,
        'not_valid_word': 0,
        'passed': 0
    }
    
    print(f"\nProcessing {input_file}...")
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile):
            total_lines += 1
            
            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} lines, kept {filtered_lines:,} ({100*filtered_lines/(line_num+1):.1f}%)")
            
            try:
                data = json.loads(line)
                
                # 1. Check potentially_invalid_sentence
                if data.get('potentially_invalid_sentence', False):
                    reasons['invalid_sentence'] += 1
                    continue
                
                # 2. Get word and check format
                word = data.get('word', '')
                if not word:
                    reasons['invalid_word'] += 1
                    continue
                
                # Check for invalid characters (not letter or apostrophe)
                if not re.match(r"^[a-zA-Z']+$", word):
                    reasons['invalid_word'] += 1
                    continue
                
                # 3. Remove single character words
                if len(word) <= 1:
                    reasons['single_char'] += 1
                    continue
                
                # 4. Check orientation (portrait-primary)
                orientation = data.get('orientation', '')
                if orientation != 'portrait-primary':
                    reasons['not_portrait'] += 1
                    continue
                
                # 5. Check keyboard dimensions (width > height)
                canvas_width = data.get('canvas_width', 0)
                canvas_height = data.get('canvas_height', 0)
                
                if canvas_width <= canvas_height:
                    reasons['wrong_dimensions'] += 1
                    continue
                
                # 6. Max keyboard width 900px
                if canvas_width > 900:
                    reasons['width_too_large'] += 1
                    continue
                
                # 7. Check if word is in valid word set
                if not is_valid_word(word, valid_words):
                    reasons['not_valid_word'] += 1
                    continue
                
                # All checks passed - write to output
                outfile.write(line)
                filtered_lines += 1
                reasons['passed'] += 1
                
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse line {line_num}")
            except Exception as e:
                print(f"  Error on line {line_num}: {e}")
    
    # Print statistics
    print("\n" + "="*60)
    print("FILTERING COMPLETE")
    print("="*60)
    print(f"\nTotal lines processed: {total_lines:,}")
    print(f"Lines kept: {filtered_lines:,} ({100*filtered_lines/total_lines:.1f}%)")
    print(f"Lines filtered: {total_lines - filtered_lines:,} ({100*(total_lines - filtered_lines)/total_lines:.1f}%)")
    
    print("\nFiltering reasons:")
    print(f"  Invalid sentence flag: {reasons['invalid_sentence']:,}")
    print(f"  Invalid word format: {reasons['invalid_word']:,}")
    print(f"  Single character: {reasons['single_char']:,}")
    print(f"  Not portrait-primary: {reasons['not_portrait']:,}")
    print(f"  Wrong dimensions (w<=h): {reasons['wrong_dimensions']:,}")
    print(f"  Width > 900px: {reasons['width_too_large']:,}")
    print(f"  Not valid word: {reasons['not_valid_word']:,}")
    print(f"  PASSED all filters: {reasons['passed']:,}")
    
    # Save statistics to JSON
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'total_lines': total_lines,
            'filtered_lines': filtered_lines,
            'filter_percentage': 100 * filtered_lines / total_lines,
            'reasons': reasons
        }, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")


def main():
    """Main execution function."""
    input_file = "data/futo/train.jsonl"
    output_file = "data/futo/train_filtered.jsonl"
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Build valid word set
    valid_words = build_valid_word_set(max_words=400000)
    
    # Filter dataset
    filter_dataset(input_file, output_file, valid_words)
    
    print(f"\nFiltered dataset saved to: {output_file}")


if __name__ == "__main__":
    main()