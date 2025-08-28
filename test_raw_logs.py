#!/usr/bin/env python3
"""
Quick test of raw log processing with just a few files.
"""

import os
import sys
sys.path.append('.')
from process_raw_logs import main

# Override the main function to limit files
def test_main():
    # Configuration
    DATA_DIR = "data"
    SWIPETRACES_DIR = os.path.join(DATA_DIR, "swipetraces")
    VOCAB_PATH = os.path.join(DATA_DIR, "data_preprocessed", "english_vocab.txt")
    
    print("🔄 Quick Test of Raw Swipe Log Processing")
    print("=" * 50)
    
    # Get just first 10 log files for testing
    log_files = [f for f in os.listdir(SWIPETRACES_DIR) if f.endswith('.log')][:10]
    
    print(f"📁 Testing with {len(log_files)} files")
    
    # Import the processing functions
    from process_raw_logs import load_english_vocab, process_log_file, split_train_val_test, save_dataset
    import json
    import random
    from collections import Counter
    
    # Set seed
    random.seed(42)
    
    # Load vocab
    vocab = load_english_vocab(VOCAB_PATH)
    print(f"📚 Loaded {len(vocab)} vocabulary words")
    
    # Process test files
    all_samples = []
    for filename in log_files:
        file_path = os.path.join(SWIPETRACES_DIR, filename)
        samples = process_log_file(file_path, vocab)
        all_samples.extend(samples)
    
    print(f"\n📊 Test Results:")
    print(f"  Total samples: {len(all_samples)}")
    
    if all_samples:
        word_counts = Counter(sample['word'] for sample in all_samples)
        print(f"  Unique words: {len(word_counts)}")
        print(f"  Sample words: {list(word_counts.keys())[:10]}")
        
        # Show sample data
        print(f"\n📝 Sample trajectory:")
        sample = all_samples[0]
        print(f"  Word: '{sample['word']}'")
        print(f"  Trajectory length: {len(sample['curve']['x'])}")
        print(f"  X range: [{min(sample['curve']['x']):.1f}, {max(sample['curve']['x']):.1f}]")
        print(f"  Y range: [{min(sample['curve']['y']):.1f}, {max(sample['curve']['y']):.1f}]")
        print(f"  Duration: {max(sample['curve']['t'])}ms")
    
    print(f"\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_main()