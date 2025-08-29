#!/usr/bin/env python3
"""
Run synthetic trace generation for the full 10k word vocabulary

Usage examples:
    # Generate traces for all 10k words (this will take ~6-8 hours)
    python run_synthetic_generation.py --full
    
    # Generate traces for first 1000 words only
    python run_synthetic_generation.py --max-words 1000
    
    # Resume generation starting from word 5000
    python run_synthetic_generation.py --start-word 5000 --max-words 1000
    
    # Test mode - just 10 words
    python run_synthetic_generation.py --test
"""

import argparse
from generate_synthetic_traces import SyntheticTraceGenerator


def main():
    parser = argparse.ArgumentParser(description='Run synthetic trace generation')
    
    parser.add_argument('--full', action='store_true', 
                       help='Generate traces for all 10k words (~6-8 hours)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: generate traces for first 10 words only')
    parser.add_argument('--start-word', type=int, default=0,
                       help='Starting word index (default: 0)')
    parser.add_argument('--max-words', type=int, default=None,
                       help='Maximum number of words to process')
    parser.add_argument('--batch-size', type=int, default=500,
                       help='Traces per output file (default: 500)')
    
    args = parser.parse_args()
    
    # Configure based on mode
    if args.test:
        max_words = 10
        print("🧪 TEST MODE: Processing 10 words only")
    elif args.full:
        max_words = None
        print("🚀 FULL GENERATION: Processing all 10k words")
        print("⏰ Estimated time: 6-8 hours")
        print("📊 Expected output: ~40k synthetic traces")
    else:
        max_words = args.max_words
        if max_words:
            print(f"📊 PARTIAL GENERATION: Processing {max_words} words")
        else:
            print("🚀 FULL GENERATION: Processing all 10k words")
    
    # Initialize generator
    generator = SyntheticTraceGenerator(
        vocab_file='data/data_preprocessed/english_vocab.txt',
        output_dir='data/synthetic_traces'
    )
    
    # Confirm for full generation
    if args.full and not args.test:
        response = input("\n⚠️  Full generation will take 6-8 hours. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Generation cancelled.")
            return
    
    print("\n🎯 Generation Configuration:")
    print(f"   Start word: {args.start_word}")
    print(f"   Max words: {max_words or 'All (~10k)'}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Noise levels: 0.5, 1, 1.5, 2")
    print(f"   Output: data/synthetic_traces/\n")
    
    try:
        generator.generate_dataset(
            batch_size=args.batch_size,
            start_word=args.start_word,
            max_words=max_words
        )
        print("\n✅ Generation completed successfully!")
        print("📁 Check data/synthetic_traces/ for output files")
        
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()