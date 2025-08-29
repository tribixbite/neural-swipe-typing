#!/usr/bin/env python3
"""
Synthetic Swipe Trace Generator

Downloads synthetic swipe traces from wordgesturegan.com API for the 10k English vocabulary.
Generates multiple traces per word with different standard deviations for data augmentation.
"""

import json
import requests
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm
import logging


class SyntheticTraceGenerator:
    """Generator for synthetic swipe traces using wordgesturegan.com API"""
    
    def __init__(self, vocab_file: str, output_dir: str = "data/synthetic_traces"):
        self.vocab_file = Path(vocab_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API configuration
        self.api_url = "http://wordgesturegan.com/gesture_from_word"
        self.headers = {
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json; charset=UTF-8',
            'Origin': 'http://wordgesturegan.com',
            'Pragma': 'no-cache',
            'Referer': 'http://wordgesturegan.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        # Generation parameters
        self.std_devs = ["0.5", "1", "1.5", "2"]  # Different noise levels
        self.request_delay = (1.0, 1.0)  # 1 second delay between requests
        self.max_retries = 3
        self.timeout = 10
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'generation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_vocabulary(self) -> List[str]:
        """Load vocabulary words from file"""
        if not self.vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_file}")
            
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]
            
        self.logger.info(f"Loaded {len(words)} words from {self.vocab_file}")
        return words
        
    def fetch_gesture(self, word: str, std_dev: str) -> Optional[Dict[str, Any]]:
        """Fetch synthetic gesture for a single word with specified noise level"""
        payload = {
            "word": word,
            "std_dev": std_dev
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=False  # --insecure flag equivalent
                )
                
                if response.status_code == 200:
                    gesture_data = response.json()
                    
                    # Add metadata to the response
                    gesture_data['word'] = word
                    gesture_data['std_dev'] = std_dev
                    gesture_data['timestamp'] = time.time()
                    
                    return gesture_data
                else:
                    self.logger.warning(f"HTTP {response.status_code} for word '{word}', std_dev {std_dev}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed for '{word}' (attempt {attempt + 1}): {e}")
                
            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(random.uniform(1, 3))
                
        self.logger.error(f"Failed to fetch gesture for word '{word}' after {self.max_retries} attempts")
        return None
        
    def generate_traces_for_word(self, word: str) -> List[Dict[str, Any]]:
        """Generate multiple traces for a single word with different noise levels"""
        traces = []
        
        for std_dev in self.std_devs:
            gesture = self.fetch_gesture(word, std_dev)
            if gesture:
                traces.append(gesture)
                
            # Random delay between requests
            delay = random.uniform(*self.request_delay)
            time.sleep(delay)
            
        return traces
        
    def save_traces_batch(self, traces: List[Dict[str, Any]], batch_num: int):
        """Save a batch of traces to JSONL file"""
        output_file = self.output_dir / f"synthetic_traces_batch_{batch_num:04d}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for trace in traces:
                json.dump(trace, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
                
        self.logger.info(f"Saved {len(traces)} traces to {output_file}")
        
    def generate_dataset(self, batch_size: int = 100, start_word: int = 0, max_words: Optional[int] = None):
        """Generate synthetic traces for the entire vocabulary"""
        words = self.load_vocabulary()
        
        # Apply word range limits
        if max_words:
            words = words[start_word:start_word + max_words]
        else:
            words = words[start_word:]
            
        self.logger.info(f"Generating synthetic traces for {len(words)} words")
        self.logger.info(f"Using std_devs: {self.std_devs}")
        self.logger.info(f"Expected total traces: {len(words) * len(self.std_devs)}")
        
        total_traces = 0
        failed_words = []
        batch_traces = []
        batch_num = 0
        
        # Progress bar
        pbar = tqdm(words, desc="Generating traces")
        
        for i, word in enumerate(pbar):
            # Skip words shorter than 3 characters (matching our filter)
            if len(word) < 3:
                continue
                
            traces = self.generate_traces_for_word(word)
            
            if traces:
                batch_traces.extend(traces)
                total_traces += len(traces)
                pbar.set_postfix({
                    'traces': total_traces,
                    'failed': len(failed_words),
                    'batch_size': len(batch_traces)
                })
            else:
                failed_words.append(word)
                
            # Save batch when it reaches target size
            if len(batch_traces) >= batch_size:
                self.save_traces_batch(batch_traces, batch_num)
                batch_traces = []
                batch_num += 1
                
        # Save remaining traces
        if batch_traces:
            self.save_traces_batch(batch_traces, batch_num)
            
        # Generate summary
        self.save_generation_summary(total_traces, failed_words, len(words))
        
    def save_generation_summary(self, total_traces: int, failed_words: List[str], total_words: int):
        """Save generation summary and statistics"""
        summary = {
            'total_words_attempted': total_words,
            'total_traces_generated': total_traces,
            'failed_words_count': len(failed_words),
            'failed_words': failed_words,
            'success_rate': (total_words - len(failed_words)) / total_words if total_words > 0 else 0,
            'std_devs_used': self.std_devs,
            'traces_per_word': len(self.std_devs),
            'generation_timestamp': time.time()
        }
        
        summary_file = self.output_dir / 'generation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Generation completed:")
        self.logger.info(f"  Total traces: {total_traces}")
        self.logger.info(f"  Failed words: {len(failed_words)}")
        self.logger.info(f"  Success rate: {summary['success_rate']:.2%}")
        self.logger.info(f"  Summary saved to: {summary_file}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate synthetic swipe traces using wordgesturegan.com API')
    
    parser.add_argument(
        '--vocab-file',
        type=str,
        default='data/data_preprocessed/english_vocab.txt',
        help='Path to vocabulary file (default: data/data_preprocessed/english_vocab.txt)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic_traces',
        help='Output directory for synthetic traces (default: data/synthetic_traces)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of traces per output file (default: 100)'
    )
    
    parser.add_argument(
        '--start-word',
        type=int,
        default=0,
        help='Start word index (default: 0)'
    )
    
    parser.add_argument(
        '--max-words',
        type=int,
        default=None,
        help='Maximum number of words to process (default: all)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: only generate traces for first 10 words'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Test mode override
    if args.test_mode:
        args.max_words = 10
        print("Running in test mode: processing only first 10 words")
    
    generator = SyntheticTraceGenerator(args.vocab_file, args.output_dir)
    
    try:
        generator.generate_dataset(
            batch_size=args.batch_size,
            start_word=args.start_word,
            max_words=args.max_words
        )
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        generator.logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()