#!/usr/bin/env python3
"""Monitor character-level model training progress."""

import time
import os
import glob
from pathlib import Path

def get_latest_checkpoint():
    """Find the latest checkpoint file."""
    checkpoints = glob.glob("checkpoints/character_model/*.ckpt")
    if checkpoints:
        # Extract accuracy from filename
        best_acc = 0
        best_ckpt = None
        for ckpt in checkpoints:
            # Parse accuracy from filename like: character-model-01-0.456.ckpt
            try:
                acc_str = ckpt.split('-')[-1].replace('.ckpt', '')
                acc = float(acc_str)
                if acc > best_acc:
                    best_acc = acc
                    best_ckpt = ckpt
            except:
                continue
        return best_ckpt, best_acc
    return None, 0

def monitor():
    """Monitor training progress."""
    print("="*60)
    print("Character-Level Model Training Monitor")
    print("="*60)
    print("Target: 70% word accuracy (matching original model)")
    print("Approach: Character-by-character generation with beam search")
    print("-"*60)
    
    last_acc = 0
    no_improvement_count = 0
    
    while True:
        ckpt, acc = get_latest_checkpoint()
        
        if acc > last_acc:
            print(f"\n✓ New best word accuracy: {acc:.1%}")
            if ckpt:
                print(f"  Checkpoint: {Path(ckpt).name}")
            
            if acc >= 0.70:
                print("\n" + "="*60)
                print(f"🎉 TARGET ACHIEVED! {acc:.1%} word accuracy")
                print("Successfully matched original model performance!")
                print("="*60)
                break
            elif acc >= 0.60:
                print(f"  Progress: Getting close to target! ({acc:.1%}/70%)")
            elif acc >= 0.50:
                print(f"  Progress: Good improvement! ({acc:.1%}/70%)")
            elif acc >= 0.30:
                print(f"  Progress: Significant improvement from baseline")
            elif acc >= 0.10:
                print(f"  Progress: Model is learning word patterns")
            elif acc > 0:
                print(f"  Progress: Model starting to generate valid words")
            
            last_acc = acc
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count % 10 == 0:  # Print status every 30 seconds
                if acc > 0:
                    print(f"  Current best: {acc:.1%} (waiting for improvement...)")
                else:
                    print("  Waiting for first checkpoint...")
        
        # Check if training might have stopped
        if no_improvement_count > 100:  # 5 minutes without improvement
            print("\nNo improvement for 5 minutes. Training may have completed.")
            print(f"Final word accuracy: {acc:.1%}")
            if acc < 0.70:
                print("\nNote: Model did not reach target accuracy.")
                print("Consider:")
                print("- Training for more epochs")
                print("- Using more training data")
                print("- Adjusting hyperparameters")
            break
        
        time.sleep(3)  # Check every 3 seconds

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        ckpt, acc = get_latest_checkpoint()
        if acc > 0:
            print(f"Latest word accuracy: {acc:.1%}")