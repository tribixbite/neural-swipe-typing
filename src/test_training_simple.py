#!/usr/bin/env python3
"""Simple test training script for English subset."""

import json
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import CurveDataset
from model import MODEL_GETTERS_DICT
from feature_extraction.feature_extractors import get_val_transform, weights_function_v1
from ns_tokenizers import CharLevelTokenizerv2


def main():
    print("="*50)
    print("Testing English dataset pipeline")
    print("="*50)
    
    # Paths
    train_path = "data/data_preprocessed/english_subset_train.jsonl" 
    voc_path = "data/data_preprocessed/voc_english_filtered.txt"
    grid_path = "data/data_preprocessed/gridname_to_grid_english.json"
    
    # Load vocabulary
    with open(voc_path, 'r') as f:
        voc = f.read().strip().split('\n')
    n_classes = len(voc)
    print(f"Vocabulary size: {n_classes}")
    
    # Load keyboard grid
    with open(grid_path, 'r') as f:
        grid_name_to_grid = json.load(f)
    print(f"Grid names: {list(grid_name_to_grid.keys())}")
    
    # Create tokenizer
    char_tokenizer = CharLevelTokenizerv2(voc_path)
    
    # Get transform (use get_val_transform since we don't need data augmentation for testing)
    transform = get_val_transform(
        gridname_to_grid_path=grid_path,
        grid_names=["qwerty_english"],
        transform_name="traj_feats_and_distance_weights",
        char_tokenizer=char_tokenizer,
        include_time=False,
        include_velocities=True,
        include_accelerations=True,
        dist_weights_func=weights_function_v1
    )
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = CurveDataset(
        path=train_path,
        grid_name_to_grid=grid_name_to_grid,
        transform=transform,
        max_n_points=200
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    print("\nTesting single sample...")
    sample = dataset[0]
    if isinstance(sample, tuple) and len(sample) == 2:
        model_input, target = sample
        print(f"✓ Sample loaded successfully")
        if isinstance(model_input, tuple):
            encoder_in, decoder_in = model_input
            print(f"  Encoder input type: {type(encoder_in)}")
            print(f"  Decoder input type: {type(decoder_in)}")
            if hasattr(encoder_in, 'shape'):
                print(f"  Encoder input shape: {encoder_in.shape}")
            elif isinstance(encoder_in, tuple):
                print(f"  Encoder input (tuple) shapes: {[x.shape for x in encoder_in]}")
        if hasattr(target, 'shape'):
            print(f"  Target shape: {target.shape}")
    
    # Create model
    print("\nCreating model...")
    model_name = "v3_weighted_and_traj_transformer_bigger"
    model = MODEL_GETTERS_DICT[model_name]()
    print(f"✓ Model created: {type(model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass with batched data
    print("\nTesting forward pass...")
    from torch.utils.data import DataLoader
    from dataset import custom_collate_fn
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    batch = next(iter(loader))
    packed_model_in, dec_out = batch
    encoder_in, decoder_in, swipe_pad_mask, word_pad_mask = packed_model_in
    
    print(f"Batch shapes:")
    if isinstance(encoder_in, tuple):
        print(f"  Encoder input (tuple): {[x.shape for x in encoder_in]}")
    else:
        print(f"  Encoder input: {encoder_in.shape}")
    print(f"  Decoder input: {decoder_in.shape}")
    print(f"  Decoder output: {dec_out.shape}")
    
    model.eval()
    with torch.no_grad():
        try:
            output = model(encoder_in, decoder_in, swipe_pad_mask, word_pad_mask)
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {output.shape}")
            
            # Test loss computation
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                dec_out.reshape(-1)
            )
            print(f"✓ Loss computation successful!")
            print(f"  Loss value: {loss.item():.4f}")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n" + "="*50)
    print("All tests passed! Ready for training.")
    print("="*50)
    return 0


if __name__ == "__main__":
    exit(main())