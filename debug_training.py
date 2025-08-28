#!/usr/bin/env python3
"""
Debug script to diagnose training issues.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizerv1
from dataset import CurveDataset, CollateFnV2
from feature_extraction.feature_extractors import (
    weights_function_v1, get_transforms
)
from model import MODEL_GETTERS_DICT


def debug_data_and_model():
    """Debug data loading and model forward pass"""
    
    # Configuration
    GRID_NAME = "qwerty_english"
    BATCH_SIZE = 2  # Small batch for debugging
    MODEL_NAME = "v3_nearest_and_traj_transformer_bigger"
    TRANSFORM_NAME = "traj_feats_and_nearest_key"
    
    # Feature configuration
    USE_TIME = False
    USE_VELOCITY = True
    USE_ACCELERATION = True
    N_COORD_FEATS = 2 * (1 + USE_VELOCITY + USE_ACCELERATION) + USE_TIME  # 6 features
    
    # Paths
    DATA_ROOT = "data"
    train_path = os.path.join(DATA_ROOT, "converted_english_swipes_train.jsonl")
    gridname_to_grid_path = os.path.join(DATA_ROOT, "data_preprocessed", "gridname_to_grid.json")
    voc_path = os.path.join(DATA_ROOT, "data_preprocessed", "english_vocab.txt")
    
    print("🔍 Starting debug analysis...")
    
    # Initialize tokenizers
    char_tokenizer = CharLevelTokenizerv2(voc_path)
    WORD_PAD_IDX = char_tokenizer.char_to_idx['<pad>']
    NUM_CLASSES = 28
    
    print(f"Vocabulary info:")
    print(f"  Total chars: {len(char_tokenizer.idx_to_char)}")
    print(f"  Char to idx sample: {dict(list(char_tokenizer.char_to_idx.items())[:10])}")
    print(f"  PAD token idx: {WORD_PAD_IDX}")
    print(f"  Output classes: {NUM_CLASSES}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        gridname_to_grid_path=gridname_to_grid_path,
        grid_names=[GRID_NAME],
        transform_name=TRANSFORM_NAME,
        char_tokenizer=char_tokenizer,
        uniform_noise_range=0,
        include_time=USE_TIME,
        include_velocities=USE_VELOCITY,
        include_accelerations=USE_ACCELERATION,
        dist_weights_func=weights_function_v1,
        ds_paths_list=[train_path],
        totals=(100,)  # Just first 100 samples
    )
    
    # Create small dataset for debugging
    print(f"\n📊 Creating debug dataset...")
    train_dataset = CurveDataset(
        data_path=train_path,
        store_gnames=False,
        get_item_transform=train_transform,
        total=100
    )
    
    print(f"  Dataset size: {len(train_dataset)}")
    
    # Test single sample
    print(f"\n🔍 Testing single sample...")
    try:
        sample = train_dataset[0]
        print(f"  Sample type: {type(sample)}")
        if isinstance(sample, tuple) and len(sample) == 2:
            (encoder_in, decoder_in), decoder_out = sample
            print(f"  Encoder input type: {type(encoder_in)}")
            if isinstance(encoder_in, tuple):
                print(f"    Traj feats shape: {encoder_in[0].shape}")
                print(f"    KB tokens shape: {encoder_in[1].shape}")
                print(f"    Traj feats range: [{encoder_in[0].min():.3f}, {encoder_in[0].max():.3f}]")
                print(f"    KB tokens range: [{encoder_in[1].min()}, {encoder_in[1].max()}]")
            print(f"  Decoder input shape: {decoder_in.shape}")
            print(f"  Decoder output shape: {decoder_out.shape}")
            print(f"  Decoder input sample: {decoder_in[:10]}")
            print(f"  Decoder output sample: {decoder_out[:10]}")
            print(f"  Decoder output unique values: {torch.unique(decoder_out)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    
    # Create data loader
    collate_fn = CollateFnV2(word_pad_idx=WORD_PAD_IDX, batch_first=False)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn)
    
    # Test batch
    print(f"\n📦 Testing batch loading...")
    try:
        batch_x, batch_y = next(iter(train_loader))
        print(f"  Batch loaded successfully!")
        print(f"  Batch x type: {type(batch_x)}")
        print(f"  Batch y shape: {batch_y.shape}")
        print(f"  Batch y dtype: {batch_y.dtype}")
        print(f"  Batch y range: [{batch_y.min()}, {batch_y.max()}]")
        
        encoder_in, decoder_in, swipe_pad_mask, dec_seq_pad_mask = batch_x
        if isinstance(encoder_in, tuple):
            traj_feats, kb_tokens = encoder_in
            print(f"  Traj feats shape: {traj_feats.shape}")
            print(f"  KB tokens shape: {kb_tokens.shape}")
            print(f"  Traj feats has NaN: {torch.isnan(traj_feats).any()}")
            print(f"  Traj feats has Inf: {torch.isinf(traj_feats).any()}")
        print(f"  Decoder input shape: {decoder_in.shape}")
        print(f"  Swipe pad mask shape: {swipe_pad_mask.shape}")
        print(f"  Dec seq pad mask shape: {dec_seq_pad_mask.shape}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test model
    print(f"\n🧠 Testing model...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {device}")
        
        model = MODEL_GETTERS_DICT[MODEL_NAME](n_coord_feats=N_COORD_FEATS)
        model = model.to(device)
        model.eval()
        
        # Move batch to device
        encoder_in, decoder_in, swipe_pad_mask, dec_seq_pad_mask = batch_x
        if isinstance(encoder_in, tuple):
            encoder_in = tuple(x.to(device) for x in encoder_in)
        else:
            encoder_in = encoder_in.to(device)
        decoder_in = decoder_in.to(device)
        swipe_pad_mask = swipe_pad_mask.to(device)
        dec_seq_pad_mask = dec_seq_pad_mask.to(device)
        batch_y = batch_y.to(device)
        
        with torch.no_grad():
            pred = model.forward(encoder_in, decoder_in, swipe_pad_mask, dec_seq_pad_mask)
            print(f"  Model output shape: {pred.shape}")
            print(f"  Model output dtype: {pred.dtype}")
            print(f"  Model output range: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"  Model output has NaN: {torch.isnan(pred).any()}")
            print(f"  Model output has Inf: {torch.isinf(pred).any()}")
            
            # Test loss computation
            pred_flat = pred.view(-1, pred.shape[-1])
            target_flat = batch_y.reshape(-1)
            
            print(f"  Pred flat shape: {pred_flat.shape}")
            print(f"  Target flat shape: {target_flat.shape}")
            print(f"  Target flat unique: {torch.unique(target_flat)}")
            
            # Check for invalid targets
            valid_mask = (target_flat >= 0) & (target_flat < NUM_CLASSES)
            invalid_targets = target_flat[~valid_mask]
            if len(invalid_targets) > 0:
                print(f"  ⚠️  Invalid targets found: {invalid_targets}")
            
            loss = F.cross_entropy(pred_flat, target_flat, ignore_index=WORD_PAD_IDX)
            print(f"  Loss: {loss.item()}")
            print(f"  Loss is NaN: {torch.isnan(loss)}")
            
        print(f"  ✅ Model forward pass successful!")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = debug_data_and_model()
    if success:
        print(f"\n✅ Debug completed successfully!")
    else:
        print(f"\n❌ Debug found issues!")