#!/usr/bin/env python3
"""
Advanced NaN debugging script to find the root cause.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ns_tokenizers import CharLevelTokenizerv2
from dataset import CurveDataset, CollateFnV2
from feature_extraction.feature_extractors import weights_function_v1, get_transforms
from model import MODEL_GETTERS_DICT


def check_nan_inf(tensor, name):
    """Check tensor for NaN or infinity values"""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    if has_nan or has_inf:
        print(f"  ⚠️ {name}: NaN={has_nan}, Inf={has_inf}")
        if has_nan:
            print(f"    NaN locations: {torch.isnan(tensor).sum()} values")
        if has_inf:
            print(f"    Inf locations: {torch.isinf(tensor).sum()} values")
        return True
    return False


def debug_model_step_by_step():
    """Debug the model forward pass step by step"""
    
    # Configuration  
    GRID_NAME = "qwerty_english"
    BATCH_SIZE = 1  # Single sample for detailed debugging
    MODEL_NAME = "v3_nearest_and_traj_transformer_bigger"
    TRANSFORM_NAME = "traj_feats_and_nearest_key"
    
    # Feature configuration
    USE_TIME = False
    USE_VELOCITY = True
    USE_ACCELERATION = True
    N_COORD_FEATS = 2 * (1 + USE_VELOCITY + USE_ACCELERATION) + USE_TIME
    
    # Paths
    DATA_ROOT = "data"
    train_path = os.path.join(DATA_ROOT, "converted_english_swipes_train.jsonl")
    gridname_to_grid_path = os.path.join(DATA_ROOT, "data_preprocessed", "gridname_to_grid.json")
    voc_path = os.path.join(DATA_ROOT, "data_preprocessed", "english_vocab.txt")
    
    print("🔍 Advanced NaN debugging...")
    
    # Initialize tokenizer
    char_tokenizer = CharLevelTokenizerv2(voc_path)
    WORD_PAD_IDX = char_tokenizer.char_to_idx['<pad>']
    NUM_CLASSES = len(char_tokenizer.idx_to_char)
    
    # Get transforms
    train_transform, _ = get_transforms(
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
        totals=(10,)  # Just 10 samples for debugging
    )
    
    # Create dataset
    train_dataset = CurveDataset(
        data_path=train_path,
        store_gnames=False,
        get_item_transform=train_transform,
        total=10
    )
    
    # Create data loader
    collate_fn = CollateFnV2(word_pad_idx=WORD_PAD_IDX, batch_first=False)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn)
    
    # Get first batch
    batch_x, batch_y = next(iter(train_loader))
    encoder_in, decoder_in, swipe_pad_mask, dec_seq_pad_mask = batch_x
    
    print(f"\n📊 Batch Analysis:")
    print(f"  Encoder input type: {type(encoder_in)}")
    
    if isinstance(encoder_in, tuple):
        traj_feats, kb_tokens = encoder_in
        print(f"  Trajectory features shape: {traj_feats.shape}")
        print(f"  Keyboard tokens shape: {kb_tokens.shape}")
        
        # Check trajectory features in detail
        print(f"\n🔍 Trajectory features analysis:")
        check_nan_inf(traj_feats, "Trajectory features")
        print(f"    Min: {traj_feats.min().item():.6f}")
        print(f"    Max: {traj_feats.max().item():.6f}")
        print(f"    Mean: {traj_feats.mean().item():.6f}")
        print(f"    Std: {traj_feats.std().item():.6f}")
        
        # Check each feature dimension
        for i in range(traj_feats.shape[-1]):
            feat_slice = traj_feats[:, :, i]
            print(f"    Feature {i}: min={feat_slice.min().item():.3f}, max={feat_slice.max().item():.3f}, mean={feat_slice.mean().item():.3f}")
            check_nan_inf(feat_slice, f"Feature {i}")
        
        # Check keyboard tokens
        print(f"\n⌨️ Keyboard tokens analysis:")
        check_nan_inf(kb_tokens.float(), "Keyboard tokens")
        print(f"    Min: {kb_tokens.min()}")
        print(f"    Max: {kb_tokens.max()}")
        print(f"    Unique values: {torch.unique(kb_tokens)}")
    
    # Check other inputs
    print(f"\n🔍 Other inputs:")
    check_nan_inf(decoder_in.float(), "Decoder input")
    check_nan_inf(swipe_pad_mask.float(), "Swipe pad mask")
    check_nan_inf(dec_seq_pad_mask.float(), "Decoder seq pad mask")
    check_nan_inf(batch_y.float(), "Target labels")
    
    print(f"  Target labels range: [{batch_y.min()}, {batch_y.max()}]")
    print(f"  Target labels unique: {torch.unique(batch_y)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🧠 Initializing model on {device}...")
    
    model = MODEL_GETTERS_DICT[MODEL_NAME](n_coord_feats=N_COORD_FEATS)
    model = model.to(device)
    
    # Check model parameters
    print(f"\n🔍 Model parameter analysis:")
    param_issues = False
    for name, param in model.named_parameters():
        if check_nan_inf(param, f"Parameter: {name}"):
            param_issues = True
    
    if not param_issues:
        print("  ✅ All model parameters are clean")
    
    # Move batch to device
    if isinstance(encoder_in, tuple):
        encoder_in = tuple(x.to(device) for x in encoder_in)
    else:
        encoder_in = encoder_in.to(device)
    decoder_in = decoder_in.to(device)
    swipe_pad_mask = swipe_pad_mask.to(device)
    dec_seq_pad_mask = dec_seq_pad_mask.to(device)
    batch_y = batch_y.to(device)
    
    # Forward pass with gradient tracking
    model.train()
    print(f"\n🚀 Forward pass...")
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    try:
        pred = model.forward(encoder_in, decoder_in, swipe_pad_mask, dec_seq_pad_mask)
        print(f"  Model output shape: {pred.shape}")
        
        has_issues = check_nan_inf(pred, "Model output")
        if not has_issues:
            print(f"  ✅ Model output is clean")
            print(f"    Min: {pred.min().item():.6f}")
            print(f"    Max: {pred.max().item():.6f}")
            print(f"    Mean: {pred.mean().item():.6f}")
        
        # Test loss computation
        print(f"\n📊 Loss computation...")
        pred_flat = pred.view(-1, pred.shape[-1])
        target_flat = batch_y.reshape(-1)
        
        print(f"  Pred flat shape: {pred_flat.shape}")
        print(f"  Target flat shape: {target_flat.shape}")
        
        # Check for invalid targets
        valid_mask = (target_flat >= 0) & (target_flat < NUM_CLASSES)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"  ⚠️ Found {invalid_count} invalid targets")
            print(f"    Invalid values: {target_flat[~valid_mask]}")
        
        # Compute loss
        loss = F.cross_entropy(pred_flat, target_flat, ignore_index=WORD_PAD_IDX)
        print(f"  Loss: {loss.item()}")
        
        loss_has_issues = check_nan_inf(loss, "Loss")
        if loss_has_issues:
            print(f"  🚨 Loss has numerical issues!")
            return False
        
        # Test backward pass
        print(f"\n⬅️ Backward pass...")
        loss.backward()
        
        # Check gradients
        print(f"\n📊 Gradient analysis:")
        grad_issues = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if check_nan_inf(param.grad, f"Gradient: {name}"):
                    grad_issues = True
                    # Print some gradient stats
                    print(f"    Grad shape: {param.grad.shape}")
                    print(f"    Grad norm: {param.grad.norm().item():.6f}")
        
        if grad_issues:
            print(f"  🚨 Found gradient issues!")
            return False
        else:
            print(f"  ✅ All gradients are clean")
        
    except Exception as e:
        print(f"  🚨 Error during forward/backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        torch.autograd.set_detect_anomaly(False)
    
    print(f"\n✅ Advanced debugging completed - no obvious issues found!")
    print(f"💡 This suggests the NaN might appear during optimization steps or multi-batch interactions.")
    
    return True


if __name__ == "__main__":
    success = debug_model_step_by_step()
    if not success:
        print(f"\n❌ Found issues that could cause NaN!")
    else:
        print(f"\n🤔 No obvious issues found - NaN might be from optimizer or batch interactions")