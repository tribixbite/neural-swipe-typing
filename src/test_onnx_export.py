#!/usr/bin/env python3
"""Simple test to verify ONNX export works"""

import json
import torch
import torch.onnx
from pathlib import Path

# Local imports
from model import MODEL_GETTERS_DICT
from export_models import EncodeModule

def test_onnx_export():
    """Test ONNX export functionality"""
    
    # Load config
    with open('export_config_70_7.json', 'r') as f:
        config = json.load(f)
        
    # Load model
    checkpoint_path = "../checkpoints/english_minimal_vram16gb/english-epoch=59-val_loss=0.611-val_word_level_accuracy=0.707.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    # Use the same function from export_models.py
    def get_state_dict_from_checkpoint(ckpt):
        def remove_prefix(s, prefix):
            if s.startswith(prefix):
                s = s[len(prefix):]
            return s
        return {remove_prefix(k, 'model.'): v for k, v in ckpt['state_dict'].items()}
        
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    
    model_params = {
        'n_coord_feats': config.get("n_coord_feats", 6),
        'n_keys': config.get("n_keys", 29),
        'vocab_size': config.get("vocab_size", 32),
        'max_word_len': config.get("max_word_len", 30)
    }
    
    model = MODEL_GETTERS_DICT[config["model_name"]](**model_params).eval()
    model.load_state_dict(state_dict)
    
    print(f"Model loaded successfully: {config['model_name']}")
    
    # Create sample data
    SWIPE_LENGTH = 10
    BATCH_SIZE = 1
    NUM_TRAJ_FEATS = config.get("n_coord_feats", 6)
    N_KEYS = config.get("n_keys", 29)
    
    sample_traj_feats = torch.ones((SWIPE_LENGTH, BATCH_SIZE, NUM_TRAJ_FEATS), dtype=torch.float32)
    sample_kb_key_weights = torch.zeros((SWIPE_LENGTH, BATCH_SIZE, N_KEYS), dtype=torch.float32)
    
    # Set some reasonable key weights
    for i in range(SWIPE_LENGTH):
        sample_kb_key_weights[i, 0, i % N_KEYS] = 0.8
        sample_kb_key_weights[i, 0, (i + 1) % N_KEYS] = 0.2
    
    encoder_in = (sample_traj_feats, sample_kb_key_weights)
    
    # Test forward pass
    print("Testing forward pass...")
    try:
        encoded = model.encode(encoder_in, None)
        print(f"Forward pass successful! Output shape: {encoded.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False
        
    # Create encoder module
    print("Creating encoder module...")
    encode_module = EncodeModule(model).eval()
    
    # Test encoder module forward pass
    print("Testing encoder module...")
    try:
        result = encode_module(sample_traj_feats, sample_kb_key_weights)
        print(f"Encoder module forward pass successful! Output shape: {result.shape}")
    except Exception as e:
        print(f"Encoder module forward pass failed: {e}")
        return False
        
    # Test ONNX export
    output_dir = Path("../results/exported_models_70_7")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting ONNX export...")
    try:
        encoder_onnx_path = output_dir / "neural_swipe_encoder.onnx"
        print(f"Exporting to: {encoder_onnx_path}")
        
        # Check if directory exists and is writable
        if not output_dir.exists():
            print(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            encode_module,
            (sample_traj_feats, sample_kb_key_weights),
            str(encoder_onnx_path),  # Convert to string
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['trajectory_features', 'keyboard_weights'],
            output_names=['encoded_sequence'],
            dynamic_axes={
                'trajectory_features': {0: 'sequence_length'},
                'keyboard_weights': {0: 'sequence_length'},
                'encoded_sequence': {0: 'sequence_length'}
            },
            verbose=False  # Reduce verbose output
        )
        
        # Check if file was created
        if encoder_onnx_path.exists():
            print(f"✅ ONNX export successful! File saved to: {encoder_onnx_path}")
            print(f"File size: {encoder_onnx_path.stat().st_size} bytes")
            return True
        else:
            print(f"❌ ONNX file was not created at {encoder_onnx_path}")
            return False
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_onnx_export()
    if success:
        print("✅ ONNX export test completed successfully!")
    else:
        print("❌ ONNX export test failed!")