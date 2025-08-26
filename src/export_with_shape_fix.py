#!/usr/bin/env python3
"""
Export models with comprehensive shape fixing.
"""

import argparse
import json
import torch
from pathlib import Path

# Local imports  
from model import MODEL_GETTERS_DICT


def comprehensive_shape_fix(state_dict, checkpoint_path):
    """Fix all known shape mismatches in state dict."""
    
    fixed_dict = {}
    
    for key, tensor in state_dict.items():
        original_key = key
        
        # Fix positional encoding shapes
        if key == 'enc_in_emb_model.weighted_sum_emb.pos_encoder.pe':
            if tensor.shape == (2048, 1, 122):
                # Transpose to [1, 2048, 122]
                fixed_dict[key] = tensor.permute(1, 0, 2)
                print(f"Fixed {key}: {tensor.shape} -> {fixed_dict[key].shape}")
                continue
                
        elif key == 'dec_in_emb_model.2.pe':
            if tensor.shape == (29, 1, 128):
                # Transpose to [1, 29, 128] 
                fixed_dict[key] = tensor.permute(1, 0, 2)
                print(f"Fixed {key}: {tensor.shape} -> {fixed_dict[key].shape}")
                continue
        
        # Keep original tensor
        fixed_dict[key] = tensor
    
    return fixed_dict


def main():
    parser = argparse.ArgumentParser(description="Export with shape fixes")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--config", required=True, help="Config file") 
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔧 Model Export with Shape Fixes")
    print("=" * 45)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    
    # Get hyperparameters 
    hparams = checkpoint.get('hyper_parameters', {})
    
    # Extract and fix state dict
    def remove_prefix(s, prefix='model.'):
        return s[len(prefix):] if s.startswith(prefix) else s
    
    raw_state_dict = {remove_prefix(k): v for k, v in checkpoint['state_dict'].items()}
    
    print("Applying shape fixes...")
    fixed_state_dict = comprehensive_shape_fix(raw_state_dict, args.checkpoint)
    
    # Use checkpoint hyperparameters
    model_params = {
        'n_coord_feats': hparams.get('n_coord_feats', 6),
        'n_keys': hparams.get('n_keys', 29),  
        'vocab_size': hparams.get('vocab_size', 32),
        'max_word_len': hparams.get('max_word_len', 30)
    }
    
    print(f"Model parameters: {model_params}")
    
    # Create and load model
    model_name = hparams.get('model_name', config["model_name"])
    model = MODEL_GETTERS_DICT[model_name](**model_params).eval()
    
    try:
        model.load_state_dict(fixed_state_dict)
        print("✅ Model loaded successfully with fixes")
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        return 1
    
    # Test model
    print("🧪 Testing model...")
    try:
        seq_len = 5
        test_traj = torch.randn(seq_len, 1, 6) 
        test_kb = torch.softmax(torch.randn(seq_len, 1, 29), dim=-1)
        
        with torch.no_grad():
            encoded = model.encode((test_traj, test_kb), None)
            
        print(f"✅ Model test passed: {encoded.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return 1
    
    # Save model
    model_path = output_dir / "model_state_dict.pt"
    torch.save(model.state_dict(), model_path)
    
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"✅ Model saved: {model_path} ({size_mb:.1f}MB)")
    
    # Save configuration
    accuracy = "unknown"
    if "accuracy=" in args.checkpoint:
        accuracy = args.checkpoint.split("accuracy=")[1].split(".ckpt")[0]
    elif "0.635" in args.checkpoint:
        accuracy = "0.635"
    elif "0.625" in args.checkpoint:
        accuracy = "0.625"
    
    export_config = {
        **config,
        "checkpoint_hyperparameters": hparams,
        "model_parameters": model_params,
        "export_info": {
            "checkpoint": args.checkpoint,
            "accuracy": accuracy,
            "vocab_size": model_params["vocab_size"],
            "model_size_mb": f"{size_mb:.1f}",
            "export_method": "comprehensive_shape_fix"
        }
    }
    
    config_path = output_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(export_config, f, indent=2)
    
    # Create loader script
    loader_script = f'''#!/usr/bin/env python3
"""Load the exported {accuracy} accuracy model."""

import torch
from model import MODEL_GETTERS_DICT

def load_model(device='cpu'):
    """Load the {accuracy} accuracy neural swipe typing model."""
    
    model_params = {model_params}
    model = MODEL_GETTERS_DICT["{model_name}"](**model_params)
    
    state_dict = torch.load("model_state_dict.pt", map_location=device)
    model.load_state_dict(state_dict)
    
    return model.eval()

def predict(model, trajectory_features, keyboard_weights):
    """Run prediction on swipe data."""
    with torch.no_grad():
        encoder_input = (trajectory_features, keyboard_weights)
        encoded = model.encode(encoder_input, None)
        return encoded

if __name__ == "__main__":
    model = load_model()
    print(f"Model loaded successfully!")
    print(f"Accuracy: {accuracy}")
    print(f"Vocab size: {model_params['vocab_size']}")
    print(f"Model size: {size_mb:.1f}MB")
'''
    
    loader_path = output_dir / "load_model.py"
    with open(loader_path, 'w') as f:
        f.write(loader_script)
    
    print(f"✅ Loader script: {loader_path}")
    print(f"✅ Configuration: {config_path}")
    
    print(f"\n🎯 Export Summary")
    print("=" * 30)
    print(f"✅ Model: {model_name}")
    print(f"📊 Accuracy: {accuracy}")
    print(f"📝 Vocab size: {model_params['vocab_size']}")  
    print(f"💾 Size: {size_mb:.1f}MB")
    print(f"📁 Output: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())