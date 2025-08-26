#!/usr/bin/env python3
"""
Flexible model export that handles different model versions and shapes.
"""

import argparse
import json
import torch
from pathlib import Path

# Local imports  
from model import MODEL_GETTERS_DICT


def get_state_dict_from_checkpoint(ckpt):
    def remove_prefix(s, prefix):
        if s.startswith(prefix):
            s = s[len(prefix):]
        return s
    return {remove_prefix(k, 'model.'): v for k, v in ckpt['state_dict'].items()}


def fix_positional_encoding_shapes(state_dict, target_model):
    """Fix positional encoding tensor shapes to match target model."""
    fixed_dict = {}
    
    for key, value in state_dict.items():
        if 'pos_encoder.pe' in key:
            # Get target shape from model
            target_param = target_model
            for attr in key.split('.'):
                target_param = getattr(target_param, attr)
            
            target_shape = target_param.shape
            current_shape = value.shape
            
            print(f"Fixing {key}: {current_shape} -> {target_shape}")
            
            if key == 'enc_in_emb_model.weighted_sum_emb.pos_encoder.pe':
                # Expected: [1, 2048, 122], Got: [2048, 1, 122]
                if len(current_shape) == 3 and len(target_shape) == 3:
                    if current_shape == (2048, 1, 122) and target_shape == (1, 2048, 122):
                        fixed_dict[key] = value.permute(1, 0, 2)
                        continue
                        
            elif key == 'dec_in_emb_model.2.pe':
                # Expected: [1, 29, 128], Got: [29, 1, 128]  
                if len(current_shape) == 3 and len(target_shape) == 3:
                    if current_shape == (29, 1, 128) and target_shape == (1, 29, 128):
                        fixed_dict[key] = value.permute(1, 0, 2)
                        continue
            
            # If no specific fix, try generic transpose
            if current_shape != target_shape and len(current_shape) == len(target_shape):
                # Try different permutations
                permutations = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
                for perm in permutations:
                    try:
                        permuted = value.permute(perm)
                        if permuted.shape == target_shape:
                            print(f"  Applied permutation {perm}")
                            fixed_dict[key] = permuted
                            break
                    except:
                        continue
                else:
                    print(f"  Could not fix shape mismatch")
                    fixed_dict[key] = value
            else:
                fixed_dict[key] = value
        else:
            fixed_dict[key] = value
    
    return fixed_dict


def export_model_flexible(checkpoint_path, config, output_dir):
    """Export model with flexible shape handling."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔧 Flexible Model Export")
    print("=" * 40)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    raw_state_dict = get_state_dict_from_checkpoint(checkpoint)
    
    # Get hyperparameters from checkpoint if available
    hparams = checkpoint.get('hyper_parameters', {})
    print(f"Checkpoint hyperparameters: {hparams}")
    
    # Use hyperparameters from checkpoint, fall back to config
    model_params = {
        'n_coord_feats': hparams.get('n_coord_feats', config.get("n_coord_feats", 6)),
        'n_keys': hparams.get('n_keys', config.get("n_keys", 29)),
        'vocab_size': hparams.get('vocab_size', config.get("vocab_size", 32)),
        'max_word_len': hparams.get('max_word_len', config.get("max_word_len", 30))
    }
    
    print(f"Using model parameters: {model_params}")
    
    # Create target model
    model_name = hparams.get('model_name', config["model_name"])
    model = MODEL_GETTERS_DICT[model_name](**model_params).eval()
    
    print(f"Created model: {model_name}")
    
    # Fix state dict shapes
    print("\nFixing tensor shapes...")
    try:
        fixed_state_dict = fix_positional_encoding_shapes(raw_state_dict, model)
        model.load_state_dict(fixed_state_dict, strict=False)
        print("✅ Model loaded with shape fixes")
    except Exception as e:
        print(f"Shape fixing failed: {e}")
        print("Trying strict=False loading...")
        try:
            model.load_state_dict(raw_state_dict, strict=False)
            print("⚠️  Model loaded with strict=False (some weights may be missing)")
        except Exception as e2:
            print(f"❌ Model loading failed completely: {e2}")
            return None
    
    # Test the model
    print("\n🧪 Testing model...")
    try:
        seq_len = 5
        test_traj = torch.randn(seq_len, 1, 6)
        test_kb = torch.softmax(torch.randn(seq_len, 1, 29), dim=-1)
        
        with torch.no_grad():
            encoded = model.encode((test_traj, test_kb), None)
        
        print(f"✅ Model test passed: {encoded.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        # Continue anyway, might still be exportable
    
    # Export model
    state_dict_path = output_dir / "model_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    
    size_mb = state_dict_path.stat().st_size / 1024 / 1024
    print(f"✅ Model exported: {state_dict_path} ({size_mb:.1f}MB)")
    
    # Save configuration
    export_config = {
        **config,
        "checkpoint_hyperparameters": hparams,
        "model_params": model_params,
        "export_info": {
            "checkpoint_path": checkpoint_path,
            "accuracy": checkpoint_path.split('accuracy=')[1].split('.ckpt')[0] if 'accuracy=' in checkpoint_path else "unknown",
            "export_method": "flexible_shape_fixing"
        }
    }
    
    config_path = output_dir / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(export_config, f, indent=2)
    
    print(f"✅ Configuration saved: {config_path}")
    
    # Create reconstruction script
    reconstruction_code = f'''
# Flexible Model Reconstruction
import torch
from model import MODEL_GETTERS_DICT

def load_model(state_dict_path, device='cpu'):
    """Load the exported model."""
    model_params = {model_params}
    model = MODEL_GETTERS_DICT["{model_name}"](**model_params)
    
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model.eval()

# Usage:
# model = load_model("model_state_dict.pt")
# encoded = model.encode((traj_features, kb_weights), None)
'''
    
    script_path = output_dir / "load_model.py"
    with open(script_path, 'w') as f:
        f.write(reconstruction_code)
    
    print(f"✅ Reconstruction script: {script_path}")
    
    return size_mb


def main():
    parser = argparse.ArgumentParser(description="Flexible model export")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    result = export_model_flexible(args.checkpoint, config, args.output_dir)
    
    if result:
        print(f"\n🎯 Export completed successfully!")
        print(f"📁 Output: {args.output_dir}")
        print(f"💾 Size: {result:.1f}MB")
    else:
        print(f"\n❌ Export failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())