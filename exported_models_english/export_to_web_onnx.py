#!/usr/bin/env python3
"""
Export PyTorch Lightning checkpoint to ONNX format for web deployment.
This creates a properly formatted ONNX model that can be loaded by ONNX Runtime Web.
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import MODEL_GETTERS_DICT, EncoderDecoderTransformerLike


class WebCompatibleSwipeModel(nn.Module):
    """
    Wrapper for the neural swipe model that's compatible with ONNX Runtime Web.
    This version handles the encoding phase only, as decoding will be done in JavaScript.
    """
    
    def __init__(self, original_model: EncoderDecoderTransformerLike):
        super().__init__()
        self.model = original_model
        self.max_seq_len = 299
        self.traj_feat_dim = 6  # x, y, vx, vy, ax, ay
        self.kb_feat_dim = 1     # keyboard token
        
    def forward(self, trajectory_features, keyboard_features):
        """
        Forward pass for encoding swipe trajectory.
        
        Args:
            trajectory_features: [batch_size, seq_len, 6] - trajectory features
            keyboard_features: [batch_size, seq_len, 1] - keyboard tokens (as float, will be converted)
            
        Returns:
            encoded: Encoded representation for decoding
        """
        batch_size = trajectory_features.shape[0]
        seq_len = trajectory_features.shape[1]
        
        # Transpose to [seq_len, batch_size, features] as expected by the model
        traj_feats = trajectory_features.transpose(0, 1)  # [seq_len, batch_size, 6]
        
        # Convert keyboard features to long integers for embedding lookup
        kb_feats = keyboard_features.long().transpose(0, 1)  # [seq_len, batch_size, 1]
        
        # Create padding mask (all False for now - no padding)
        pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        # Encode the trajectory
        encoded = self.model.encode((traj_feats, kb_feats), pad_mask)
        
        # For initial simplification, we'll output the encoder states
        # In a full implementation, we'd also handle the decoder
        
        # Get a simple prediction by passing through decoder once
        # Start with SOS token (29)
        sos_token = torch.tensor([[29]], dtype=torch.long)  # [1, batch_size]
        
        # Run one step of decoding to get initial logits
        logits = self.model.decode(
            sos_token,
            encoded,
            curve_pad_mask=pad_mask,
            word_pad_mask=None
        )
        
        # Return the logits for the first prediction
        # Shape: [batch_size, vocab_size]
        return logits.squeeze(0)  # Remove sequence dimension


def load_checkpoint(checkpoint_path: str, model_name: str = "v3_nearest_and_traj_transformer_bigger"):
    """Load model from PyTorch Lightning checkpoint."""
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model class
    if model_name not in MODEL_GETTERS_DICT:
        available = list(MODEL_GETTERS_DICT.keys())
        raise ValueError(f"Model {model_name} not found. Available: {available}")
    
    model_getter = MODEL_GETTERS_DICT[model_name]
    model = model_getter()
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            # Remove 'model.' prefix if present
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            
            # Fix naming inconsistencies in checkpoint
            new_key = new_key.replace('enc_in_emb_key_emb.', 'enc_in_emb_model.key_emb.')
            new_key = new_key.replace('dec_in_emb_0.', 'dec_in_emb_model.0.')
            new_key = new_key.replace('dec_in_emb_2.', 'dec_in_emb_model.2.')
            
            state_dict[new_key] = value
        
        # Load with relaxed strictness
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  Missing keys: {missing[:5]}...")  # Show first 5
        if unexpected:
            print(f"⚠️  Unexpected keys: {unexpected[:5]}...")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ Model loaded successfully")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    
    return model


def export_to_onnx(model, output_path: str, opset_version: int = 11):
    """Export model to ONNX format optimized for web."""
    
    print(f"🔄 Exporting to ONNX with opset version {opset_version}...")
    
    # Create wrapper model
    wrapped_model = WebCompatibleSwipeModel(model)
    wrapped_model.eval()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 299
    
    # Trajectory features: [batch_size, seq_len, 6]
    dummy_trajectory = torch.randn(batch_size, seq_len, 6)
    
    # Keyboard features: [batch_size, seq_len, 1] - must be integers for embedding
    dummy_keyboard = torch.randint(0, 26, (batch_size, seq_len, 1), dtype=torch.float32)
    
    # Export to ONNX
    try:
        torch.onnx.export(
            wrapped_model,
            (dummy_trajectory, dummy_keyboard),
            output_path,
            input_names=['trajectory_features', 'keyboard_features'],
            output_names=['logits'],
            dynamic_axes={
                'trajectory_features': {0: 'batch_size'},
                'keyboard_features': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        print(f"✅ ONNX model exported to: {output_path}")
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📦 Model size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise


def create_metadata(output_dir: Path, model_name: str):
    """Create metadata files for the web deployment."""
    
    metadata = {
        "model_type": "neural_swipe_typing",
        "architecture": model_name,
        "input_features": {
            "trajectory": ["x", "y", "vx", "vy", "ax", "ay"],
            "keyboard": ["nearest_key_index"]
        },
        "output": {
            "type": "logits",
            "vocab_size": 30,
            "special_tokens": {
                "eos": 26,
                "unk": 27,
                "pad": 28,
                "sos": 29
            }
        },
        "preprocessing": {
            "max_sequence_length": 299,
            "coordinate_normalization": "0_to_1",
            "velocity_calculation": "finite_difference",
            "acceleration_calculation": "finite_difference"
        },
        "char_to_idx": {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5,
            "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11,
            "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17,
            "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23,
            "y": 24, "z": 25,
            "<eos>": 26, "<unk>": 27, "<pad>": 28, "<sos>": 29
        }
    }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📝 Metadata saved to: {metadata_path}")
    
    # Create README for web deployment
    readme_content = """# Neural Swipe Typing - Web Deployment

## Model Files
- `model_web.onnx`: ONNX model for ONNX Runtime Web
- `model_metadata.json`: Model configuration and preprocessing info

## Usage

```javascript
// Load with ONNX Runtime Web
const session = await ort.InferenceSession.create('./model_web.onnx');

// Prepare inputs
const trajectoryFeatures = new Float32Array(299 * 6);  // [seq_len * 6]
const keyboardFeatures = new Float32Array(299 * 1);    // [seq_len * 1]

// Create tensors
const trajTensor = new ort.Tensor('float32', trajectoryFeatures, [1, 299, 6]);
const kbTensor = new ort.Tensor('float32', keyboardFeatures, [1, 299, 1]);

// Run inference
const results = await session.run({
    'trajectory_features': trajTensor,
    'keyboard_features': kbTensor
});

// Get logits
const logits = results.logits.data;
```

## Feature Extraction

Trajectory features for each point:
- x: normalized x coordinate (0-1)
- y: normalized y coordinate (0-1)  
- vx: x velocity (computed from consecutive points)
- vy: y velocity
- ax: x acceleration
- ay: y acceleration

Keyboard features:
- nearest_key_index: Index of nearest keyboard key (0-25)
"""
    
    readme_path = output_dir / "README_WEB.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📚 README saved to: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Export neural swipe model to ONNX for web")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints_english/english-epoch=51-val_loss=1.248-val_word_acc=0.659.ckpt",
        help="Path to PyTorch Lightning checkpoint"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="v3_nearest_and_traj_transformer_bigger",
        help="Model architecture name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./web_model/",
        help="Output directory for web model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (11 for max compatibility)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting ONNX export for web deployment...")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Load model from checkpoint
        model = load_checkpoint(args.checkpoint, args.model_name)
        
        # Export to ONNX
        onnx_path = output_dir / "model_web.onnx"
        export_to_onnx(model, str(onnx_path), args.opset)
        
        # Create metadata
        create_metadata(output_dir, args.model_name)
        
        print("\n🎉 Export completed successfully!")
        print(f"📂 Web model files created in: {output_dir}")
        print("\n🌐 To use in browser:")
        print(f"   1. Copy {output_dir} to your web server")
        print(f"   2. Load model_web.onnx with ONNX Runtime Web")
        print(f"   3. See README_WEB.md for usage instructions")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()