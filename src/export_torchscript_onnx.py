#!/usr/bin/env python3
"""
Export Neural Swipe Typing models to TorchScript (.pt) and ONNX (.onnx) formats.
"""

import argparse
import json
import torch
import torch.onnx
from pathlib import Path

# Local imports  
from model import MODEL_GETTERS_DICT


def comprehensive_shape_fix(state_dict):
    """Fix tensor shapes for different model versions."""
    fixed_dict = {}
    
    for key, tensor in state_dict.items():
        if key == 'enc_in_emb_model.weighted_sum_emb.pos_encoder.pe':
            if tensor.shape == (2048, 1, 122):
                fixed_dict[key] = tensor.permute(1, 0, 2)
                print(f"Fixed {key}: {tensor.shape} -> {fixed_dict[key].shape}")
                continue
                
        elif key == 'dec_in_emb_model.2.pe':
            if tensor.shape == (29, 1, 128):
                fixed_dict[key] = tensor.permute(1, 0, 2)
                print(f"Fixed {key}: {tensor.shape} -> {fixed_dict[key].shape}")
                continue
        
        fixed_dict[key] = tensor
    
    return fixed_dict


class SimpleEncoderWrapper(torch.nn.Module):
    """Simplified encoder for reliable TorchScript/ONNX export."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, trajectory_features, keyboard_weights):
        """Encode swipe trajectory and keyboard weights."""
        encoder_input = (trajectory_features, keyboard_weights)
        return self.model.encode(encoder_input, src_key_padding_mask=None)


def export_model_formats(checkpoint_path, config, output_dir):
    """Export model to both TorchScript and ONNX formats."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 TorchScript & ONNX Export")
    print("=" * 50)
    
    # Load and prepare model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    hparams = checkpoint.get('hyper_parameters', {})
    
    def remove_prefix(s, prefix='model.'):
        return s[len(prefix):] if s.startswith(prefix) else s
    
    raw_state_dict = {remove_prefix(k): v for k, v in checkpoint['state_dict'].items()}
    fixed_state_dict = comprehensive_shape_fix(raw_state_dict)
    
    model_params = {
        'n_coord_feats': hparams.get('n_coord_feats', 6),
        'n_keys': hparams.get('n_keys', 29),
        'vocab_size': hparams.get('vocab_size', 32),
        'max_word_len': hparams.get('max_word_len', 30)
    }
    
    accuracy = "unknown"
    if "accuracy=" in checkpoint_path:
        accuracy = checkpoint_path.split("accuracy=")[1].split(".ckpt")[0]
    elif "0.707" in checkpoint_path:
        accuracy = "70.7%"
    elif "0.635" in checkpoint_path:
        accuracy = "63.5%"
    elif "0.625" in checkpoint_path:
        accuracy = "62.5%"
    
    print(f"📋 Model: {accuracy} accuracy")
    print(f"📋 Vocab size: {model_params['vocab_size']}")
    print(f"📋 Parameters: {model_params}")
    
    # Create and load model
    model_name = hparams.get('model_name', config["model_name"])
    model = MODEL_GETTERS_DICT[model_name](**model_params).eval()
    model.load_state_dict(fixed_state_dict)
    
    print("✅ Model loaded successfully")
    
    # Create sample data for tracing
    seq_len = 10
    sample_traj = torch.randn(seq_len, 1, 6)
    sample_kb = torch.softmax(torch.randn(seq_len, 1, 29), dim=-1)
    
    # Test model first
    print("🧪 Testing model...")
    with torch.no_grad():
        encoded = model.encode((sample_traj, sample_kb), None)
        print(f"✅ Model test passed: {encoded.shape}")
    
    # Export TorchScript
    print("\n📦 Exporting TorchScript (.pt)...")
    try:
        encoder_wrapper = SimpleEncoderWrapper(model)
        
        # Try tracing first
        try:
            traced_model = torch.jit.trace(encoder_wrapper, (sample_traj, sample_kb))
            torchscript_path = output_dir / f"neural_swipe_encoder_{accuracy.replace('.', '_').replace('%', 'pct')}.pt"
            traced_model.save(str(torchscript_path))
            
            ts_size_mb = torchscript_path.stat().st_size / 1024 / 1024
            print(f"✅ TorchScript (traced): {torchscript_path} ({ts_size_mb:.1f}MB)")
            
            # Test the traced model
            loaded_traced = torch.jit.load(str(torchscript_path))
            test_output = loaded_traced(sample_traj, sample_kb)
            print(f"✅ TorchScript test passed: {test_output.shape}")
            
        except Exception as e:
            print(f"⚠️  Tracing failed: {e}")
            print("Trying scripting...")
            
            scripted_model = torch.jit.script(encoder_wrapper)
            torchscript_path = output_dir / f"neural_swipe_encoder_scripted_{accuracy.replace('.', '_').replace('%', 'pct')}.pt"
            scripted_model.save(str(torchscript_path))
            
            ts_size_mb = torchscript_path.stat().st_size / 1024 / 1024
            print(f"✅ TorchScript (scripted): {torchscript_path} ({ts_size_mb:.1f}MB)")
            
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
        torchscript_path = None
        ts_size_mb = 0
    
    # Export ONNX
    print("\n📦 Exporting ONNX (.onnx)...")
    try:
        onnx_path = output_dir / f"neural_swipe_encoder_{accuracy.replace('.', '_').replace('%', 'pct')}.onnx"
        
        # Use simpler ONNX export settings
        torch.onnx.export(
            encoder_wrapper,
            (sample_traj, sample_kb),
            str(onnx_path),
            export_params=True,
            opset_version=11,  # Use stable opset version
            do_constant_folding=False,  # Disable to avoid issues
            input_names=['trajectory_features', 'keyboard_weights'],
            output_names=['encoded_sequence'],
            dynamic_axes={
                'trajectory_features': {0: 'sequence_length'},
                'keyboard_weights': {0: 'sequence_length'},
                'encoded_sequence': {0: 'sequence_length'}
            },
            verbose=False
        )
        
        if onnx_path.exists():
            onnx_size_mb = onnx_path.stat().st_size / 1024 / 1024
            print(f"✅ ONNX: {onnx_path} ({onnx_size_mb:.1f}MB)")
            
            # Test ONNX model
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_path))
                
                onnx_inputs = {
                    'trajectory_features': sample_traj.numpy(),
                    'keyboard_weights': sample_kb.numpy()
                }
                onnx_output = session.run(None, onnx_inputs)[0]
                print(f"✅ ONNX test passed: {onnx_output.shape}")
                
            except ImportError:
                print("⚠️  ONNX Runtime not available for testing")
            except Exception as e:
                print(f"⚠️  ONNX test failed: {e}")
        else:
            print("❌ ONNX file was not created")
            onnx_path = None
            onnx_size_mb = 0
            
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        onnx_path = None
        onnx_size_mb = 0
    
    # Save export metadata
    export_info = {
        "model_info": {
            "accuracy": accuracy,
            "architecture": model_name,
            "vocab_size": model_params['vocab_size'],
            "model_parameters": model_params
        },
        "exported_files": {
            "torchscript": str(torchscript_path) if torchscript_path else None,
            "torchscript_size_mb": ts_size_mb,
            "onnx": str(onnx_path) if onnx_path else None,
            "onnx_size_mb": onnx_size_mb
        },
        "input_specification": {
            "trajectory_features": {
                "shape": [seq_len, 1, 6],
                "description": "Swipe coordinates, velocities, accelerations"
            },
            "keyboard_weights": {
                "shape": [seq_len, 1, 29],
                "description": "Distance-weighted keyboard key probabilities"
            }
        },
        "output_specification": {
            "encoded_sequence": {
                "shape": [seq_len, 1, 128],
                "description": "Encoded swipe representation"
            }
        }
    }
    
    metadata_path = output_dir / f"export_info_{accuracy.replace('.', '_').replace('%', 'pct')}.json"
    with open(metadata_path, 'w') as f:
        json.dump(export_info, f, indent=2)
    
    print(f"✅ Metadata: {metadata_path}")
    
    # Create usage examples
    usage_examples = f'''
# {accuracy} Accuracy Neural Swipe Typing Model Usage

## TorchScript Usage
```python
import torch

# Load model
model = torch.jit.load("{torchscript_path.name if torchscript_path else 'model.pt'}")

# Prepare input
trajectory_features = torch.randn(10, 1, 6)  # [seq_len, batch, features]
keyboard_weights = torch.softmax(torch.randn(10, 1, 29), dim=-1)  # [seq_len, batch, keys]

# Run inference
with torch.no_grad():
    encoded = model(trajectory_features, keyboard_weights)
    print(f"Encoded shape: {{encoded.shape}}")  # [10, 1, 128]
```

## ONNX Usage
```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("{onnx_path.name if onnx_path else 'model.onnx'}")

# Prepare input
inputs = {{
    'trajectory_features': np.random.randn(10, 1, 6).astype(np.float32),
    'keyboard_weights': np.random.randn(10, 1, 29).astype(np.float32)
}}

# Run inference
outputs = session.run(None, inputs)
encoded = outputs[0]
print(f"Encoded shape: {{encoded.shape}}")  # (10, 1, 128)
```

## Android Integration (TorchScript)
```kotlin
// Load model
val module = LiteModuleLoader.load(torchscriptPath)

// Prepare tensors
val trajTensor = Tensor.fromBlob(trajectoryData, longArrayOf(seqLen, 1, 6))
val kbTensor = Tensor.fromBlob(keyboardWeights, longArrayOf(seqLen, 1, 29))

// Run inference
val encoded = module.forward(IValue.from(trajTensor), IValue.from(kbTensor)).toTensor()
```

## Model Specifications
- **Accuracy**: {accuracy}
- **Input**: Swipe trajectory (6D) + keyboard layout (29D)
- **Output**: Encoded sequence (128D)
- **Architecture**: Transformer encoder with distance-weighted embeddings
'''
    
    usage_path = output_dir / f"USAGE_{accuracy.replace('.', '_').replace('%', 'pct')}.md"
    with open(usage_path, 'w') as f:
        f.write(usage_examples)
    
    print(f"✅ Usage guide: {usage_path}")
    
    print(f"\n🎯 Export Summary for {accuracy} model")
    print("=" * 40)
    if torchscript_path:
        print(f"✅ TorchScript: {ts_size_mb:.1f}MB")
    if onnx_path:
        print(f"✅ ONNX: {onnx_size_mb:.1f}MB")
    print(f"📁 Output: {output_dir}")
    
    return torchscript_path, onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export to TorchScript and ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    try:
        torchscript_path, onnx_path = export_model_formats(args.checkpoint, config, args.output_dir)
        
        if torchscript_path or onnx_path:
            print("\n🎉 Export completed successfully!")
            return 0
        else:
            print("\n❌ Export failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Export failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())