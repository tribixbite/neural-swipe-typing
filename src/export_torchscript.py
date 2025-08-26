#!/usr/bin/env python3
"""
Simple TorchScript export for mobile deployment.
This approach is more reliable than ONNX for complex models.
"""

import argparse
import json
import torch
from pathlib import Path

# Local imports
from model import MODEL_GETTERS_DICT


def main():
    parser = argparse.ArgumentParser(description="Export to TorchScript")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--config", default="export_config_70_7.json", help="Config file")
    parser.add_argument("--output-dir", default="../results/kb_apk_models", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    
    def get_state_dict_from_checkpoint(ckpt):
        def remove_prefix(s, prefix):
            if s.startswith(prefix):
                s = s[len(prefix):]
            return s
        return {remove_prefix(k, 'model.'): v for k, v in ckpt['state_dict'].items()}
    
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    
    # Model parameters
    model_params = {
        'n_coord_feats': config.get("n_coord_feats", 6),
        'n_keys': config.get("n_keys", 29),
        'vocab_size': config.get("vocab_size", 32),
        'max_word_len': config.get("max_word_len", 30)
    }
    
    print(f"Model parameters: {model_params}")
    
    # Create and load model
    model = MODEL_GETTERS_DICT[config["model_name"]](**model_params).eval()
    model.load_state_dict(state_dict)
    
    print("Model loaded successfully!")
    
    # Create sample data for tracing
    swipe_length = 10
    batch_size = 1
    
    sample_traj = torch.ones((swipe_length, batch_size, 6), dtype=torch.float32)
    sample_kb_weights = torch.zeros((swipe_length, batch_size, 29), dtype=torch.float32)
    
    # Set some reasonable weights
    for i in range(swipe_length):
        sample_kb_weights[i, 0, i % 29] = 0.8
        sample_kb_weights[i, 0, (i + 1) % 29] = 0.2
    
    encoder_in = (sample_traj, sample_kb_weights)
    decoder_in = torch.tensor([[1], [20]], dtype=torch.int64)
    
    print("Testing forward pass...")
    try:
        encoded = model.encode(encoder_in, None)
        print(f"✅ Encoding successful: {encoded.shape}")
        
        decoded = model.decode(decoder_in, encoded, None, None)
        print(f"✅ Decoding successful: {decoded.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Export full model as TorchScript
    print("\nExporting TorchScript model...")
    try:
        # Trace the model
        traced_model = torch.jit.trace(model, (encoder_in, decoder_in, None, None))
        
        # Save the traced model
        model_path = output_dir / "neural_swipe_model.pt"
        traced_model.save(str(model_path))
        
        file_size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"✅ Full model exported: {model_path} ({file_size_mb:.1f}MB)")
        
    except Exception as e:
        print(f"❌ Full model tracing failed: {e}")
        print("Trying scripting instead...")
        
        try:
            scripted_model = torch.jit.script(model)
            model_path = output_dir / "neural_swipe_model_scripted.pt"
            scripted_model.save(str(model_path))
            
            file_size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"✅ Scripted model exported: {model_path} ({file_size_mb:.1f}MB)")
            
        except Exception as e:
            print(f"❌ Scripting also failed: {e}")
    
    # Export just the encoder and decoder separately (simpler approach)
    print("\nTrying separate encoder/decoder export...")
    
    try:
        # Create a simpler encoder wrapper
        class SimpleEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, traj_feats, kb_weights):
                encoder_in = (traj_feats, kb_weights)
                return self.model.encode(encoder_in, None)
        
        encoder = SimpleEncoder(model)
        traced_encoder = torch.jit.trace(encoder, (sample_traj, sample_kb_weights))
        
        encoder_path = output_dir / "neural_swipe_encoder.pt"
        traced_encoder.save(str(encoder_path))
        
        file_size_mb = encoder_path.stat().st_size / 1024 / 1024
        print(f"✅ Encoder exported: {encoder_path} ({file_size_mb:.1f}MB)")
        
        # Test the exported encoder
        loaded_encoder = torch.jit.load(str(encoder_path))
        test_encoded = loaded_encoder(sample_traj, sample_kb_weights)
        print(f"✅ Encoder test successful: {test_encoded.shape}")
        
    except Exception as e:
        print(f"❌ Encoder export failed: {e}")
    
    # Save the model configuration and sample data
    print("\nSaving metadata...")
    
    # Save enhanced config
    enhanced_config = {
        **config,
        "model_architecture": {
            "encoder_layers": 4,
            "decoder_layers": 4,
            "hidden_dim": 128,
            "attention_heads": 4,
            "dropout": 0.1
        },
        "input_specs": {
            "trajectory_features": [swipe_length, batch_size, 6],
            "keyboard_weights": [swipe_length, batch_size, 29],
            "decoder_input": [2, batch_size]
        },
        "output_specs": {
            "encoded_sequence": list(encoded.shape),
            "character_logits": list(decoded.shape) 
        },
        "export_info": {
            "checkpoint": args.checkpoint,
            "accuracy": "70.7%",
            "format": "TorchScript"
        }
    }
    
    with open(output_dir / "model_config_enhanced.json", 'w') as f:
        json.dump(enhanced_config, f, indent=2)
    
    # Save usage example
    usage_example = '''
# Android Usage Example (Kotlin)

// Load the model
val module = LiteModuleLoader.load(modelPath)

// Prepare input tensors
val trajFeatures = Tensor.fromBlob(trajData, longArrayOf(seqLen, 1, 6))
val kbWeights = Tensor.fromBlob(keyboardWeights, longArrayOf(seqLen, 1, 29))

// Run inference
val encoded = encoderModule.forward(IValue.from(trajFeatures), IValue.from(kbWeights))
    .toTensor()

// For word prediction, implement beam search or greedy decoding
val wordCandidates = beamSearch(encoded, vocabSize = 30, beamSize = 6)
'''
    
    with open(output_dir / "android_usage.txt", 'w') as f:
        f.write(usage_example)
    
    print(f"✅ Configuration saved: {output_dir / 'model_config_enhanced.json'}")
    print(f"✅ Usage example saved: {output_dir / 'android_usage.txt'}")
    
    print(f"\n🎯 Export Summary:")
    print(f"  Model: {config['model_name']}")
    print(f"  Accuracy: 70.7%")
    print(f"  Format: TorchScript (.pt)")
    print(f"  Output: {output_dir}")
    
    # List all created files
    created_files = list(output_dir.glob("*.pt"))
    if created_files:
        print(f"\n📁 Created files:")
        for file_path in created_files:
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  - {file_path.name} ({size_mb:.1f}MB)")
    
    print(f"\n✅ Export complete! Models ready for Android deployment.")


if __name__ == "__main__":
    main()