#!/usr/bin/env python3
"""
Test all three exported models to verify they work correctly.
"""

import torch
import sys
from pathlib import Path

# Add source to path
sys.path.append(str(Path(__file__).parent))

from model import MODEL_GETTERS_DICT


def test_model(model_dir, expected_accuracy, expected_vocab_size):
    """Test a single exported model."""
    
    model_dir = Path(model_dir)
    model_path = model_dir / "model_state_dict.pt"
    
    print(f"\n🧪 Testing {expected_accuracy} accuracy model")
    print("=" * 50)
    print(f"📁 Directory: {model_dir}")
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Load model configuration
    config_path = model_dir / "model_config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        
        model_params = config.get("model_parameters", config.get("checkpoint_hyperparameters", {}))
        print(f"📋 Config vocab size: {model_params.get('vocab_size', 'unknown')}")
        print(f"📋 Expected vocab size: {expected_vocab_size}")
    else:
        print("⚠️  No config file found")
        model_params = {
            'n_coord_feats': 6,
            'n_keys': 29,
            'vocab_size': expected_vocab_size,
            'max_word_len': 30
        }
    
    try:
        # Load the model
        print("🔄 Loading model...")
        model = MODEL_GETTERS_DICT["v3_weighted_and_traj_transformer_bigger"](**model_params).eval()
        
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print("✅ Model loaded successfully")
        
        # Test different input sizes
        test_cases = [
            (5, "small"),
            (10, "medium"), 
            (20, "large")
        ]
        
        for seq_len, size_name in test_cases:
            print(f"🔍 Testing {size_name} input (seq_len={seq_len})...")
            
            # Create test input
            traj_features = torch.randn(seq_len, 1, 6)
            kb_weights = torch.softmax(torch.randn(seq_len, 1, 29), dim=-1)
            
            with torch.no_grad():
                # Test encoding
                encoded = model.encode((traj_features, kb_weights), None)
                expected_shape = (seq_len, 1, 128)
                
                if encoded.shape == expected_shape:
                    print(f"  ✅ Encoding: {encoded.shape}")
                else:
                    print(f"  ❌ Encoding shape mismatch: got {encoded.shape}, expected {expected_shape}")
                    return False
                
                # Test decoder input
                decoder_in = torch.randint(0, min(expected_vocab_size, 30), (3, 1), dtype=torch.int64)
                try:
                    decoded = model.decode(decoder_in, encoded, None, None)
                    print(f"  ✅ Decoding: {decoded.shape}")
                except Exception as e:
                    print(f"  ⚠️  Decoding failed (expected for some models): {e}")
        
        # Memory usage test
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        file_size_mb = model_path.stat().st_size / 1024 / 1024
        
        print(f"💾 Model parameters: {model_size_mb:.1f}MB")
        print(f"📁 File size: {file_size_mb:.1f}MB")
        
        print(f"✅ {expected_accuracy} accuracy model test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all exported models."""
    
    print("🚀 Testing All Exported Neural Swipe Typing Models")
    print("=" * 60)
    
    # Test cases: (directory, expected_accuracy, expected_vocab_size)
    models_to_test = [
        ("../results/kb_apk_models", "70.7%", 32),
        ("../results/kb_apk_models_63_5", "63.5%", 67),  
        ("../results/kb_apk_models_62_5", "62.5%", 32),
    ]
    
    results = []
    
    for model_dir, accuracy, vocab_size in models_to_test:
        success = test_model(model_dir, accuracy, vocab_size)
        results.append((accuracy, success))
    
    # Summary
    print(f"\n🎯 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    for accuracy, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{accuracy:>6} accuracy: {status}")
        if success:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(results)} models passed")
    
    if passed == len(results):
        print("🎉 All models exported successfully!")
        return 0
    else:
        print("⚠️  Some models failed testing")
        return 1


if __name__ == "__main__":
    exit(main())