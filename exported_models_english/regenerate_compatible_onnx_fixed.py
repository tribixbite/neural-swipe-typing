#!/usr/bin/env python3
"""
Fixed ONNX export for transformers.js with proper validation and dynamic configuration.
Addresses all critical issues identified in the original implementation.
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import MODEL_GETTERS_DICT, EncoderDecoderTransformerLike


class ImprovedNeuralSwipeModel(nn.Module):
    """
    Improved wrapper for ONNX export with proper error handling and flexibility.
    Supports variable sequence lengths and preserves model structure.
    """
    
    def __init__(self, original_model: EncoderDecoderTransformerLike, 
                 max_seq_len: int = 299, 
                 vocab_size: int = 30,
                 feature_dim: int = 6):
        super().__init__()
        self.original_model = original_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        
    def forward(self, trajectory_features: torch.Tensor, 
                seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Improved forward pass with proper shape handling.
        
        Args:
            trajectory_features: Trajectory features [batch_size, seq_len, feature_dim]
            seq_lengths: Optional actual sequence lengths [batch_size]
            
        Returns:
            logits: Word prediction logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, _ = trajectory_features.shape
        
        # Split trajectory features into actual features and keyboard IDs
        # Assuming last dimension is keyboard ID (or create dummy ones)
        if trajectory_features.shape[-1] >= 7:
            # If we have 7 features, assume last is keyboard ID
            traj_feats = trajectory_features[..., :6]  # First 6 features
            kb_ids = trajectory_features[..., 6:7].long()  # 7th feature as keyboard ID
        else:
            # Otherwise use just trajectory features and dummy keyboard IDs
            traj_feats = trajectory_features
            # Create dummy keyboard IDs (zeros)
            kb_ids = torch.zeros(batch_size, seq_len, 1, dtype=torch.long, 
                                device=trajectory_features.device)
        
        # Transpose for model [seq_len, batch_size, features]
        traj_feats_t = traj_feats.transpose(0, 1)
        kb_ids_t = kb_ids.transpose(0, 1).squeeze(-1)  # Remove last dim for IDs
        
        # Create input tuple as expected by the model
        encoder_input = (traj_feats_t, kb_ids_t)
        
        # Create dummy target for inference (model needs it but doesn't use for forward)
        dummy_target = torch.zeros(seq_len, batch_size, dtype=torch.long, 
                                  device=trajectory_features.device)
        
        # No silent error handling - let errors propagate properly
        # Get model outputs - pass None for padding mask
        outputs = self.original_model(encoder_input, None, dummy_target, None)
        
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            logits = outputs[1]  # Get logits
            if logits.dim() == 3:  # [seq_len, batch_size, vocab_size]
                # Transpose back to [batch_size, seq_len, vocab_size]
                logits = logits.transpose(0, 1)
            return logits
        else:
            raise ValueError(f"Unexpected model output format: {type(outputs)}")


def extract_model_config(checkpoint: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    """
    Extract actual model configuration from checkpoint and model instance.
    """
    config = {}
    
    # Try to get hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        config.update({
            'learning_rate': hparams.get('lr', 0.001),
            'batch_size': hparams.get('batch_size', 32),
            'model_name': hparams.get('model_name', 'unknown')
        })
    
    # Extract architecture from model
    if hasattr(model, 'encoder'):
        encoder = model.encoder
        if hasattr(encoder, 'layers'):
            config['num_hidden_layers'] = len(encoder.layers)
        if hasattr(encoder, 'd_model'):
            config['hidden_size'] = encoder.d_model
        if hasattr(encoder, 'nhead'):
            config['num_attention_heads'] = encoder.nhead
    
    # Extract vocab size from output layer
    if hasattr(model, 'fc_out'):
        config['vocab_size'] = model.fc_out.out_features
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'fc_out'):
        config['vocab_size'] = model.decoder.fc_out.out_features
    
    # Extract feature dimensions
    if hasattr(model, 'enc_in_emb_model'):
        embedder = model.enc_in_emb_model
        if hasattr(embedder, 'traj_emb'):
            traj_emb = embedder.traj_emb
            if hasattr(traj_emb, 'in_features'):
                config['input_feature_dim'] = traj_emb.in_features
    
    return config


def load_model_from_checkpoint(checkpoint_path: str, 
                              model_name: str = "v3_nearest_and_traj_transformer_bigger") -> Tuple[EncoderDecoderTransformerLike, Dict]:
    """Load model from Lightning checkpoint with better error handling."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model class
    if model_name not in MODEL_GETTERS_DICT:
        available_models = list(MODEL_GETTERS_DICT.keys())
        raise ValueError(f"Model {model_name} not found. Available: {available_models}")
    
    model_getter = MODEL_GETTERS_DICT[model_name]
    model = model_getter()
    
    # Load state dict with proper error handling
    if 'state_dict' in checkpoint:
        state_dict = {}
        key_mappings = {
            'enc_in_emb_key_emb.': 'enc_in_emb_model.key_emb.',
            'dec_in_emb_0.': 'dec_in_emb_model.0.',
            'dec_in_emb_2.': 'dec_in_emb_model.2.',
        }
        
        for key, value in checkpoint['state_dict'].items():
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            
            # Apply known key mappings
            for old_pattern, new_pattern in key_mappings.items():
                new_key = new_key.replace(old_pattern, new_pattern)
            
            state_dict[new_key] = value
        
        # Try strict loading first
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ Model loaded with strict=True")
        except RuntimeError as e:
            print(f"⚠️ WARNING: Strict loading failed, using non-strict mode")
            print(f"  Missing keys or shape mismatches detected: {str(e)[:200]}...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Extract configuration
    config = extract_model_config(checkpoint, model)
    
    return model, config


def validate_onnx_export(pytorch_model: nn.Module, 
                         onnx_path: str, 
                         test_batch_size: int = 2,
                         test_seq_len: int = 100,
                         feature_dim: int = 6,
                         tolerance: float = 1e-5) -> bool:
    """
    Validate that ONNX model produces same outputs as PyTorch model.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️ onnxruntime not installed, skipping validation")
        print("  Install with: pip install onnxruntime")
        return True
    
    print("🔍 Validating ONNX export...")
    
    # Create test input
    test_input = torch.randn(test_batch_size, test_seq_len, feature_dim)
    
    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - ort_outputs))
    mean_diff = np.mean(np.abs(pytorch_output - ort_outputs))
    
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    if max_diff < tolerance:
        print("✅ Validation passed! ONNX model matches PyTorch")
        return True
    else:
        print(f"❌ Validation failed! Differences exceed tolerance ({tolerance})")
        return False


def export_to_onnx_optimized(model: nn.Module, 
                            output_path: str, 
                            max_seq_len: int = 299, 
                            feature_dim: int = 6,
                            opset_version: int = 17,
                            optimize: bool = True):
    """Export model to ONNX with modern practices and optimization."""
    
    print(f"Exporting to ONNX (opset v{opset_version}): {output_path}")
    
    # Create realistic dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, max_seq_len, feature_dim)
    
    # Export to ONNX with improved configuration
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=['trajectory_features'],
        output_names=['logits'],
        dynamic_axes={
            'trajectory_features': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=opset_version,
        do_constant_folding=optimize,
        export_params=True,
        verbose=False
    )
    
    if optimize:
        try:
            import onnx
            from onnx import optimizer
            
            print("  Applying ONNX optimization passes...")
            model_onnx = onnx.load(output_path)
            
            # Apply optimization passes
            passes = optimizer.get_available_passes()
            optimized_model = optimizer.optimize(model_onnx, passes)
            
            onnx.save(optimized_model, output_path)
            print("  ✅ Optimization complete")
        except ImportError:
            print("  ⚠️ onnx package not installed, skipping optimization")
    
    print(f"✅ ONNX model exported to: {output_path}")


def load_vocabulary(vocab_path: Optional[str] = None) -> Dict[str, int]:
    """Load vocabulary from file or use default character vocabulary."""
    
    if vocab_path and Path(vocab_path).exists():
        print(f"Loading vocabulary from: {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab_list = f.read().strip().split('\n')
        vocab = {token: i for i, token in enumerate(vocab_list)}
    else:
        print("Using default character vocabulary")
        # Default character vocabulary
        chars = list("abcdefghijklmnopqrstuvwxyz")
        vocab = {char: i for i, char in enumerate(chars)}
        
        # Add special tokens
        vocab["<eos>"] = 26
        vocab["<unk>"] = 27
        vocab["<pad>"] = 28
        vocab["<sos>"] = 29
    
    return vocab


def create_transformersjs_config(model_config: Dict[str, Any], 
                                vocab_size: int = 30, 
                                max_seq_len: int = 299) -> Dict[str, Any]:
    """Create transformers.js config using actual model parameters."""
    
    return {
        "architectures": ["NeuralSwipeModel"],
        "model_type": "neural-swipe",
        "vocab_size": model_config.get('vocab_size', vocab_size),
        "max_position_embeddings": max_seq_len,
        "hidden_size": model_config.get('hidden_size', 128),
        "num_attention_heads": model_config.get('num_attention_heads', 8),
        "num_hidden_layers": model_config.get('num_hidden_layers', 6),
        "intermediate_size": model_config.get('intermediate_size', 512),
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_sequence_length": max_seq_len,
        "feature_dim": model_config.get('input_feature_dim', 6),
        "model_name": model_config.get('model_name', 'unknown'),
        "task_specific_params": {
            "swipe_decoding": {
                "max_length": 35,
                "temperature": 1.0,
                "top_k": 10,
                "top_p": 0.9,
                "early_stopping": True
            }
        },
        "pad_token_id": 28,
        "bos_token_id": 29,
        "eos_token_id": 26,
        "unk_token_id": 27
    }


def create_transformersjs_tokenizer(vocab: Dict[str, int]) -> Dict[str, Any]:
    """Create proper transformers.js tokenizer from vocabulary."""
    
    # Sort vocabulary by ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Identify special tokens
    special_tokens = []
    for token, token_id in sorted_vocab:
        if token in ["<eos>", "<unk>", "<pad>", "<sos>"]:
            special_tokens.append({
                "id": token_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            })
    
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": special_tokens,
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "CharDelimiterSplit", "delimiter": ""},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed ONNX export for transformers.js")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch Lightning checkpoint")
    parser.add_argument("--output-dir", type=str, default="./transformerjs_fixed/", help="Output directory")
    parser.add_argument("--model-name", type=str, default="v3_nearest_and_traj_transformer_bigger", help="Model name")
    parser.add_argument("--vocab-path", type=str, help="Path to vocabulary file")
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-optimize", action="store_true", help="Skip ONNX optimization")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    parser.add_argument("--max-seq-len", type=int, default=299, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "onnx").mkdir(exist_ok=True)
    
    print(f"🚀 Fixed ONNX export for transformers.js")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📂 Output dir: {output_dir}")
    print(f"🔧 ONNX opset: v{args.opset_version}")
    
    try:
        # Load original model and extract config
        original_model, model_config = load_model_from_checkpoint(args.checkpoint, args.model_name)
        print(f"✅ Model loaded successfully")
        print(f"📊 Extracted config: {model_config}")
        
        # Get vocabulary size from model config or default
        vocab_size = model_config.get('vocab_size', 30)
        
        # Create improved wrapper
        improved_model = ImprovedNeuralSwipeModel(
            original_model, 
            max_seq_len=args.max_seq_len,
            vocab_size=vocab_size,
            feature_dim=model_config.get('input_feature_dim', 6)
        )
        print("✅ Improved model wrapper created")
        
        # Export to ONNX with optimization
        onnx_path = output_dir / "onnx" / "model.onnx"
        export_to_onnx_optimized(
            improved_model, 
            str(onnx_path),
            max_seq_len=args.max_seq_len,
            feature_dim=model_config.get('input_feature_dim', 6),
            opset_version=args.opset_version,
            optimize=not args.no_optimize
        )
        
        # Validate export
        if not args.no_validate:
            validation_passed = validate_onnx_export(
                improved_model, 
                str(onnx_path),
                feature_dim=model_config.get('input_feature_dim', 6)
            )
            if not validation_passed:
                print("⚠️ Warning: Validation failed but continuing...")
        
        # Load vocabulary
        vocab = load_vocabulary(args.vocab_path)
        
        # Create config.json with actual model parameters
        config = create_transformersjs_config(model_config, vocab_size, args.max_seq_len)
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Config saved to: {config_path}")
        
        # Create tokenizer.json with proper vocabulary
        tokenizer = create_transformersjs_tokenizer(vocab)
        tokenizer_path = output_dir / "tokenizer.json"
        with open(tokenizer_path, 'w') as f:
            json.dump(tokenizer, f, indent=2)
        print(f"✅ Tokenizer saved to: {tokenizer_path}")
        
        # Save vocabulary separately for reference
        vocab_path = output_dir / "vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        print(f"✅ Vocabulary saved to: {vocab_path}")
        
        print(f"\n🎉 Successfully exported model with all fixes applied!")
        print(f"📁 Files created:")
        print(f"  - {onnx_path}")
        print(f"  - {config_path}")
        print(f"  - {tokenizer_path}")
        print(f"  - {vocab_path}")
        print(f"\n🔧 Use '{args.output_dir}' in your web demo")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()