#!/usr/bin/env python3
"""
Improved ExecuTorch export for neural swipe typing model.
Addresses dynamic shape issues and transformer model complexities.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.export import Dim, export

# Add src to path to import modules
sys.path.append('src')

from model import MODEL_GETTERS_DICT, EncoderDecoderTransformerLike
from pl_module import LitNeuroswipeModel


class ExecuTorchCompatibleWrapper(nn.Module):
    """
    ExecuTorch-compatible wrapper that fixes dynamic shape issues in the transformer model.
    """
    
    def __init__(self, model: EncoderDecoderTransformerLike, model_name: str, max_seq_len: int = 35):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        
        # Pre-compute the triangular mask to avoid len() calls
        self.register_buffer('tgt_mask', self._get_static_mask(max_seq_len))
    
    def _get_static_mask(self, max_seq_len: int) -> torch.Tensor:
        """Create a static triangular mask for the decoder."""
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, 
                traj_feats: torch.Tensor,
                kb_features: torch.Tensor,
                decoder_input: torch.Tensor, 
                encoder_padding_mask: torch.Tensor,
                decoder_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that avoids dynamic shape operations.
        """
        # Get the actual sequence length from decoder input
        seq_len = decoder_input.shape[0]
        
        # Use pre-computed mask, slicing to the needed size
        current_mask = self.tgt_mask[:seq_len, :seq_len]
        
        # Combine trajectory and keyboard features into the encoder input
        if "nearest" in self.model_name:
            encoder_input = (traj_feats, kb_features.long())
        elif "weighted" in self.model_name:
            encoder_input = (traj_feats, kb_features)
        else:
            encoder_input = torch.cat([traj_feats, kb_features], dim=-1)
        
        # Encode the input
        encoded = self.model.encode(encoder_input, encoder_padding_mask)
        
        # Decode with static mask
        decoder_output = self._decode_with_static_mask(
            decoder_input, encoded, encoder_padding_mask, decoder_padding_mask, current_mask
        )
        
        return decoder_output
    
    def _decode_with_static_mask(self, y, x_encoded, memory_key_padding_mask, tgt_key_padding_mask, tgt_mask):
        """Decode with pre-computed static mask to avoid dynamic operations."""
        y = self.model.dec_in_emb_model(y)
        dec_out = self.model.decoder(y, x_encoded, tgt_mask=tgt_mask, 
                                   memory_key_padding_mask=memory_key_padding_mask, 
                                   tgt_key_padding_mask=tgt_key_padding_mask)
        return self.model.out(dec_out)


class InferenceOnlyWrapper(nn.Module):
    """
    Simplified inference-only wrapper for single-forward-pass export.
    This avoids the encoder-decoder complexity by combining everything into one forward pass.
    """
    
    def __init__(self, model: EncoderDecoderTransformerLike, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name
        
    def forward(self, 
                traj_feats: torch.Tensor,
                kb_features: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass for inference with fixed decoder input.
        Returns logits for the first predicted character only.
        """
        batch_size = traj_feats.shape[1]
        seq_len = traj_feats.shape[0]
        
        # Create fixed inputs for single-step inference
        decoder_input = torch.ones(1, batch_size, dtype=torch.long)  # SOS token only
        encoder_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        decoder_padding_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
        
        # Combine trajectory and keyboard features
        if "nearest" in self.model_name:
            encoder_input = (traj_feats, kb_features.long())
        elif "weighted" in self.model_name:
            encoder_input = (traj_feats, kb_features)
        else:
            encoder_input = torch.cat([traj_feats, kb_features], dim=-1)
        
        # Encode
        encoded = self.model.encode(encoder_input, encoder_padding_mask)
        
        # Decode first step only
        y = self.model.dec_in_emb_model(decoder_input)
        
        # No mask needed for single-step decoding
        dec_out = self.model.decoder(y, encoded, 
                                   memory_key_padding_mask=encoder_padding_mask, 
                                   tgt_key_padding_mask=decoder_padding_mask)
        logits = self.model.out(dec_out)
        
        return logits.squeeze(0)  # Remove sequence dimension for single-step


def load_model_from_checkpoint(checkpoint_path: str, model_name: str = None) -> Tuple[EncoderDecoderTransformerLike, str]:
    """Load the PyTorch Lightning model from checkpoint and extract the core model."""
    
    # Load the Lightning checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get the model constructor from the saved model_name
    saved_model_name = checkpoint.get('hyper_parameters', {}).get('model_name', model_name or 'v3_weighted_and_traj_transformer_bigger')
    print(f"Loading model: {saved_model_name}")
    
    if saved_model_name not in MODEL_GETTERS_DICT:
        raise ValueError(f"Unknown model name: {saved_model_name}. Available: {list(MODEL_GETTERS_DICT.keys())}")
    
    # Create the model using the getter function with CPU device
    model_getter = MODEL_GETTERS_DICT[saved_model_name]
    
    # Force CPU device by temporarily disabling CUDA
    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        model = model_getter(device='cpu')  # Load on CPU for export
    finally:
        torch.cuda.is_available = original_cuda_available
    
    # Extract the core model state dict from Lightning wrapper
    core_model_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            # Remove the 'model.' prefix to match the core model structure
            new_key = key[6:]
            core_model_state_dict[new_key] = value
    
    # Load the state dict into the core model
    model.load_state_dict(core_model_state_dict)
    model.eval()
    
    # Move model to CPU for export
    model = model.cpu()
    
    # Force all buffers to CPU
    for module in model.modules():
        for buffer_name, buffer in module.named_buffers():
            if buffer.device != torch.device('cpu'):
                buffer.data = buffer.data.cpu()
                setattr(module, buffer_name, buffer.cpu())
    
    return model, saved_model_name


def create_sample_inputs_for_executorch(model_name: str, approach: str = "full", batch_size: int = 1, seq_len: int = 100) -> Tuple[torch.Tensor, ...]:
    """Create sample inputs optimized for ExecuTorch export."""
    
    if approach == "inference_only":
        # Simplified inputs for inference-only wrapper
        traj_feats = torch.randn(seq_len, batch_size, 6)
        
        if "nearest" in model_name:
            kb_features = torch.randint(0, 30, (seq_len, batch_size))
        elif "weighted" in model_name:
            kb_features = torch.randn(seq_len, batch_size, 30)
        else:
            kb_features = torch.randn(seq_len, batch_size, 122)
        
        return traj_feats, kb_features
    
    else:
        # Full inputs for complete wrapper
        traj_feats = torch.randn(seq_len, batch_size, 6)
        
        if "nearest" in model_name:
            kb_features = torch.randint(0, 30, (seq_len, batch_size))
        elif "weighted" in model_name:
            kb_features = torch.randn(seq_len, batch_size, 30)
        else:
            kb_features = torch.randn(seq_len, batch_size, 122)
        
        decoder_input = torch.ones(35, batch_size, dtype=torch.long)
        encoder_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        decoder_padding_mask = torch.zeros(batch_size, 35, dtype=torch.bool)
        
        return traj_feats, kb_features, decoder_input, encoder_padding_mask, decoder_padding_mask


def export_to_executorch_improved(model: nn.Module, output_path: str, sample_inputs: Tuple[torch.Tensor, ...], 
                                dynamic_shapes: Dict = None, approach: str = "full"):
    """Export model to ExecuTorch format with improved handling of transformer issues."""
    
    try:
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from torch.export import export
    except ImportError as e:
        print(f"ExecuTorch not installed or missing components: {e}")
        print("Please install ExecuTorch following the official instructions.")
        return False
    
    print(f"Exporting to ExecuTorch ({approach} approach): {output_path}")
    
    try:
        # Export the model with dynamic shapes if provided
        print("Step 1: Exporting model with torch.export...")
        if dynamic_shapes:
            exported_program = export(model, sample_inputs, dynamic_shapes=dynamic_shapes)
        else:
            exported_program = export(model, sample_inputs)
        
        print("Step 2: Converting to ExecuTorch Edge dialect...")
        # Convert to ExecuTorch with XNNPACK backend for mobile CPU optimization
        executorch_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()]
        ).to_executorch()
        
        print("Step 3: Saving to file...")
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(executorch_program.buffer)
        
        print(f"ExecuTorch export successful: {output_path}")
        return True
        
    except Exception as e:
        print(f"ExecuTorch export failed: {e}")
        print("Error details:")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Export neural swipe typing model to ExecuTorch format')
    parser.add_argument('checkpoint_path', help='Path to the Lightning checkpoint file (.ckpt)')
    parser.add_argument('--output-dir', default='./exported_models_executorch', help='Output directory for exported models')
    parser.add_argument('--model-name', help='Model architecture name (auto-detected if not specified)')
    parser.add_argument('--approach', choices=['full', 'inference_only'], default='inference_only',
                       help='Export approach: full (complete model) or inference_only (simplified)')
    parser.add_argument('--backend', choices=['xnnpack', 'cpu'], default='xnnpack',
                       help='Backend to optimize for')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    
    # Load the model
    try:
        model, model_name = load_model_from_checkpoint(args.checkpoint_path, args.model_name)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Choose wrapper based on approach
    if args.approach == "inference_only":
        print("Using inference-only wrapper (simplified single-step prediction)")
        wrapped_model = InferenceOnlyWrapper(model, model_name)
        sample_inputs = create_sample_inputs_for_executorch(model_name, "inference_only")
        
        # Dynamic shapes for inference-only approach
        dynamic_shapes = {
            "traj_feats": {0: Dim("seq_len", min=10, max=299)},
            "kb_features": {0: Dim("seq_len", min=10, max=299)},
        }
        
    else:
        print("Using full wrapper (complete encoder-decoder)")
        wrapped_model = ExecuTorchCompatibleWrapper(model, model_name)
        sample_inputs = create_sample_inputs_for_executorch(model_name, "full")
        
        # Dynamic shapes for full approach - must match all input arguments
        dynamic_shapes = {
            "traj_feats": {0: Dim("seq_len", min=10, max=299)},
            "kb_features": {0: Dim("seq_len", min=10, max=299)},
            "decoder_input": {},  # Fixed shape
            "encoder_padding_mask": {1: Dim("seq_len", min=10, max=299)},  # batch x seq_len
            "decoder_padding_mask": {},  # Fixed shape
        }
    
    wrapped_model.eval()
    
    # Test the model with sample inputs
    print("Testing model with sample inputs...")
    try:
        with torch.no_grad():
            output = wrapped_model(*sample_inputs)
            print(f"Model output shape: {output.shape}")
    except Exception as e:
        print(f"Error during model test: {e}")
        return 1
    
    # Export to ExecuTorch
    base_name = Path(args.checkpoint_path).stem
    executorch_path = output_dir / f"{base_name}_{args.approach}.pte"
    
    success = export_to_executorch_improved(
        wrapped_model, 
        str(executorch_path), 
        sample_inputs,
        dynamic_shapes=dynamic_shapes,
        approach=args.approach
    )
    
    if success:
        # Save metadata
        metadata = {
            "model_architecture": "EncoderDecoderTransformerLike", 
            "model_name": model_name,
            "export_approach": args.approach,
            "backend": args.backend,
            "file_path": str(executorch_path),
            "input_format": {
                "inference_only": {
                    "traj_feats": "[seq_len, batch_size, 6]",
                    "kb_features": "[seq_len, batch_size, num_keys]"
                },
                "full": {
                    "traj_feats": "[seq_len, batch_size, 6]", 
                    "kb_features": "[seq_len, batch_size, num_keys]",
                    "decoder_input": "[35, batch_size]",
                    "encoder_padding_mask": "[batch_size, seq_len]",
                    "decoder_padding_mask": "[batch_size, 35]"
                }
            }[args.approach],
            "output_format": {
                "inference_only": "[batch_size, vocab_size] (first character logits only)",
                "full": "[35, batch_size, vocab_size] (complete sequence logits)"
            }[args.approach],
            "dynamic_shapes": str(dynamic_shapes),  # Convert to string for JSON serialization
            "pytorch_version": torch.__version__
        }
        
        metadata_path = output_dir / f"{base_name}_{args.approach}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved: {metadata_path}")
        print("ExecuTorch export completed successfully!")
        return 0
    else:
        print("ExecuTorch export failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())