#!/usr/bin/env python3
"""
Export neural swipe typing model from Lightning checkpoint to ONNX and ExecuTorch formats
for deployment in Android keyboard app.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule

# Add src to path to import modules
sys.path.append('src')

from model import MODEL_GETTERS_DICT, EncoderDecoderTransformerLike
from pl_module import LitNeuroswipeModel
from ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizerv1
from feature_extraction.feature_extractors import get_val_transform


class SwipeModelWrapper(nn.Module):
    """Wrapper to make the model ONNX/ExecuTorch exportable by handling the complex input structure."""
    
    def __init__(self, model: EncoderDecoderTransformerLike, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name
    
    def forward(self, 
                traj_feats: torch.Tensor,
                kb_features: torch.Tensor,
                decoder_input: torch.Tensor, 
                encoder_padding_mask: torch.Tensor,
                decoder_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            traj_feats: Trajectory features [seq_len, batch_size, 6]
            kb_features: Keyboard features - either key IDs (int) or weights (float) [seq_len, batch_size, n_keys_or_30]
            decoder_input: Decoder input token IDs [seq_len, batch_size]
            encoder_padding_mask: Padding mask for encoder [batch_size, seq_len]
            decoder_padding_mask: Padding mask for decoder [batch_size, seq_len]
            
        Returns:
            logits: Output logits [seq_len, batch_size, vocab_size]
        """
        # Combine trajectory and keyboard features into the encoder input
        if "nearest" in self.model_name:
            # For nearest models, kb_features are key IDs (integers)
            encoder_input = (traj_feats, kb_features.long())
        elif "weighted" in self.model_name:
            # For weighted models, kb_features are weights (floats)
            encoder_input = (traj_feats, kb_features)
        else:
            # For other models, concatenate features
            encoder_input = torch.cat([traj_feats, kb_features], dim=-1)
        
        # The model.forward takes (x, y, x_pad_mask, y_pad_mask)
        return self.model.forward(
            encoder_input, 
            decoder_input, 
            encoder_padding_mask, 
            decoder_padding_mask
        )


def load_model_from_checkpoint(checkpoint_path: str, model_name: str = "v3_weighted_and_traj_transformer_bigger") -> EncoderDecoderTransformerLike:
    """Load the PyTorch Lightning model from checkpoint and extract the core model."""
    
    # Load the Lightning checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get the model constructor from the saved model_name or use provided one
    saved_model_name = checkpoint.get('hyper_parameters', {}).get('model_name', model_name)
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
    
    # Move model to CPU for export (ensures all parameters and buffers are on CPU)
    model = model.cpu()
    
    # Force all buffers to CPU (important for positional encodings)
    for module in model.modules():
        for buffer_name, buffer in module.named_buffers():
            if buffer.device != torch.device('cpu'):
                buffer.data = buffer.data.cpu()
                setattr(module, buffer_name, buffer.cpu())
    
    return model


def create_sample_inputs(model_name: str, batch_size: int = 1, seq_len: int = 100) -> Tuple[torch.Tensor, ...]:
    """Create sample inputs for tracing/export."""
    
    # Trajectory features are common to all models
    traj_feats = torch.randn(seq_len, batch_size, 6)  # 6 trajectory features
    
    # Different models have different keyboard feature formats
    if "nearest" in model_name:
        # For nearest models: key IDs (integers)
        kb_features = torch.randint(0, 30, (seq_len, batch_size))
    elif "weighted" in model_name:
        # For weighted models: key weights (floats)
        kb_features = torch.randn(seq_len, batch_size, 30)  # weights for 30 keys
    else:
        # Fallback: extended features
        kb_features = torch.randn(seq_len, batch_size, 122)  # 128 - 6 = 122
    
    # Decoder input: token IDs (start with SOS token = 1)
    decoder_input = torch.ones(35, batch_size, dtype=torch.long)  # max word length
    
    # Padding masks (False = not padded, True = padded)
    encoder_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    decoder_padding_mask = torch.zeros(batch_size, 35, dtype=torch.bool)
    
    return traj_feats, kb_features, decoder_input, encoder_padding_mask, decoder_padding_mask


def export_to_onnx(model: nn.Module, output_path: str, sample_inputs: Tuple[torch.Tensor, ...]):
    """Export model to ONNX format."""
    try:
        import onnx
    except ImportError:
        print("ONNX not installed. Installing...")
        os.system("pip install onnx onnxruntime")
        import onnx
    
    print(f"Exporting to ONNX: {output_path}")
    
    # Define input names and dynamic axes
    input_names = ['traj_feats', 'kb_features', 'decoder_input', 'encoder_padding_mask', 'decoder_padding_mask']
    output_names = ['logits']
    
    dynamic_axes = {
        'traj_feats': {0: 'encoder_seq_len', 1: 'batch_size'},
        'kb_features': {0: 'encoder_seq_len', 1: 'batch_size'},
        'decoder_input': {0: 'decoder_seq_len', 1: 'batch_size'},
        'encoder_padding_mask': {0: 'batch_size', 1: 'encoder_seq_len'},
        'decoder_padding_mask': {0: 'batch_size', 1: 'decoder_seq_len'},
        'logits': {0: 'decoder_seq_len', 1: 'batch_size'}
    }
    
    torch.onnx.export(
        model,
        sample_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX export successful: {output_path}")


def export_to_executorch(model: nn.Module, output_path: str, sample_inputs: Tuple[torch.Tensor, ...]):
    """Export model to ExecuTorch format."""
    try:
        from executorch.exir import to_edge
        from executorch.exir.dialects._ops import ops as exir_ops
        import executorch.exir as exir
    except ImportError:
        print("ExecuTorch not installed. Please install it following the official instructions.")
        print("For now, skipping ExecuTorch export...")
        return
    
    print(f"Exporting to ExecuTorch: {output_path}")
    
    try:
        # Capture the model using torch.export
        exported_model = torch.export.export(model, sample_inputs)
        
        # Convert to ExecuTorch Edge dialect
        edge_program = to_edge(exported_model)
        
        # Convert to ExecuTorch format
        executorch_program = edge_program.to_executorch()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(executorch_program.buffer)
        
        print(f"ExecuTorch export successful: {output_path}")
        
    except Exception as e:
        print(f"ExecuTorch export failed: {e}")
        print("This might be due to unsupported operations. Consider simplifying the model.")


def save_model_metadata(checkpoint_path: str, output_dir: str):
    """Save model metadata and configuration for Android app."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    metadata = {
        "model_architecture": "EncoderDecoderTransformerLike",
        "model_name": checkpoint.get('hyper_parameters', {}).get('model_name', 'v3_weighted_and_traj_transformer_bigger'),
        "vocab_size": 30,  # Character vocab size
        "max_encoder_seq_len": 299,  # Max trajectory sequence length
        "max_decoder_seq_len": 35,   # Max word length
        "d_model": 128,
        "input_features": {
            "trajectory_features": 6,  # x, y, dx/dt, dy/dt, d2x/dt2, d2y/dt2
            "keyboard_embedding": 122  # 128 - 6
        },
        "preprocessing": {
            "include_coords": True,
            "include_time": False,
            "include_velocities": True,
            "include_accelerations": True,
            "transform_name": "traj_feats_and_distance_weights"
        },
        "export_info": {
            "pytorch_version": torch.__version__,
            "checkpoint_epoch": checkpoint.get('epoch', 'unknown'),
            "validation_loss": checkpoint.get('val_loss', 'unknown'),
            "validation_accuracy": checkpoint.get('val_word_acc', 'unknown')
        }
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Export neural swipe typing model to ONNX and ExecuTorch formats')
    parser.add_argument('checkpoint_path', help='Path to the Lightning checkpoint file (.ckpt)')
    parser.add_argument('--output-dir', default='./exported_models', help='Output directory for exported models')
    parser.add_argument('--model-name', help='Model architecture name (auto-detected if not specified)')
    parser.add_argument('--formats', nargs='+', choices=['onnx', 'executorch', 'both'], default=['both'],
                       help='Export formats')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    
    # Load the model
    try:
        model = load_model_from_checkpoint(args.checkpoint_path, args.model_name)
        print("Model loaded successfully")
        
        # Get the actual model name used
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model_name = checkpoint.get('hyper_parameters', {}).get('model_name', args.model_name or 'v3_weighted_and_traj_transformer_bigger')
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Wrap the model for export
    wrapped_model = SwipeModelWrapper(model, model_name)
    wrapped_model.eval()
    
    # Create sample inputs
    sample_inputs = create_sample_inputs(model_name)
    print("Created sample inputs for export")
    
    # Test the model with sample inputs
    try:
        with torch.no_grad():
            output = wrapped_model(*sample_inputs)
            print(f"Model output shape: {output.shape}")
    except Exception as e:
        print(f"Error during model test: {e}")
        return 1
    
    # Export to requested formats
    base_name = Path(args.checkpoint_path).stem
    
    if 'onnx' in args.formats or 'both' in args.formats:
        onnx_path = output_dir / f"{base_name}.onnx"
        try:
            export_to_onnx(wrapped_model, str(onnx_path), sample_inputs)
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    if 'executorch' in args.formats or 'both' in args.formats:
        executorch_path = output_dir / f"{base_name}.pte"
        try:
            export_to_executorch(wrapped_model, str(executorch_path), sample_inputs)
        except Exception as e:
            print(f"ExecuTorch export failed: {e}")
    
    # Save model metadata
    save_model_metadata(args.checkpoint_path, str(output_dir))
    
    print("Export process completed!")
    return 0


if __name__ == '__main__':
    sys.exit(main())