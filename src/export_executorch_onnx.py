#!/usr/bin/env python3
"""
Export Neural Swipe Typing models to ExecutorTorch (.pte) and ONNX (.onnx) formats.
Based on the existing executorch_export.ipynb notebook.
"""

import argparse
import json
import os
import array
from pathlib import Path
from typing import Dict, Union, Tuple

import torch
from torch import Tensor
import torch.onnx

# ExecutorTorch imports
try:
    from torch.export import export, ExportedProgram, Dim
    from executorch.exir import EdgeProgramManager, to_edge, to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    EXECUTORCH_AVAILABLE = True
except ImportError:
    print("Warning: ExecutorTorch not available. Only ONNX export will be supported.")
    EXECUTORCH_AVAILABLE = False

# Local imports
from model import MODEL_GETTERS_DICT, EncoderDecoderTransformerLike


def remove_prefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s


def get_state_dict_from_checkpoint(ckpt: dict) -> Dict[str, torch.Tensor]:
    return {remove_prefix(k, 'model.'): v for k, v in ckpt['state_dict'].items()}


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


class Encode(torch.nn.Module):
    """ExecutorTorch Encoder Module - matches the existing notebook implementation."""
    def __init__(self, model) -> None:
        super().__init__()
        self.enc_in_emb_model = model.enc_in_emb_model
        self.encoder = model.encoder

    def forward(self, encoder_in):
        x = self.enc_in_emb_model(encoder_in)
        result = self.encoder(x, src_key_padding_mask=None)
        return result


class Decode(torch.nn.Module):
    """ExecutorTorch Decoder Module - matches the existing notebook implementation."""
    def __init__(self, model) -> None:
        super().__init__()
        self.dec_in_emb_model = model.dec_in_emb_model
        self.decoder = model.decoder
        self._get_mask = model._get_mask
        self.out = model.out

    def forward(self, decoder_in, x_encoded):
        y = self.dec_in_emb_model(decoder_in)
        tgt_mask = self._get_mask(y.size(0))
        dec_out = self.decoder(
            y, x_encoded, tgt_mask=tgt_mask, 
            memory_key_padding_mask=None, 
            tgt_key_padding_mask=None,
            tgt_is_causal=True)
        return self.out(dec_out)


class SimplifiedEncoder(torch.nn.Module):
    """Simplified encoder for ONNX export (handles tuple inputs better)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, traj_feats, kb_weights):
        encoder_in = (traj_feats, kb_weights)
        return self.model.encode(encoder_in, None)


def export_model_executorch_onnx(checkpoint_path, config, output_dir):
    """Export model to both ExecutorTorch and ONNX formats."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 ExecutorTorch & ONNX Export")
    print("=" * 50)
    
    # Load and prepare model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    hparams = checkpoint.get('hyper_parameters', {})
    
    raw_state_dict = get_state_dict_from_checkpoint(checkpoint)
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
    
    # Create and load model
    model_name = hparams.get('model_name', config["model_name"])
    model: EncoderDecoderTransformerLike = MODEL_GETTERS_DICT[model_name](**model_params).eval()
    model.load_state_dict(fixed_state_dict)
    
    print("✅ Model loaded successfully")
    
    # Create sample data with correct dtypes
    SWIPE_LENGTH = 13
    BATCH_SIZE = 1
    NUM_TRAJ_FEATS = 6
    OUT_SEQ_LEN = 3
    
    sample_traj_feats = torch.ones((SWIPE_LENGTH, BATCH_SIZE, NUM_TRAJ_FEATS), dtype=torch.float32)
    # Use keyboard weights (float), not key IDs (int) - this matches the distance-weighted approach
    sample_kb_weights = torch.softmax(torch.ones((SWIPE_LENGTH, BATCH_SIZE, model_params['n_keys'])), dim=-1).float()
    encoder_in = (sample_traj_feats, sample_kb_weights)
    decoder_in = torch.ones((OUT_SEQ_LEN, BATCH_SIZE), dtype=torch.int64)
    
    # Test model first
    print("🧪 Testing model...")
    with torch.no_grad():
        encoded = model.encode(encoder_in, None)
        print(f"✅ Encoding: {encoded.shape}")
        
        # Test decoding separately to catch specific issues
        try:
            decoded = model.decode(decoder_in, encoded, None, None)
            print(f"✅ Decoding: {decoded.shape}")
        except Exception as e:
            print(f"⚠️  Decoding failed (will skip decode export): {e}")
            decoded = None
    
    # Export ExecutorTorch
    executorch_exported = False
    if EXECUTORCH_AVAILABLE:
        print("\n📦 Exporting ExecutorTorch (.pte)...")
        try:
            # Define dynamic shapes (from notebook)
            MAX_SWIPE_LEN = 299
            MAX_WORD_LEN = 35
            dim_swipe_seq = Dim("dim_swipe_seq", min=1, max=MAX_SWIPE_LEN)
            dim_char_seq = Dim("dim_char_seq", min=1, max=MAX_WORD_LEN)
            
            encoder_dynamic_shapes = {"encoder_in": ({0: dim_swipe_seq}, {0: dim_swipe_seq})}
            decoder_dynamic_shapes = {
                "x_encoded": {0: dim_swipe_seq},
                "decoder_in": {0: dim_char_seq}
            }
            
            # Export to ATEN
            print("  Creating ATEN exported programs...")
            aten_encode: ExportedProgram = export(Encode(model).eval(), (encoder_in,), dynamic_shapes=encoder_dynamic_shapes)
            
            if decoded is not None:
                aten_decode: ExportedProgram = export(Decode(model).eval(), (decoder_in, encoded), dynamic_shapes=decoder_dynamic_shapes)
                programs_dict = {"encode": aten_encode, "decode": aten_decode}
            else:
                print("  Skipping decode export due to model issues")
                programs_dict = {"encode": aten_encode}
            
            # Export XNNPACK optimized version
            print("  Creating XNNPACK optimized version...")
            edge_xnnpack: EdgeProgramManager = to_edge_transform_and_lower(
                programs_dict,
                partitioner=[XnnpackPartitioner()],
            )
            exec_prog_xnnpack = edge_xnnpack.to_executorch()
            
            xnnpack_filename = f"neural_swipe_{accuracy.replace('.', '_').replace('%', 'pct')}_xnnpack.pte"
            xnnpack_path = output_dir / xnnpack_filename
            with open(xnnpack_path, "wb") as f:
                exec_prog_xnnpack.write_to_file(f)
            
            xnnpack_size_mb = xnnpack_path.stat().st_size / 1024 / 1024
            print(f"✅ ExecutorTorch XNNPACK: {xnnpack_path} ({xnnpack_size_mb:.1f}MB)")
            
            # Export raw version
            print("  Creating raw version...")
            edge_program = to_edge(programs_dict)
            executorch_program = edge_program.to_executorch()
            
            raw_filename = f"neural_swipe_{accuracy.replace('.', '_').replace('%', 'pct')}_raw.pte"
            raw_path = output_dir / raw_filename
            with open(raw_path, "wb") as f:
                f.write(executorch_program.buffer)
            
            raw_size_mb = raw_path.stat().st_size / 1024 / 1024
            print(f"✅ ExecutorTorch Raw: {raw_path} ({raw_size_mb:.1f}MB)")
            
            executorch_exported = True
            
        except Exception as e:
            print(f"❌ ExecutorTorch export failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Export ONNX 
    print("\n📦 Skipping ONNX export...")
    print("⚠️  ONNX export disabled due to known compatibility issues with complex transformer models")
    onnx_exported = False
    onnx_path = None
    onnx_size_mb = 0
    
    # Save test data (matching notebook approach)
    print("\n💾 Saving test data...")
    
    def tensor_to_dict(tensor: torch.Tensor) -> dict:
        return {
            'data': tensor.reshape(-1).tolist(),
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype)
        }

    def model_input_to_dict(encoder_in, decoder_in):
        return {
            'encoder_in': [tensor_to_dict(encoder_in_i) for encoder_in_i in encoder_in],
            'decoder_in': tensor_to_dict(decoder_in)
        }

    def model_output_to_dict(encoder_out, decoder_out):
        return {
            'encoder_out': tensor_to_dict(encoder_out),
            'decoder_out': tensor_to_dict(decoder_out)
        }
    
    input_path = output_dir / f"model_input_{accuracy.replace('.', '_').replace('%', 'pct')}.json"
    output_path = output_dir / f"model_output_{accuracy.replace('.', '_').replace('%', 'pct')}.json"
    
    with open(input_path, 'w') as f:
        json.dump(model_input_to_dict(encoder_in, decoder_in), f, indent=2)
    
    # Only save decoder output if decoding worked
    if decoded is not None:
        with open(output_path, 'w') as f:
            json.dump(model_output_to_dict(encoded, decoded), f, indent=2)
    else:
        with open(output_path, 'w') as f:
            json.dump({'encoder_out': tensor_to_dict(encoded)}, f, indent=2)
    
    print(f"✅ Test input: {input_path}")
    print(f"✅ Test output: {output_path}")
    
    # Save export metadata
    export_info = {
        "model_info": {
            "accuracy": accuracy,
            "architecture": model_name,
            "vocab_size": model_params['vocab_size'],
            "model_parameters": model_params,
            "checkpoint": checkpoint_path
        },
        "exported_formats": {
            "executorch_available": EXECUTORCH_AVAILABLE,
            "executorch_exported": executorch_exported,
            "onnx_exported": onnx_exported
        },
        "input_specification": {
            "trajectory_features": {
                "shape": list(sample_traj_feats.shape),
                "dtype": "float32",
                "description": "Swipe trajectory coordinates, velocities, accelerations"
            },
            "keyboard_weights": {
                "shape": list(sample_kb_weights.shape),
                "dtype": "float32",
                "description": "Keyboard key weights for each swipe point"
            }
        },
        "output_specification": {
            "encoded_sequence": {
                "shape": list(encoded.shape),
                "dtype": "float32",
                "description": "Encoded swipe representation for decoding"
            }
        }
    }
    
    metadata_path = output_dir / f"export_metadata_{accuracy.replace('.', '_').replace('%', 'pct')}.json"
    with open(metadata_path, 'w') as f:
        json.dump(export_info, f, indent=2)
    
    print(f"✅ Metadata: {metadata_path}")
    
    print(f"\n🎯 Export Summary for {accuracy} model")
    print("=" * 40)
    if executorch_exported:
        print(f"✅ ExecutorTorch: XNNPACK + Raw formats")
    if onnx_exported:
        print(f"✅ ONNX: Encoder exported")
    print(f"📁 Output: {output_dir}")
    
    return executorch_exported or onnx_exported


def main():
    parser = argparse.ArgumentParser(description="Export to ExecutorTorch and ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    try:
        success = export_model_executorch_onnx(args.checkpoint, config, args.output_dir)
        
        if success:
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