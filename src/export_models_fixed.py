#!/usr/bin/env python3
"""
Fixed model export script that works around ONNX export issues.
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


class SimpleEncodeModule(torch.nn.Module):
    """Simplified encoder module that avoids ONNX export issues."""
    
    def __init__(self, model):
        super().__init__()
        self.enc_in_emb_model = model.enc_in_emb_model
        self.encoder = model.encoder

    def forward(self, trajectory_features, keyboard_weights):
        # Combine inputs manually to avoid tuple issues
        x_traj = trajectory_features
        x_kb = self.enc_in_emb_model.weighted_sum_emb(keyboard_weights)
        
        # Concatenate features (avoiding tuple unpacking in ONNX)
        x_combined = torch.cat([x_traj, x_kb], dim=-1)
        
        # Apply encoder
        result = self.encoder(x_combined, src_key_padding_mask=None)
        return result


class SimpleDecodeModule(torch.nn.Module):
    """Simplified decoder module."""
    
    def __init__(self, model):
        super().__init__()
        self.dec_in_emb_model = model.dec_in_emb_model
        self.decoder = model.decoder
        self.out = model.out
        self.max_len = 30

    def forward(self, decoder_input, encoded_sequence):
        # Simple embedding without mask generation
        y = self.dec_in_emb_model(decoder_input)
        
        # Create simple causal mask
        seq_len = y.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Apply decoder
        dec_out = self.decoder(
            y, encoded_sequence,
            tgt_mask=mask,
            memory_key_padding_mask=None,
            tgt_key_padding_mask=None,
            tgt_is_causal=True
        )
        
        return self.out(dec_out)


def export_pytorch_model(checkpoint_path, config, output_dir):
    """Export PyTorch model files (fallback when ONNX fails)."""
    print("Exporting PyTorch model files...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    
    model_params = {
        'n_coord_feats': config.get("n_coord_feats", 6),
        'n_keys': config.get("n_keys", 29),
        'vocab_size': config.get("vocab_size", 32),
        'max_word_len': config.get("max_word_len", 30)
    }
    
    model = MODEL_GETTERS_DICT[config["model_name"]](**model_params).eval()
    model.load_state_dict(state_dict)
    
    # Create wrapper modules
    encode_module = SimpleEncodeModule(model).eval()
    decode_module = SimpleDecodeModule(model).eval()
    
    # Save PyTorch models
    encoder_path = output_dir / "neural_swipe_encoder.pt"
    decoder_path = output_dir / "neural_swipe_decoder.pt"
    
    torch.jit.save(torch.jit.script(encode_module), encoder_path)
    torch.jit.save(torch.jit.script(decode_module), decoder_path)
    
    print(f"PyTorch models exported:")
    print(f"  Encoder: {encoder_path} ({encoder_path.stat().st_size / 1024 / 1024:.1f}MB)")
    print(f"  Decoder: {decoder_path} ({decoder_path.stat().st_size / 1024 / 1024:.1f}MB)")
    
    return encoder_path, decoder_path


def export_onnx_simple(checkpoint_path, config, output_dir):
    """Simplified ONNX export with reduced complexity."""
    print("Attempting simplified ONNX export...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = get_state_dict_from_checkpoint(checkpoint)
        
        model_params = {
            'n_coord_feats': config.get("n_coord_feats", 6),
            'n_keys': config.get("n_keys", 29),
            'vocab_size': config.get("vocab_size", 32),
            'max_word_len': config.get("max_word_len", 30)
        }
        
        model = MODEL_GETTERS_DICT[config["model_name"]](**model_params).eval()
        model.load_state_dict(state_dict)
        
        # Create sample data
        swipe_length = 10
        batch_size = 1
        
        sample_traj = torch.ones((swipe_length, batch_size, 6), dtype=torch.float32)
        sample_kb_weights = torch.zeros((swipe_length, batch_size, 29), dtype=torch.float32)
        
        # Set some weights
        for i in range(swipe_length):
            sample_kb_weights[i, 0, i % 29] = 0.8
            sample_kb_weights[i, 0, (i + 1) % 29] = 0.2
        
        decoder_in = torch.tensor([[1], [20]], dtype=torch.int64)
        
        # Try to export encoder (simplified version)
        try:
            encode_module = SimpleEncodeModule(model).eval()
            encoded = encode_module(sample_traj, sample_kb_weights)
            
            encoder_path = output_dir / "neural_swipe_encoder_simple.onnx"
            torch.onnx.export(
                encode_module,
                (sample_traj, sample_kb_weights),
                encoder_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=False,  # Disable to avoid issues
                input_names=['trajectory_features', 'keyboard_weights'],
                output_names=['encoded_sequence'],
                verbose=False
            )
            print(f"✅ Encoder ONNX: {encoder_path} ({encoder_path.stat().st_size / 1024 / 1024:.1f}MB)")
            
        except Exception as e:
            print(f"❌ Encoder ONNX failed: {e}")
            encoded = model.encode((sample_traj, sample_kb_weights), None)
        
        # Try to export decoder
        try:
            decode_module = SimpleDecodeModule(model).eval()
            
            decoder_path = output_dir / "neural_swipe_decoder_simple.onnx"
            torch.onnx.export(
                decode_module,
                (decoder_in, encoded),
                decoder_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=False,
                input_names=['decoder_input', 'encoded_sequence'],
                output_names=['character_logits'],
                verbose=False
            )
            print(f"✅ Decoder ONNX: {decoder_path} ({decoder_path.stat().st_size / 1024 / 1024:.1f}MB)")
            
        except Exception as e:
            print(f"❌ Decoder ONNX failed: {e}")
            
    except Exception as e:
        print(f"❌ ONNX export completely failed: {e}")
        print("Falling back to PyTorch export...")
        return export_pytorch_model(checkpoint_path, config, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Fixed model export")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--config", default="export_config_70_7.json", help="Config file")
    parser.add_argument("--output-dir", default="../results/kb_apk_models", help="Output directory")
    parser.add_argument("--format", choices=["onnx", "pytorch", "both"], default="both", help="Export format")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"Exporting model with format: {args.format}")
    
    if args.format in ["onnx", "both"]:
        try:
            export_onnx_simple(args.checkpoint, config, args.output_dir)
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    if args.format in ["pytorch", "both"]:
        try:
            export_pytorch_model(args.checkpoint, config, args.output_dir)
        except Exception as e:
            print(f"PyTorch export failed: {e}")
    
    print("\n✅ Export complete! Check the output directory for model files.")


if __name__ == "__main__":
    main()