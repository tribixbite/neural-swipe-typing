#!/usr/bin/env python3
"""
Working ONNX export that properly handles the model's tuple input format.
Exports encoder and decoder separately as recommended by Gemini analysis.
"""

import torch
import torch.nn as nn
import json
import sys
import os
import math
from pathlib import Path
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import MODEL_GETTERS_DICT


class EncoderWrapper(nn.Module):
    """Wrapper for exporting just the encoder part of the model."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.d_model = 128  # From model architecture
        
    def forward(self, src_combined):
        """
        Args:
            src_combined: [batch_size, seq_len, 7] where last dim is:
                         [x, y, vx, vy, ax, ay, kb_id]
        Returns:
            memory: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = src_combined.shape
        
        # Split combined input
        traj_feats = src_combined[..., :6]  # First 6 features
        kb_ids = src_combined[..., 6].long().clamp(0, 29)  # Last feature as keyboard ID
        
        # Transpose to model's expected format [seq_len, batch_size, ...]
        traj_feats = traj_feats.transpose(0, 1)
        kb_ids = kb_ids.transpose(0, 1)
        
        # Create encoder input tuple
        encoder_input = (traj_feats, kb_ids)
        
        # Encode without padding mask (pass None)
        with torch.no_grad():
            x_embedded = self.model.enc_in_emb_model(encoder_input)
            memory = self.model.encoder(x_embedded, src_key_padding_mask=None)
            
        # Transpose back to batch-first
        return memory.transpose(0, 1)


class DecoderWrapper(nn.Module):
    """Wrapper for exporting the decoder part of the model."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = model.device
        
    def forward(self, tgt_ids, memory):
        """
        Args:
            tgt_ids: [batch_size, tgt_len] - token IDs generated so far
            memory: [batch_size, src_len, d_model] - encoder output
        Returns:
            logits: [batch_size, tgt_len, vocab_size]
        """
        batch_size, tgt_len = tgt_ids.shape
        
        # Transpose to model format
        tgt_ids = tgt_ids.transpose(0, 1)  # [tgt_len, batch_size]
        memory = memory.transpose(0, 1)    # [src_len, batch_size, d_model]
        
        with torch.no_grad():
            # Embed target tokens
            tgt_embedded = self.model.dec_in_emb_model(tgt_ids)
            
            # Create causal mask for autoregressive decoding
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(self.device)
            
            # Decode
            decoded = self.model.decoder(
                tgt_embedded, memory, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=None,
                tgt_key_padding_mask=None
            )
            
            # Project to vocabulary
            logits = self.model.out(decoded)
            
        # Transpose back to batch-first
        return logits.transpose(0, 1)


def load_model(checkpoint_path, model_name="v3_nearest_and_traj_transformer_bigger"):
    """Load model from checkpoint with proper state dict handling."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model
    model = MODEL_GETTERS_DICT[model_name]()
    
    # Fix state dict keys
    state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.', '') if key.startswith('model.') else key
        new_key = new_key.replace('enc_in_emb_key_emb.', 'enc_in_emb_model.key_emb.')
        new_key = new_key.replace('dec_in_emb_0.', 'dec_in_emb_model.0.')
        new_key = new_key.replace('dec_in_emb_2.', 'dec_in_emb_model.2.')
        state_dict[new_key] = value
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model


def export_encoder(model, output_dir):
    """Export the encoder to ONNX."""
    
    encoder_wrapper = EncoderWrapper(model)
    encoder_wrapper.eval()
    
    # Create dummy input [batch_size, seq_len, 7]
    batch_size = 1
    seq_len = 100
    dummy_input = torch.randn(batch_size, seq_len, 7)
    
    onnx_path = output_dir / "encoder.onnx"
    
    print(f"Exporting encoder to {onnx_path}")
    torch.onnx.export(
        encoder_wrapper,
        dummy_input,
        str(onnx_path),
        input_names=['src_combined'],
        output_names=['memory'],
        dynamic_axes={
            'src_combined': {0: 'batch_size', 1: 'sequence_length'},
            'memory': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✅ Encoder exported successfully")
    return onnx_path


def export_decoder(model, output_dir):
    """Export the decoder to ONNX."""
    
    decoder_wrapper = DecoderWrapper(model)
    decoder_wrapper.eval()
    
    # Create dummy inputs
    batch_size = 1
    tgt_len = 10
    src_len = 100
    d_model = 128
    
    dummy_tgt = torch.randint(0, 30, (batch_size, tgt_len))
    dummy_memory = torch.randn(batch_size, src_len, d_model)
    
    onnx_path = output_dir / "decoder.onnx"
    
    print(f"Exporting decoder to {onnx_path}")
    torch.onnx.export(
        decoder_wrapper,
        (dummy_tgt, dummy_memory),
        str(onnx_path),
        input_names=['tgt_ids', 'memory'],
        output_names=['logits'],
        dynamic_axes={
            'tgt_ids': {0: 'batch_size', 1: 'tgt_length'},
            'memory': {0: 'batch_size', 1: 'src_length'},
            'logits': {0: 'batch_size', 1: 'tgt_length'}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✅ Decoder exported successfully")
    return onnx_path


def create_config(output_dir):
    """Create configuration file for the exported models."""
    
    config = {
        "model_type": "neural-swipe-encoder-decoder",
        "encoder": {
            "input_dim": 7,
            "hidden_size": 128,
            "num_layers": 4,
            "num_heads": 4
        },
        "decoder": {
            "vocab_size": 30,
            "hidden_size": 128,
            "num_layers": 4,
            "num_heads": 4
        },
        "tokenizer": {
            "vocab_size": 30,
            "pad_token_id": 28,
            "sos_token_id": 29,
            "eos_token_id": 26,
            "unk_token_id": 27
        },
        "features": {
            "trajectory": ["x", "y", "vx", "vy", "ax", "ay"],
            "keyboard": "nearest_key_id"
        },
        "max_seq_len": 299,
        "max_word_len": 35
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Config saved to {config_path}")
    return config_path


def validate_exports(encoder_path, decoder_path):
    """Validate that ONNX models can be loaded."""
    
    try:
        import onnxruntime as ort
        
        print("Validating ONNX exports...")
        
        # Test encoder
        encoder_session = ort.InferenceSession(str(encoder_path))
        print(f"  ✅ Encoder loaded successfully")
        print(f"     Inputs: {[i.name for i in encoder_session.get_inputs()]}")
        print(f"     Outputs: {[o.name for o in encoder_session.get_outputs()]}")
        
        # Test decoder
        decoder_session = ort.InferenceSession(str(decoder_path))
        print(f"  ✅ Decoder loaded successfully")
        print(f"     Inputs: {[i.name for i in decoder_session.get_inputs()]}")
        print(f"     Outputs: {[o.name for o in decoder_session.get_outputs()]}")
        
        return True
        
    except ImportError:
        print("⚠️  onnxruntime not installed, skipping validation")
        print("   Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def main():
    # Setup paths
    checkpoint_path = "../checkpoints_english/english-epoch=51-val_loss=1.248-val_word_acc=0.659.ckpt"
    output_dir = Path("./onnx_working/")
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting ONNX export (encoder-decoder separation)")
    print("=" * 50)
    
    try:
        # Load model
        model = load_model(checkpoint_path)
        print(f"✅ Model loaded successfully")
        
        # Export encoder
        encoder_path = export_encoder(model, output_dir)
        
        # Export decoder
        decoder_path = export_decoder(model, output_dir)
        
        # Create config
        config_path = create_config(output_dir)
        
        # Validate exports
        print("=" * 50)
        validate_exports(encoder_path, decoder_path)
        
        print("=" * 50)
        print("🎉 Export completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"   - Encoder: {encoder_path}")
        print(f"   - Decoder: {decoder_path}")
        print(f"   - Config: {config_path}")
        
        print("\n📝 Next steps for web implementation:")
        print("   1. Load both ONNX models in JavaScript")
        print("   2. Preprocess swipe to 7D features")
        print("   3. Run encoder once to get memory")
        print("   4. Run decoder autoregressively to generate word")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()