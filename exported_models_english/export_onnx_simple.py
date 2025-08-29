#!/usr/bin/env python3
"""
Simplified ONNX export that directly wraps the model's encode functionality.
"""

import torch
import torch.nn as nn
import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import MODEL_GETTERS_DICT


class EncoderOnlyWrapper(nn.Module):
    """Simple wrapper that only exports the encoder part."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, trajectory_features):
        """
        Args:
            trajectory_features: [batch_size, seq_len, 7] where last dim is kb_id
        Returns:
            encoded: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = trajectory_features.shape
        
        # Split features
        traj_feats = trajectory_features[..., :6].transpose(0, 1)
        # Clamp keyboard IDs to valid range (0-29)
        kb_ids = trajectory_features[..., 6].clamp(0, 29).long().transpose(0, 1)
        
        # Create encoder input tuple
        encoder_input = (traj_feats, kb_ids)
        
        # Encode only (no decoder)
        with torch.no_grad():
            encoded = self.model.encode(encoder_input, None)
            
        # Transpose back to batch first
        return encoded.transpose(0, 1)


def main():
    # Load checkpoint
    checkpoint_path = "../checkpoints_english/english-epoch=51-val_loss=1.248-val_word_acc=0.659.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model
    model_name = "v3_nearest_and_traj_transformer_bigger"
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
    
    # Create wrapper
    wrapper = EncoderOnlyWrapper(model)
    
    # Export
    output_dir = Path("./onnx_encoder_only/")
    output_dir.mkdir(exist_ok=True)
    
    dummy_input = torch.randn(1, 100, 7)  # batch=1, seq=100, features=7
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_dir / "encoder.onnx"),
        input_names=['trajectory_features'],
        output_names=['encoded'],
        dynamic_axes={
            'trajectory_features': {0: 'batch_size', 1: 'sequence_length'},
            'encoded': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print(f"✅ Exported encoder to {output_dir / 'encoder.onnx'}")
    
    # Save config
    config = {
        "model_type": "neural-swipe-encoder",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "feature_dim": 7,
        "max_seq_len": 299
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Saved config to {output_dir / 'config.json'}")


if __name__ == "__main__":
    main()