#!/usr/bin/env python3
"""
Convert checkpoint from seq_first to batch_first format.
This fixes the PositionalEncoding tensor shape mismatch after batch_first optimization.
"""

import torch
import argparse
from pathlib import Path


def convert_checkpoint(input_path: str, output_path: str):
    """Convert checkpoint from seq_first to batch_first format."""
    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Parameters that need shape conversion
    conversions = []
    
    for key, tensor in checkpoint['state_dict'].items():
        if 'pos_encoder.pe' in key or key.endswith('.pe'):
            # Convert from [seq_len, 1, d_model] to [1, seq_len, d_model]
            if len(tensor.shape) == 3 and tensor.shape[1] == 1:
                old_shape = tensor.shape
                new_tensor = tensor.transpose(0, 1)  # [1, seq_len, d_model]
                checkpoint['state_dict'][key] = new_tensor
                conversions.append(f"{key}: {old_shape} -> {new_tensor.shape}")
    
    print(f"Converted {len(conversions)} PositionalEncoding tensors:")
    for conv in conversions:
        print(f"  {conv}")
    
    # Save converted checkpoint
    print(f"Saving converted checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("✓ Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint to batch_first format")
    parser.add_argument("input", help="Input checkpoint path")
    parser.add_argument("output", help="Output checkpoint path")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input checkpoint {args.input} does not exist")
        return 1
    
    convert_checkpoint(args.input, args.output)
    return 0


if __name__ == "__main__":
    exit(main())