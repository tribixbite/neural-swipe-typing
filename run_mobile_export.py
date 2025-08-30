#!/usr/bin/env python3
"""
Load trained mobile model and export to ONNX and ExecuTorch formats
"""

import torch
import pytorch_lightning as pl
from mobile_optimized_model import create_mobile_model
from export_mobile_models import MobileModelExporter
from train_mobile_model import MobileSwipeTrainer
import json
import os

def main():
    print("=== Mobile Model Export Pipeline ===")
    
    # Load best checkpoint
    checkpoint_path = "checkpoints/mobile_model/mobile-swipe-epoch=06-val_acc=1.000-v1.ckpt"
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the Lightning model from checkpoint
    lightning_model = MobileSwipeTrainer.load_from_checkpoint(checkpoint_path, map_location='cpu')
    lightning_model.eval()
    
    # Extract the PyTorch model and move to CPU for export
    pytorch_model = lightning_model.model
    pytorch_model.cpu()
    pytorch_model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in pytorch_model.parameters()):,} parameters")
    
    # Load configuration
    config_path = 'configs/config_mobile_optimized.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create exporter
    exporter = MobileModelExporter(pytorch_model, config)
    
    # Create deployment package
    output_dir = 'mobile_deployment_package'
    print(f"Creating deployment package in: {output_dir}")
    
    package_files = exporter.create_deployment_package(output_dir)
    
    print("\n=== Export Summary ===")
    for key, path in package_files.items():
        if os.path.exists(path):
            file_size = os.path.getsize(path) / 1024  # KB
            print(f"{key}: {path} ({file_size:.1f} KB)")
        else:
            print(f"{key}: {path} (script created)")
    
    print("\n=== Mobile Model Export Completed ===")
    print("Available formats:")
    print("- ONNX for web deployment: mobile_deployment_package/swipe_model_web.onnx")
    print("- ExecuTorch for Android: mobile_deployment_package/swipe_model_android.pte (or export script)")
    print("- Integration examples: mobile_deployment_package/integration_examples/")
    
    return package_files

if __name__ == "__main__":
    main()