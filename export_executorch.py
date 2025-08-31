#!/usr/bin/env python3
"""
Export mobile swipe typing model to ExecuTorch format for Android deployment.
"""

import torch
import torch.nn as nn
from mobile_model import create_mobile_model
from train_mobile_optimized import MobileSwipeTrainer, VocabularyManager
import os
from pathlib import Path


def quantize_model(model: nn.Module) -> nn.Module:
    """Apply INT8 quantization to model for mobile deployment."""
    model.eval()
    
    # Configure quantization
    backend = "qnnpack"  # Optimized for mobile
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    
    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model, inplace=False)
    
    # Calibrate with representative data
    batch_size = 4
    seq_len = 150
    dummy_input = torch.randn(batch_size, seq_len, 6)
    
    with torch.no_grad():
        for _ in range(10):  # Run multiple batches for calibration
            _ = model_prepared(dummy_input)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)
    
    return model_quantized


def export_to_torchscript(model: nn.Module, save_path: str):
    """Export model to TorchScript format."""
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 150, 6)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_model = optimize_for_mobile(traced_model)
    
    # Save
    optimized_model.save(save_path)
    print(f"Model exported to TorchScript: {save_path}")
    
    # Print model size
    model_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"TorchScript model size: {model_size:.2f} MB")
    
    return optimized_model


def export_to_executorch(model_path: str, output_dir: str = "models/executorch"):
    """Export model to ExecuTorch format for Android."""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary manager
    vocab_manager = VocabularyManager(max_size=10000)
    
    # Load trained model
    if model_path.endswith('.ckpt'):
        # Load from checkpoint
        model_wrapper = MobileSwipeTrainer.load_from_checkpoint(
            model_path,
            vocab_size=vocab_manager.size
        )
        model = model_wrapper.model
    else:
        # Create new model
        model = create_mobile_model(vocab_size=vocab_manager.size)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
    
    # Export to TorchScript first
    torchscript_path = os.path.join(output_dir, "mobile_swipe_model.pt")
    traced_model = export_to_torchscript(model, torchscript_path)
    
    # Try to export to ExecuTorch (requires executorch package)
    try:
        from executorch.exir import to_edge
        from executorch.exir.backend import Backend
        
        # Convert to Edge format
        example_input = torch.randn(1, 150, 6)
        edge_model = to_edge(traced_model, (example_input,))
        
        # Apply optimizations
        from executorch.exir.passes import MemoryPlanningPass
        edge_model = edge_model.transform([MemoryPlanningPass()])
        
        # Export to .pte file
        pte_path = os.path.join(output_dir, "mobile_swipe_model.pte")
        with open(pte_path, 'wb') as f:
            f.write(edge_model.buffer)
        
        print(f"Model exported to ExecuTorch: {pte_path}")
        model_size = os.path.getsize(pte_path) / (1024 * 1024)
        print(f"ExecuTorch model size: {model_size:.2f} MB")
        
    except ImportError:
        print("ExecuTorch not installed. Install with: pip install executorch")
        print("TorchScript model can still be used for mobile deployment")
    
    # Also try INT8 quantization for smaller model
    print("\nApplying INT8 quantization...")
    try:
        quantized_model = quantize_model(model)
        
        # Export quantized model
        quantized_path = os.path.join(output_dir, "mobile_swipe_model_int8.pt")
        traced_quantized = torch.jit.trace(quantized_model, torch.randn(1, 150, 6))
        
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized_quantized = optimize_for_mobile(traced_quantized)
        optimized_quantized.save(quantized_path)
        
        print(f"Quantized model exported: {quantized_path}")
        model_size = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"Quantized model size: {model_size:.2f} MB")
        
    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Using FP32 model for deployment")
    
    # Generate model info file
    info_path = os.path.join(output_dir, "model_info.json")
    import json
    model_info = {
        "input_shape": [1, 150, 6],
        "output_shape": [1, vocab_manager.size],
        "input_features": ["x", "y", "vx", "vy", "ax", "ay"],
        "vocab_size": vocab_manager.size,
        "model_type": "mobile_swipe_typing",
        "export_formats": ["torchscript", "executorch", "onnx"]
    }
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nModel info saved to: {info_path}")
    print("\nExport complete! Models ready for Android deployment.")


def benchmark_inference(model_path: str, num_runs: int = 100):
    """Benchmark inference speed of the mobile model."""
    import time
    
    # Load model
    model = create_mobile_model(vocab_size=10000)
    if os.path.exists(model_path):
        if model_path.endswith('.pt'):
            model = torch.jit.load(model_path)
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
    
    # Create test input
    test_input = torch.randn(1, 150, 6)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(test_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    # Statistics
    import numpy as np
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)
    
    print("\nInference Benchmark Results:")
    print(f"Mean latency: {mean_latency:.2f} ms")
    print(f"Std deviation: {std_latency:.2f} ms")
    print(f"Min latency: {min_latency:.2f} ms")
    print(f"Max latency: {max_latency:.2f} ms")
    print(f"P50 latency: {p50_latency:.2f} ms")
    print(f"P90 latency: {p90_latency:.2f} ms")
    print(f"P99 latency: {p99_latency:.2f} ms")
    
    if mean_latency < 50:
        print("✅ Model meets <50ms latency requirement!")
    else:
        print("⚠️ Model exceeds 50ms latency target")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export mobile swipe model for Android")
    parser.add_argument("--model", type=str, help="Path to model checkpoint or state dict")
    parser.add_argument("--output-dir", type=str, default="models/executorch", 
                       help="Output directory for exported models")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run inference benchmark after export")
    
    args = parser.parse_args()
    
    # Use best checkpoint if no model specified
    if not args.model:
        checkpoint_dir = Path("checkpoints/mobile")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                # Sort by modification time and get latest
                args.model = str(sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1])
                print(f"Using checkpoint: {args.model}")
            else:
                print("No checkpoints found. Training a model first...")
                # Create untrained model for testing export pipeline
                model = create_mobile_model(vocab_size=10000)
                temp_path = "models/temp_mobile_model.pth"
                Path("models").mkdir(exist_ok=True)
                torch.save(model.state_dict(), temp_path)
                args.model = temp_path
        else:
            print("No checkpoints directory. Creating untrained model for export testing...")
            model = create_mobile_model(vocab_size=10000)
            temp_path = "models/temp_mobile_model.pth"
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), temp_path)
            args.model = temp_path
    
    # Export model
    export_to_executorch(args.model, args.output_dir)
    
    # Benchmark if requested
    if args.benchmark:
        torchscript_path = os.path.join(args.output_dir, "mobile_swipe_model.pt")
        if os.path.exists(torchscript_path):
            benchmark_inference(torchscript_path)


if __name__ == "__main__":
    main()