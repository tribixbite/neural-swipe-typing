#!/usr/bin/env python3
"""
Training runner script using uv for Neural Swipe Typing project.
Provides easy interface to start training with optimal configurations.
"""
import subprocess
import sys
import os
import argparse
import json
import tempfile
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and optionally check for errors."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def get_optimal_batch_sizes(vram_gb, num_gpus=1):
    """Calculate optimal batch sizes based on available VRAM."""
    vram_per_gpu = vram_gb // num_gpus if num_gpus > 1 else vram_gb
    
    # Conservative batch sizes to avoid OOM with safety margin
    batch_configs = {
        12: {"train": 256, "val": 384, "workers": 4},   # RTX 3060, 4070
        16: {"train": 384, "val": 512, "workers": 6},   # RTX 4090M, 4080
        24: {"train": 512, "val": 768, "workers": 8},   # RTX 4090, 3090  
        48: {"train": 768, "val": 1024, "workers": 8},  # 2x RTX 3090
    }
    
    # Find closest VRAM config
    vram_key = min(batch_configs.keys(), key=lambda x: abs(x - vram_per_gpu))
    config = batch_configs[vram_key].copy()
    
    # Scale batch sizes for multi-GPU (total across all GPUs)
    if num_gpus > 1:
        config["train"] *= num_gpus
        config["val"] *= num_gpus
    
    return config

def create_vram_optimized_config(base_config_path, vram_gb, num_gpus=1):
    """Create a VRAM-optimized config file."""
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    batch_config = get_optimal_batch_sizes(vram_gb, num_gpus)
    
    # Update training config with VRAM-optimized settings
    config["training_config"]["batch_size_train"] = batch_config["train"]
    config["training_config"]["batch_size_val"] = batch_config["val"]
    config["training_config"]["num_workers"] = batch_config["workers"]
    
    # Adjust learning rate based on batch size (linear scaling)
    base_lr = 1e-4
    batch_scale = batch_config["train"] / 256  # 256 is baseline batch size
    config["training_config"]["learning_rate"] = base_lr * batch_scale
    
    # Update checkpoint and log directories to include VRAM info
    vram_suffix = f"_vram{vram_gb}gb"
    if num_gpus > 1:
        vram_suffix += f"_{num_gpus}gpu"
    
    config["training_config"]["checkpoint_dir"] = f"./checkpoints/english_minimal{vram_suffix}/"
    config["training_config"]["log_dir"] = f"./lightning_logs/english_minimal{vram_suffix}/"
    
    # Add comment about optimization
    config["training_config"]["dataset_stats"]["comment"] = (
        f"VRAM-optimized for {vram_gb}GB VRAM: "
        f"batch_train={batch_config['train']}, batch_val={batch_config['val']}, "
        f"workers={batch_config['workers']}, lr={config['training_config']['learning_rate']:.2e}"
    )
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Run Neural Swipe Typing training")
    parser.add_argument(
        "--config", 
        default="configs/config_english_minimal.json",
        help="Training configuration file (default: config_english_minimal.json)"
    )
    parser.add_argument(
        "--gpu-optimized",
        action="store_true", 
        help="Use GPU-optimized config (larger batches, more workers)"
    )
    parser.add_argument(
        "--vram",
        type=int,
        choices=[12, 16, 24, 48],
        help="GPU VRAM in GB (12, 16, 24, 48) - automatically sets optimal batch sizes"
    )
    parser.add_argument(
        "--gpus", 
        type=int, 
        default=1,
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--max-epochs", 
        type=int, 
        default=100,
        help="Maximum number of epochs (default: 100)"
    )
    parser.add_argument(
        "--checkpoint", 
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Print command without running"
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Handle VRAM-based optimization
    config_path = args.config
    temp_config_file = None
    
    if args.vram:
        # Create VRAM-optimized config
        base_config = "configs/config_english_minimal.json"
        optimized_config = create_vram_optimized_config(base_config, args.vram, args.gpus)
        
        # Write to temporary file
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(optimized_config, temp_config_file, indent=2)
        temp_config_file.close()
        config_path = temp_config_file.name
        
        batch_config = get_optimal_batch_sizes(args.vram, args.gpus)
        print("🚀 Starting Neural Swipe Typing training...")
        print(f"💾 VRAM: {args.vram}GB optimized")
        print(f"📊 Batch sizes: Train={batch_config['train']}, Val={batch_config['val']}")
        print(f"👷 Workers: {batch_config['workers']}")
        print(f"📈 Learning rate: {optimized_config['training_config']['learning_rate']:.2e}")
        
    elif args.gpu_optimized:
        config_path = "configs/config_english_minimal_gpu_optimized.json" 
        print("🚀 Starting Neural Swipe Typing training...")
        print("⚡ GPU-optimized mode: Large batches, 8 workers, 2x learning rate")
        
    else:
        print("🚀 Starting Neural Swipe Typing training...")
        print("📋 Standard configuration")
    
    print(f"📁 Configuration: {config_path}")
    print(f"🎯 GPUs: {args.gpus}")
    print(f"🔄 Max epochs: {args.max_epochs}")
    
    # Check if base config file exists (not temp file)
    if not args.vram:
        base_config_path = Path(config_path)
        if not base_config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            print("Available configs:")
            for config in Path("configs").glob("config_*.json"):
                print(f"  - {config}")
            sys.exit(1)
    
    # Build training command
    cmd = [
        "python", "src/train_english.py",
        "--config", str(config_path),
        "--gpus", str(args.gpus),
        "--max-epochs", str(args.max_epochs)
    ]
    
    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            print(f"❌ Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)
        cmd.extend(["--checkpoint", args.checkpoint])
    
    if args.dry_run:
        print(f"🔍 Would run: {' '.join(cmd)}")
        return
    
    # Pre-flight checks
    print("\n🔧 Pre-flight checks...")
    
    # Check GPU availability if requested
    if args.gpus > 0:
        try:
            result = subprocess.run([
                "python", "-c", 
                "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
            ], check=True, capture_output=True, text=True)
            print(f"✅ GPU status: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"❌ GPU check failed: {e}")
            print("Continuing anyway...")
    
    # Check if datasets exist
    data_files = [
        "data/data_preprocessed/english_filtered_train.jsonl",
        "data/data_preprocessed/english_filtered_valid.jsonl", 
        "data/data_preprocessed/voc_english_minimal.txt"
    ]
    
    missing_files = [f for f in data_files if not Path(f).exists()]
    if missing_files:
        print("⚠️  Missing data files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Run data preparation first or check paths in config file.")
    
    print(f"\n🎯 Starting training with {args.config}...")
    print("📊 Monitor progress:")
    print("  - TensorBoard: tensorboard --logdir lightning_logs/")
    print("  - Checkpoints: checkpoints/english/")
    print("  - GPU usage: nvidia-smi -l 1")
    print("\n" + "="*50)
    
    # Run training
    try:
        run_command(cmd)
        print("\n🎉 Training completed successfully!")
        
        # Show available checkpoints
        checkpoint_dir = Path("checkpoints/english")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                print("\n📁 Available checkpoints:")
                for ckpt in sorted(checkpoints):
                    print(f"  - {ckpt}")
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        if temp_config_file:
            os.unlink(temp_config_file.name)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("Note: Lightning saves checkpoints automatically")
        if temp_config_file:
            os.unlink(temp_config_file.name)
    finally:
        # Clean up temporary config file
        if temp_config_file and os.path.exists(temp_config_file.name):
            os.unlink(temp_config_file.name)

if __name__ == "__main__":
    main()