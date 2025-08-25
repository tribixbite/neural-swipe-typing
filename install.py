#!/usr/bin/env python3
"""
Installation script using uv for Neural Swipe Typing project.
Installs all dependencies and sets up the development environment.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and optionally check for errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode == 0

def main():
    print("🚀 Setting up Neural Swipe Typing environment with uv...")
    
    # Check if uv is installed
    if not run_command(["uv", "--version"], check=False):
        print("❌ uv is not installed. Please install it first:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # Ensure we're in the project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("\n📦 Creating virtual environment and installing dependencies...")
    
    # Create venv and install dependencies
    run_command(["uv", "venv"])
    
    # Install core dependencies
    dependencies = [
        "torch==2.1.2",
        "torchvision", 
        "torchaudio",
        "lightning==2.2.4",
        "torchmetrics",
        "numpy==1.24.3",
        "pandas==2.1.3", 
        "tqdm==4.66.3",
        "requests==2.31.0",
        "gdown==5.2.0",
        "matplotlib",
        "scikit-learn",
        "jupyterlab",
        "ipykernel"
    ]
    
    for dep in dependencies:
        if not run_command(["uv", "add", dep], check=False):
            print(f"⚠️  Warning: Failed to install {dep}")
    
    print("\n🎯 Setting up project structure...")
    
    # Create necessary directories
    dirs_to_create = [
        "data/data_preprocessed",
        "results/predictions", 
        "checkpoints/english",
        "lightning_logs/english"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    print("\n🔧 Verifying installation...")
    
    # Test imports
    test_imports = [
        "torch",
        "lightning", 
        "torchmetrics",
        "numpy",
        "pandas",
        "tqdm"
    ]
    
    for module in test_imports:
        try:
            result = subprocess.run(
                ["uv", "run", "python", "-c", f"import {module}; print(f'{module}: OK')"],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✅ {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to import {module}")
    
    print("\n🎉 Installation complete!")
    print("\n📋 Next steps:")
    print("1. Run training: ./run_training.py")
    print("2. Or activate environment: source .venv/bin/activate") 
    print('3. Check GPU: uv run python -c "import torch; print(f\'CUDA: {torch.cuda.is_available()}\')"')

if __name__ == "__main__":
    main()