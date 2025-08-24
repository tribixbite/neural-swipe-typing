# GPU Training Instructions for Windows/WSL

## Critical Fixes Applied
✅ Fixed keyboard tokenizer bug (was using Cyrillic instead of English)
✅ Fixed feature extraction to use correct tokenizer
✅ Fixed validation settings (now validates every epoch instead of every 11 epochs)

## Setup for Windows with NVIDIA GPU

### Recommended: Use WSL2
WSL2 provides better compatibility and stability for ML workloads.

#### 1. Install Prerequisites
```bash
# In Windows:
# 1. Install latest NVIDIA Game Ready or Studio Driver
# 2. Install WSL2: wsl --install

# In WSL2 Ubuntu:
# Install CUDA Toolkit for WSL2 (check NVIDIA docs for latest)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Verify GPU Access
```bash
# Check NVIDIA driver
nvidia-smi

# Test PyTorch CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Running Training

### Basic Command
```bash
cd /path/to/neural-swipe-typing
python src/train_english.py --config configs/config_english.json --gpus 1
```

### With Custom Settings
```bash
# Adjust batch size if you get OOM errors or want to use more VRAM
# Edit configs/config_english.json:
# - "batch_size_train": 256  (reduce if OOM, increase if lots of free VRAM)
# - "batch_size_val": 512

# Run with more epochs if needed
python src/train_english.py --config configs/config_english.json --gpus 1 --max-epochs 100

# Resume from checkpoint
python src/train_english.py --config configs/config_english.json --gpus 1 --checkpoint checkpoints/english/last.ckpt
```

## Monitoring Training

### GPU Usage
```bash
# In separate terminal, monitor GPU usage
nvidia-smi -l 1  # Updates every second

# Look for:
# - Memory-Usage: Should be 70-90% for optimal batch size
# - GPU-Util: Should be >90% during training
```

### Training Progress
- TensorBoard logs will be saved to `./lightning_logs/english/`
- Checkpoints saved to `./checkpoints/english/`
- Best model based on validation loss will be saved

### When Will Training Stop?

Training stops when ONE of these conditions is met:
1. **Max epochs reached**: Default 100 epochs
2. **Early stopping triggered**: If validation loss doesn't improve for 15 epochs (updated from 35)
3. **Manual interruption**: Ctrl+C (checkpoint will be saved)

### Expected Training Time

With ~70k training samples and batch size 256:
- **Steps per epoch**: 273
- **Validation**: Every epoch (updated from every 11 epochs)
- **Per epoch time**: Depends on GPU, typically 2-10 minutes on modern GPUs
- **Total time**: 
  - Best case (early stopping at ~30 epochs): 1-5 hours
  - Worst case (full 100 epochs): 3-17 hours

### Quick Performance Check
```bash
# After starting training, the progress bar shows speed
# Look for "it/s" (iterations per second)
# Calculate: total_time = (273 steps * epochs) / it_s
```

## Optimizing Batch Size

1. Start with default `batch_size_train: 256`
2. If OOM error: Halve to 128, then 64 if needed
3. If lots of free VRAM (check nvidia-smi): Try 384, 512, or higher
4. Larger batch = faster training + better convergence (usually)

## Memory Requirements
- **VRAM**: 4-8GB minimum, 8-16GB recommended
- **System RAM**: 16GB minimum (with num_workers=4)
- Reduce `num_workers` in config if RAM issues

## After Training

Best checkpoint will be in `./checkpoints/english/` with name like:
`english-epoch=XX-val_loss=X.XXX-val_word_level_accuracy=X.XXX.ckpt`

Use this for inference/testing.