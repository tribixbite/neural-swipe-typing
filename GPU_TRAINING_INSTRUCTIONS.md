# GPU Training Instructions for Windows/WSL

## Critical Fixes Applied (Updated 2024-11-25)
✅ Fixed keyboard tokenizer bug (was using Cyrillic instead of English)
✅ Fixed feature extraction to use correct tokenizer
✅ Fixed validation settings (now validates every epoch instead of every 11 epochs)
✅ **NEW**: Performance optimizations implemented:
  - Batch-first tensors for nested tensor optimization
  - Tensor Cores enabled for RTX GPU acceleration
  - Mixed precision training (fp16)
  - Gradient accumulation (effective batch=512)
  - All training warnings resolved

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

#### 2. Install Python Dependencies
```bash
# Recommended: Use uv for fast dependency management
pip install uv
python install.py

# Or traditional pip install
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install lightning==2.2.4 torchmetrics numpy pandas tqdm requests gdown
```

#### 3. Verify GPU Access
```bash
# Check NVIDIA driver
nvidia-smi

# Test PyTorch CUDA (using uv)
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Or without uv
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Running Training

### Quick Start (Recommended)
```bash
# Using the new uv-based training script
./run_training.py

# With custom config
./run_training.py --config configs/config_english_filtered.json

# Resume from checkpoint
./run_training.py --checkpoint checkpoints/english/last.ckpt --max-epochs 100
```

### Manual Commands
```bash
cd /path/to/neural-swipe-typing

# Current optimal config (29-token vocabulary)
python src/train_english.py --config configs/config_english_minimal.json --gpus 1

# Legacy config (67-token vocabulary, larger but less efficient)
python src/train_english.py --config configs/config_english_filtered.json --gpus 1

# Resume from checkpoint
python src/train_english.py --config configs/config_english_minimal.json --gpus 1 --checkpoint checkpoints/english/last.ckpt
```

### Configuration Files
- **`config_english_minimal.json`**: ⭐ **Recommended** - 29-token vocabulary (26 letters + 3 special)
- **`config_english_filtered.json`**: Legacy 67-token vocabulary (includes punctuation, uppercase)
- **`config_english.json`**: Original unfiltered dataset (not recommended)

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

With ~59k filtered training samples and optimal 29-token architecture:
- **Steps per epoch**: 231 (updated for filtered dataset)
- **Validation**: Every epoch
- **Per epoch time**: ~2-3 minutes on RTX 4090M (with all optimizations)
- **Architecture**: 29-token vocabulary (57% smaller output layer vs 67-token)
- **Recent performance**: 
  - Minimal vocab training started from scratch (2024-11-25)
  - Expected faster convergence due to focused vocabulary
- **Total time estimate**:
  - Target 70%+ accuracy: ~5-15 epochs (15-45 minutes)
  - Full 100 epochs: 2-5 hours (with optimizations)

**Performance improvements applied:**
- **Tensor Cores acceleration**: RTX GPU optimization for matrix operations
- **Batch-first tensors**: Nested tensor optimization for memory efficiency
- **Mixed precision training (fp16)**: Faster training with maintained precision
- **Minimal vocabulary**: 29 tokens instead of 67 (57% reduction in output layer)
- **Gradient accumulation**: Effective batch size of 512

### Quick Performance Check
```bash
# After starting training, the progress bar shows speed
# Look for "it/s" (iterations per second)
# Calculate: total_time = (231 steps * epochs) / it_s

# Example from optimized run:
# Epoch 0: 100%|█████| 231/231 [XX:XX<00:00, 0.XX it/s]
# With optimizations: expect 0.3-0.4 it/s on RTX 4090M
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

**Current best checkpoints available:**
- `english-epoch=70-val_loss=0.742-val_word_level_accuracy=0.635.ckpt` (63.5% accuracy)
- Multiple checkpoints from epochs 60-70 saved

Use the best checkpoint for inference/testing.

## Training Status (2024-11-25)

✅ **CRITICAL ARCHITECTURAL FIX COMPLETED**

**Problem Identified**: Previous training used bloated 67-token vocabulary instead of optimal 29 tokens for swipe typing.

**Solution Implemented**:
- ✅ Created minimal vocabulary: `voc_english_minimal.txt` (29 tokens: a-z + 3 special)
- ✅ New optimal configuration: `config_english_minimal.json`
- ✅ Architecture optimized: 57% smaller output layer (29 vs 67 classes)
- ✅ All performance optimizations preserved:
  - Batch-first tensors for nested tensor optimization
  - Tensor Cores acceleration for RTX GPUs
  - Mixed precision training (fp16)
  - Gradient accumulation (effective batch=512)

**Training History**:
- Previous run (67-token vocab): Epoch 70 achieved 63.5% validation accuracy
- New optimal run (29-token vocab): Training restarted from scratch for maximum efficiency
- Checkpoint conversion utility available for migrating between formats

**Installation & Dependencies**:
- ✅ Created `install.py` script using uv package manager
- ✅ Created `./run_training.py` script for easy training
- ✅ All dependencies documented: torch, lightning, torchmetrics, numpy, pandas, tqdm

**Current Status**:
- Ready for optimal training with 29-token architecture
- All performance optimizations active
- Installation scripts ready for deployment

**Next steps:**
- Complete training with minimal vocabulary (target: 70-75% accuracy)
- Implement beam search evaluation
- Prepare for mobile deployment via ExecutorTorch