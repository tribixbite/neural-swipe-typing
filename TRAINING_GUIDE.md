# High-Performance Training Guide

## Quick Start Commands

### **Single GPU (RTX 4090M - 16GB VRAM)**
```bash
# Start fresh training (VRAM-optimized for 16GB)
./run_training.py --vram 16

# Resume from checkpoint
./run_training.py --vram 16 --checkpoint checkpoints/english_minimal_vram16gb/last.ckpt

# Monitor training
watch -n 2 nvidia-smi
tensorboard --logdir lightning_logs/
```

### **Single GPU (RTX 4090 - 24GB VRAM)**
```bash
# Start fresh training (VRAM-optimized for 24GB)
./run_training.py --vram 24

# Resume from checkpoint  
./run_training.py --vram 24 --checkpoint checkpoints/english_minimal_vram24gb/last.ckpt
```

### **Dual GPU (2x RTX 3090 - 48GB total VRAM)**
```bash
# Multi-GPU training with DDP (48GB total)
./run_training.py --vram 48 --gpus 2

# Resume multi-GPU training
./run_training.py --vram 48 --gpus 2 --checkpoint checkpoints/english_minimal_vram48gb_2gpu/last.ckpt
```

## Optimization Settings

### **Automatic VRAM Optimization**

The `--vram` flag automatically configures optimal settings:

| VRAM | GPU Examples | Batch Train/Val | Workers | Learning Rate | Expected Speedup |
|------|-------------|-----------------|---------|---------------|------------------|
| 12GB | RTX 3060, 4070 | 256/384 | 4 | 1.0e-4 | 2x |
| 16GB | **RTX 4090M**, 4080 | 384/512 | 6 | 1.5e-4 | 2.5x |  
| 24GB | RTX 4090, 3090 | 512/768 | 8 | 2.0e-4 | 3x |
| 48GB | 2x RTX 3090 | 768/1024 | 8 | 3.0e-4 | 4x |

### **Hardware Detection**
```bash
# Check your VRAM
nvidia-smi | grep MiB

# Examples:
# RTX 4090M: 16GB VRAM → use --vram 16
# RTX 4090:  24GB VRAM → use --vram 24  
# RTX 3060:  12GB VRAM → use --vram 12
# 2x RTX 3090: 48GB total → use --vram 48 --gpus 2
```

## Training Lifecycle Management

### **1. Start New Training**
```bash
# Fresh start for your RTX 4090M (16GB VRAM)
./run_training.py --vram 16

# Files created:
# - checkpoints/english_minimal_vram16gb/
# - lightning_logs/english_minimal_vram16gb/
```

### **2. Resume Training**
```bash
# Resume from last checkpoint (RTX 4090M)
./run_training.py --vram 16 --checkpoint checkpoints/english_minimal_vram16gb/last.ckpt

# Resume from specific epoch
./run_training.py --vram 16 --checkpoint checkpoints/english_minimal_vram16gb/epoch=25.ckpt

# Resume with different max epochs
./run_training.py --vram 16 --checkpoint checkpoints/english_minimal_vram16gb/last.ckpt --max-epochs 200
```

### **3. Find Best Checkpoints**
```bash
# List all checkpoints sorted by validation accuracy
ls -la checkpoints/english_minimal_vram16gb/ | grep "val_word_level_accuracy" | sort -k1 -r

# Example output:
# english-epoch=45-val_loss=0.523-val_word_level_accuracy=0.782.ckpt
# english-epoch=42-val_loss=0.547-val_word_level_accuracy=0.771.ckpt
```

## Performance Monitoring

### **Real-Time Monitoring**
```bash
# Terminal 1: GPU usage
watch -n 1 'nvidia-smi | grep -E "(MiB|%)"'

# Terminal 2: Training progress  
./run_training.py --gpu-optimized

# Terminal 3: TensorBoard
tensorboard --logdir lightning_logs/ --port 6006
# Open: http://localhost:6006
```

### **Expected Performance**
| Hardware | VRAM | Batch Size | Time/Epoch | Total Time (100 epochs) |
|----------|------|------------|------------|-------------------------|
| RTX 4090M | 16GB | 384/512 | 60-90s | **2-2.5 hours** |
| RTX 4090 | 24GB | 512/768 | 45-60s | **1.5-2 hours** |
| 2x RTX 3090 | 48GB | 768/1024 | 25-35s | **45-60 minutes** |
| RTX 4090M | - | 256/512 (default) | 2-3min | 3-5 hours |

## Multi-GPU Training (2x RTX 3090)

### **Setup**
```bash
# Enable multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1

# Lightning automatically uses DistributedDataParallel (DDP)
./run_training.py --gpu-optimized --gpus 2
```

### **Multi-GPU Benefits**
- **Effective batch size**: 1024 (512 per GPU)
- **Linear speedup**: ~1.8x faster than single GPU
- **Memory**: 48GB total VRAM (24GB per GPU)
- **Can use even larger batches**: Try 768/1536 per GPU

### **Multi-GPU Considerations**
- Learning rate scaling: Automatically handled by our 2e-4 LR
- Batch normalization: Works correctly with Lightning DDP
- Checkpoint compatibility: Works seamlessly between single/multi-GPU

## Advanced Optimizations

### **1. Memory-Bandwidth Optimization**
```python
# Enable in training script (already implemented):
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')  # Tensor Cores
```

### **2. Custom Batch Sizes for Your Hardware**
```bash
# Edit config for your specific VRAM:
# 24GB VRAM: batch_train=768, batch_val=1536
# 48GB VRAM (2x3090): batch_train=1024, batch_val=2048
```

### **3. Profiling Training Bottlenecks**
```bash
# Profile first few batches to find bottlenecks
python src/train_english.py --config configs/config_english_minimal_gpu_optimized.json --max-epochs 1 --profiler simple

# Check profiler output in lightning_logs/
```

## Troubleshooting

### **Out of Memory (OOM)**
```bash
# Reduce batch size in config:
# RTX 4090M: Try 384/768 if 512/1024 fails
# 2x RTX 3090: Try 768/1536 if 1024/2048 fails

# Or reduce num_workers:
# Edit config: "num_workers": 4
```

### **Slow Data Loading**
```bash
# Check if CPU is bottleneck:
htop  # Should see 8 Python processes with high CPU

# If num_workers too high for your RAM:
# Edit config: "num_workers": 4 or 2
```

### **Training Stalls**
```bash
# Check for deadlocks with persistent_workers:
# If training hangs, kill and restart:
pkill -f train_english.py
./run_training.py --gpu-optimized --checkpoint checkpoints/*/last.ckpt
```

## Summary Commands

```bash
# RTX 4090M (16GB VRAM)
./run_training.py --vram 16

# Resume training  
./run_training.py --vram 16 --checkpoint checkpoints/english_minimal_vram16gb/last.ckpt

# Multi-GPU training (2x RTX 3090)
./run_training.py --vram 48 --gpus 2

# Monitor progress
nvidia-smi -l 1
tensorboard --logdir lightning_logs/

# Find best model
ls checkpoints/english_minimal_vram16gb/ | grep accuracy | sort -r | head -1
```

**Expected Results**: 70-75% validation accuracy in 15-30 epochs with VRAM optimization:
- **RTX 4090M (16GB)**: 15-45 minutes
- **RTX 4090 (24GB)**: 12-30 minutes  
- **2x RTX 3090 (48GB)**: 6-18 minutes