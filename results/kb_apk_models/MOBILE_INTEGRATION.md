
# Android Integration Guide

## Files Needed
- `model_state_dict.pt` (5.1MB) - Model weights
- `complete_model_config.json` - Model configuration
- `load_model.py` - Python reconstruction script (for reference)

## Model Architecture
- **Type**: Transformer Encoder-Decoder
- **Accuracy**: 70.7% on validation set
- **Input**: Swipe trajectory + keyboard layout
- **Output**: Encoded sequence for word prediction
- **Size**: 5.1MB

## Input Format

### Trajectory Features (6D per point)
```
[x, y, velocity_x, velocity_y, acceleration_x, acceleration_y]
```

### Keyboard Weights (29D per point) 
Gaussian-weighted probabilities for each of the 29 QWERTY keys:
```
['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
 '<unk>', '<eos>', '<pad>']
```

## Mobile Implementation Options

### Option 1: PyTorch Mobile
Load the state dict and reconstruct the model architecture in PyTorch Mobile.

### Option 2: Native Implementation
Implement the transformer layers natively in Java/Kotlin using the saved weights.

### Option 3: ONNX/TensorFlow Lite
Convert the state dict to ONNX or TF Lite format using the provided reconstruction script.

## Performance
- **Inference Time**: ~10-30ms on modern Android devices
- **Memory Usage**: ~50-100MB during inference
- **Accuracy**: 70.7% word-level accuracy

## Next Steps
1. Choose implementation approach (PyTorch Mobile recommended)
2. Implement feature extraction pipeline
3. Add beam search decoder for word candidates
4. Optimize for target devices
