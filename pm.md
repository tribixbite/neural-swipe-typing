# Project Management - Neural Swipe Typing

## ONNX Export Issues (Critical)

### 🔴 Critical Issues
- [ ] **Silent Error Masking** - Returns random data on failure instead of failing loudly (lines 78-82)
- [ ] **Configuration Drift** - Hard-coded params don't match actual model (lines 165-168, 108-110)

### 🟠 High Priority Issues  
- [ ] **No Export Validation** - Missing verification that ONNX output matches PyTorch
- [ ] **Rigid Input Constraints** - Fixed seq length, flattened input, only last timestep
- [ ] **Outdated ONNX Configuration** - Uses opset v11 instead of v17+

### 🟡 Medium Priority Issues
- [ ] **Incomplete Tokenizer** - BPE config with no merges, wrong vocabulary
- [ ] **Missing Optimizations** - No ONNX optimization passes, quantization, or memory efficiency

## Fix Implementation Plan

1. Remove silent error handling - Delete try/except returning random data
2. Add export validation - Compare PyTorch vs ONNX outputs
3. Extract real model config - Use checkpoint hyperparameters 
4. Support variable sequence lengths with proper dynamic axes
5. Update to modern ONNX opset (v17)
6. Load proper tokenizer from training artifacts
7. Add ONNX optimization and quantization options
8. Implement memory-efficient checkpoint loading

## Progress
- [x] Issues identified by Gemini analysis
- [x] Fixes implemented in `regenerate_compatible_onnx_fixed.py`
- [x] Testing revealed model architecture incompatibilities
- [ ] Full ONNX export blocked by PyTorch graph conversion issues

## Implementation Status

### Completed Fixes
- ✅ Removed silent error handling (no more random data fallback)
- ✅ Added ONNX export validation function
- ✅ Extracted real model config from checkpoint
- ✅ Support for variable sequence lengths with dynamic axes
- ✅ Updated to modern ONNX opset (v14-17)
- ✅ Proper tokenizer configuration from vocabulary
- ✅ Added ONNX optimization passes
- ✅ Memory-efficient checkpoint loading

### Remaining Issues
- ⚠️ Model expects tuple input format (trajectory_features, keyboard_ids)
- ⚠️ Complex transformer architecture causes ONNX graph conversion errors
- ⚠️ Padding mask type incompatibilities between PyTorch and ONNX
- ⚠️ Both ONNX and TorchScript export fail due to "required keyword attribute 'value' has the wrong type"

## Complete Model Documentation Created

### Model Architecture Understood:
- **Input**: Swipe points (x,y,t) → 6D trajectory features + keyboard proximity ID
- **Encoder**: 4-layer transformer, 128 dims, 4 heads
- **Decoder**: Autoregressive character generation
- **Vocabulary**: 30 tokens (26 letters + special tokens)
- **Padding**: Boolean masks for variable-length sequences

### Export Strategies Attempted:
1. **ONNX Split Export** (export_onnx_working.py):
   - Separate encoder and decoder models
   - Input wrapper to combine tuple features
   - Failed due to graph conversion errors

2. **TorchScript Export** (export_torchscript.py):
   - Complete model with wrapper
   - Java interface code generated
   - Failed due to same attribute type error

### Root Cause:
The model uses complex positional encoding buffers and transformer layers that are incompatible with current PyTorch export mechanisms. The error occurs in the graph optimization phase.

### Recommended Next Steps:
1. Simplify model architecture (remove positional encoding buffers)
2. Use older PyTorch version with better export support
3. Implement model from scratch in target language
4. Use model serving instead of edge deployment