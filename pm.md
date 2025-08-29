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
- ⚠️ May need to simplify model architecture or use TorchScript instead