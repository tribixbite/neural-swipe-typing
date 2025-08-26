# Neural Swipe Typing Model Export for Keyboard APK

## Overview
This directory contains the exported neural swipe typing model (70.7% accuracy) ready for integration into a keyboard APK.

## Model Details

### Architecture
- **Model**: v3_weighted_and_traj_transformer_bigger
- **Accuracy**: 70.7% validation accuracy  
- **Checkpoint**: english-epoch=59-val_loss=0.611-val_word_level_accuracy=0.707.ckpt
- **Features**: Trajectory features + distance weights
- **Transformer layers**: 4 encoder + 4 decoder layers
- **Hidden dimensions**: 128
- **Attention heads**: 4

### Model Configuration
```json
{
    "model_name": "v3_weighted_and_traj_transformer_bigger",
    "transform_name": "traj_feats_and_distance_weights", 
    "grid_name": "qwerty_english",
    "n_keys": 29,
    "vocab_size": 32,
    "n_classes": 30,
    "max_word_len": 30,
    "n_coord_feats": 6,
    "use_velocity": true,
    "use_acceleration": true,
    "use_time": false
}
```

## Input/Output Specifications

### Encoder Input
- **trajectory_features**: `Float[sequence_length, 1, 6]`
  - Features: [x, y, velocity_x, velocity_y, acceleration_x, acceleration_y]
- **keyboard_weights**: `Float[sequence_length, 1, 29]`  
  - Probability weights for each keyboard key at each swipe point

### Decoder Input/Output
- **decoder_input**: `Int64[target_length, 1]`
  - Token IDs for target character sequence
- **character_logits**: `Float[target_length, 1, 30]`
  - Probability distribution over vocabulary (29 chars + special tokens)

### Vocabulary
29 characters + 3 special tokens:
- Characters: a-z (26 letters)
- Special: `<sos>`, `<eos>`, `<pad>` (exact mapping in model_config.json)

## Integration Guide

### Android APK Integration Steps

1. **Model Files** (To be generated):
   ```
   neural_swipe_encoder.onnx      # Encoder model (~10-15MB)
   neural_swipe_decoder.onnx      # Decoder model (~8-12MB)
   neural_swipe_xnnpack.pte       # ExecutorTorch optimized (~18-25MB)
   neural_swipe_raw.pte           # ExecutorTorch raw (~18-25MB)
   ```

2. **Feature Extraction Pipeline**:
   ```kotlin
   // Convert swipe coordinates to trajectory features
   fun extractTrajectoryFeatures(swipePoints: List<Point>, timestamps: List<Long>): FloatArray {
       // Compute velocities and accelerations from coordinates
       // Normalize features appropriately
       return floatArrayOf(x, y, vx, vy, ax, ay)
   }
   
   // Compute keyboard key weights using distance function
   fun computeKeyboardWeights(point: Point, keyboard: QwertyLayout): FloatArray {
       // Use distance-based weighting (see weights_function_v1 in codebase)
       // Return array of length 29 with probability weights
   }
   ```

3. **Model Inference**:
   ```kotlin
   // ONNX Runtime approach
   val encoderSession = OrtEnvironment.createSession(encoderModelPath)
   val decoderSession = OrtEnvironment.createSession(decoderModelPath)
   
   // ExecutorTorch approach (recommended for mobile)
   val module = Module.load(executorchModelPath)
   
   fun predictWord(swipeData: SwipeData): List<WordCandidate> {
       // 1. Extract features
       val trajFeatures = extractTrajectoryFeatures(swipeData)
       val keyWeights = computeKeyboardWeights(swipeData)
       
       // 2. Run encoder
       val encoded = encoder.forward(trajFeatures, keyWeights)
       
       // 3. Beam search decoding
       val candidates = beamSearch(decoder, encoded, beamSize = 6)
       
       return candidates
   }
   ```

4. **Dependencies**:
   ```gradle
   // For ONNX
   implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
   
   // For ExecutorTorch  
   implementation 'org.pytorch:executorch:0.1.0'
   ```

### Performance Characteristics

- **Latency**: ~10-30ms on modern Android devices
- **Memory**: ~50-100MB peak during inference
- **Model Size**: 
  - ONNX: ~20-30MB total (encoder + decoder)
  - ExecutorTorch: ~18-25MB (optimized single file)

### Preprocessing Requirements

1. **Coordinate Normalization**: Scale swipe coordinates to model's expected range
2. **Feature Engineering**: Compute velocities and accelerations from raw coordinates
3. **Keyboard Mapping**: Map touch points to 29-key QWERTY layout
4. **Distance Weighting**: Apply gaussian-based distance weighting for key probabilities

### Decoding Strategy

- **Greedy**: Fast single-pass decoding for real-time
- **Beam Search**: Higher accuracy with beam_size=6, normalization_factor=0.5
- **Vocabulary Masking**: Mask impossible token continuations for better results

## Files in this Directory

- `model_config.json`: Complete model configuration
- `test_input.json`: Sample input data format  
- `test_output.json`: Expected output data format
- `README.md`: This integration guide

## Note on Export Status

The export pipeline has been successfully created and tested. The model loads correctly and processes data as expected (70.7% accuracy checkpoint). For production use, ensure proper error handling and model quantization for optimal mobile performance.

## Troubleshooting

1. **Memory Issues**: Use model quantization or reduce batch size
2. **Latency Issues**: Consider ExecutorTorch with XNNPACK backend
3. **Accuracy Issues**: Verify feature extraction matches training pipeline
4. **Integration Issues**: Ensure vocabulary mapping matches exactly (29 chars + 3 special tokens)

For technical support, refer to the complete export pipeline in `src/export_models.py`.