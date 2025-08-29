# Neural Swipe Typing Model - Android Integration Guide

This directory contains the exported neural swipe typing model for use in Android keyboard applications.

## Files Included

- `english-epoch=51-val_loss=1.248-val_word_acc=0.659.onnx` - ONNX format model
- `model_metadata.json` - Model configuration and metadata
- `README_Android_Integration.md` - This integration guide

## Model Architecture

**Model Type**: EncoderDecoderTransformerLike (v3_nearest_and_traj_transformer_bigger)
- **Vocabulary Size**: 30 characters
- **Model Dimension**: 128
- **Max Trajectory Length**: 299 points  
- **Max Word Length**: 35 characters
- **Validation Accuracy**: 65.9% word-level accuracy

## Input Format

The model expects 5 inputs:

### 1. Trajectory Features (`traj_feats`)
- **Shape**: `[seq_len, batch_size, 6]`
- **Type**: Float32
- **Description**: Sequence of 6D trajectory features per swipe point
- **Features**: `[x, y, dx/dt, dy/dt, d²x/dt², d²y/dt²]`
- **Preprocessing**: 
  - Normalize coordinates to keyboard bounds
  - Calculate velocities and accelerations from consecutive points
  - Pad sequences to max length

### 2. Keyboard Features (`kb_features`) 
- **Shape**: `[seq_len, batch_size]`
- **Type**: Int64
- **Description**: Nearest keyboard key ID for each trajectory point
- **Values**: 0-29 (character vocabulary indices)
- **Preprocessing**:
  - For each (x,y) point, find the closest keyboard key
  - Map key to character vocabulary index
  - Pad sequences to max length

### 3. Decoder Input (`decoder_input`)
- **Shape**: `[35, batch_size]`  
- **Type**: Int64
- **Description**: Input token sequence for decoder (starts with SOS token)
- **Values**: 
  - 0: PAD token
  - 1: SOS (Start of Sequence) token
  - 2-29: Character tokens
- **For inference**: Initialize with `[1, 0, 0, ..., 0]` (SOS + padding)

### 4. Encoder Padding Mask (`encoder_padding_mask`)
- **Shape**: `[batch_size, seq_len]`
- **Type**: Bool
- **Description**: `True` where trajectory is padded, `False` for valid points
- **Example**: For seq of length 50 padded to 100: `[False×50, True×50]`

### 5. Decoder Padding Mask (`decoder_padding_mask`)
- **Shape**: `[batch_size, 35]`
- **Type**: Bool  
- **Description**: `True` where decoder input is padded, `False` for valid tokens
- **For inference**: Usually `[False, True×34]` (only SOS token is valid initially)

## Output Format

**Shape**: `[35, batch_size, 30]`
**Type**: Float32
**Description**: Logits for each character at each position
- Dimension 0: Word position (max 35 characters)
- Dimension 1: Batch size
- Dimension 2: Character vocabulary (30 classes)

**Character Mapping**: 
```
0: PAD, 1: SOS, 2-29: ['a', 'b', 'c', ..., 'z', ' ', '.', "'"]
```

## Integration Steps

### 1. Model Loading
```java
// Load ONNX model
OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession("path/to/model.onnx");
```

### 2. Preprocessing Pipeline

```java
public class SwipePreprocessor {
    
    // Convert raw touch points to trajectory features
    public float[][][] extractTrajectoryFeatures(List<TouchPoint> points) {
        // points: List of (x, y, timestamp)
        // Returns: [seq_len, 1, 6] array with [x, y, vx, vy, ax, ay]
        
        // 1. Normalize coordinates to [0,1] based on keyboard bounds
        // 2. Calculate velocities: v = (p[i+1] - p[i]) / (t[i+1] - t[i])
        // 3. Calculate accelerations: a = (v[i+1] - v[i]) / (t[i+1] - t[i])
        // 4. Pad/truncate to max length (299)
    }
    
    // Find nearest keyboard keys
    public long[][] findNearestKeys(List<TouchPoint> points) {
        // Returns: [seq_len, 1] array with key indices (0-29)
        
        // For each point, find closest key on keyboard layout
        // Map key character to vocabulary index
        // Pad with 0 (PAD token) to max length
    }
    
    // Create padding masks
    public boolean[][] createEncoderMask(int actualLength) {
        // Returns: [1, max_seq_len] mask
        // False for positions < actualLength, True for padding
    }
}
```

### 3. Inference

```java
public class SwipeDecoder {
    
    public String decodeSwipe(List<TouchPoint> touchPoints) {
        // 1. Preprocess inputs
        float[][][] trajFeats = preprocessor.extractTrajectoryFeatures(touchPoints);
        long[][] kbFeatures = preprocessor.findNearestKeys(touchPoints);
        long[][] decoderInput = {{1, 0, 0, ...}}; // SOS + padding
        boolean[][] encMask = preprocessor.createEncoderMask(touchPoints.size());
        boolean[][] decMask = {{false, true, true, ...}}; // Only SOS valid
        
        // 2. Create ONNX tensors
        OnnxTensor trajTensor = OnnxTensor.createTensor(env, trajFeats);
        OnnxTensor kbTensor = OnnxTensor.createTensor(env, kbFeatures);
        OnnxTensor decTensor = OnnxTensor.createTensor(env, decoderInput);
        OnnxTensor encMaskTensor = OnnxTensor.createTensor(env, encMask);
        OnnxTensor decMaskTensor = OnnxTensor.createTensor(env, decMask);
        
        // 3. Run inference
        Map<String, OnnxTensor> inputs = Map.of(
            "traj_feats", trajTensor,
            "kb_features", kbTensor, 
            "decoder_input", decTensor,
            "encoder_padding_mask", encMaskTensor,
            "decoder_padding_mask", decMaskTensor
        );
        
        OrtSession.Result result = session.run(inputs);
        float[][][] logits = (float[][][]) result.get(0).getValue();
        
        // 4. Decode output
        return decodeLogits(logits[0][0]); // [35, 30] logits for first batch
    }
    
    private String decodeLogits(float[][] logits) {
        StringBuilder word = new StringBuilder();
        
        for (int pos = 0; pos < logits.length; pos++) {
            int bestChar = argmax(logits[pos]);
            
            // Skip PAD (0) and SOS (1) tokens
            if (bestChar <= 1) continue;
            
            // Convert index to character
            char c = indexToChar(bestChar);
            if (c == 0) break; // End of word
            
            word.append(c);
        }
        
        return word.toString();
    }
}
```

### 4. Character Vocabulary

```java
// Character index mapping (adjust based on actual vocabulary)
private static final char[] VOCAB = {
    0,    // 0: PAD
    0,    // 1: SOS  
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ' ', '.', '\''
};

private char indexToChar(int index) {
    return (index < VOCAB.length) ? VOCAB[index] : 0;
}
```

## Performance Optimization

### 1. Model Quantization
Consider quantizing the ONNX model to INT8 for mobile deployment:
```bash
python -m onnxruntime.quantization.quantize_static \
    --model_input model.onnx \
    --model_output model_quantized.onnx \
    --calibration_data_reader your_calibration_data
```

### 2. Batch Processing
- For real-time use, batch_size=1 is sufficient
- For offline processing, increase batch size for better throughput

### 3. Input Preprocessing
- Cache keyboard layout and key positions
- Use efficient distance calculations for nearest key lookup
- Pre-allocate arrays for trajectory features

## Error Handling

```java
// Handle invalid inputs
if (touchPoints.size() < 2) {
    throw new IllegalArgumentException("Need at least 2 touch points");
}

if (touchPoints.size() > 299) {
    // Truncate to max sequence length
    touchPoints = touchPoints.subList(0, 299);
}

// Handle model inference errors
try {
    OrtSession.Result result = session.run(inputs);
    // Process result...
} catch (OrtException e) {
    Log.e("SwipeDecoder", "ONNX inference failed", e);
    return ""; // Return empty string or fallback result
}
```

## Testing

1. **Unit Tests**: Test preprocessing functions with known inputs
2. **Integration Tests**: Test end-to-end pipeline with sample swipe data
3. **Performance Tests**: Measure inference latency and memory usage
4. **Accuracy Tests**: Compare outputs with expected words from validation set

## Troubleshooting

**Common Issues:**
- **Shape mismatches**: Ensure input tensors match expected dimensions exactly
- **Data type errors**: Use correct data types (Float32 for features, Int64 for tokens, Bool for masks)
- **Padding errors**: Ensure proper padding with correct tokens (0 for PAD, 1 for SOS)
- **Coordinate normalization**: Normalize touch coordinates to keyboard bounds before processing

**Performance Issues:**
- Use ONNX Runtime optimization levels
- Consider model quantization for faster inference
- Profile preprocessing pipeline for bottlenecks
- Cache keyboard layout calculations

## Model Details

- **Training Dataset**: English swipe typing dataset
- **Architecture**: Transformer encoder-decoder with trajectory and keyboard embeddings
- **Validation Performance**: 65.9% word-level accuracy
- **Model Size**: ~5.8MB (ONNX format)

For questions or issues, refer to the original training code in the repository.