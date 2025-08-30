# Neural Swipe Typing Model - ExecuTorch Android Integration Guide

This directory contains the ExecuTorch-optimized neural swipe typing model for high-performance Android deployment.

## Files Included

- `english-epoch=51-val_loss=1.248-val_word_acc=0.659_inference_only.pte` - ExecuTorch model file
- `english-epoch=51-val_loss=1.248-val_word_acc=0.659_inference_only_metadata.json` - Model metadata
- `README_ExecuTorch_Android.md` - This integration guide

## Model Overview

**Export Approach**: Inference-Only (Single-Step Prediction)
- **Architecture**: v3_nearest_and_traj_transformer_bigger
- **Backend Optimization**: XNNPACK (Mobile CPU)
- **Validation Accuracy**: 65.9% word-level accuracy
- **Inference Mode**: Single character prediction per forward pass
- **Dynamic Shapes**: Supports trajectory lengths 10-299 points

## Architecture Details

This ExecuTorch model uses a **simplified inference-only approach** that avoids the complex encoder-decoder sequence generation. Instead of generating complete words, it:

1. **Encodes** the full swipe trajectory into a feature representation
2. **Predicts** the first character of the intended word
3. **Returns** logits for all 30 vocabulary characters

This approach is optimized for:
- **Fast inference** (single forward pass)
- **ExecuTorch compatibility** (no dynamic control flow)
- **Mobile efficiency** (reduced memory and compute)

## Input Format

The model expects **2 input tensors**:

### 1. Trajectory Features (`traj_feats`)
- **Shape**: `[seq_len, batch_size, 6]`
- **Type**: Float32
- **Description**: Sequence of 6D trajectory features per swipe point
- **Features**: `[x, y, dx/dt, dy/dt, d²x/dt², d²y/dt²]`
- **Dynamic Range**: seq_len can be 10-299 points

### 2. Keyboard Features (`kb_features`)
- **Shape**: `[seq_len, batch_size]` (for nearest model)
- **Type**: Int64
- **Description**: Nearest keyboard key ID for each trajectory point
- **Values**: 0-29 (character vocabulary indices)

## Output Format

**Shape**: `[batch_size, vocab_size]`  
**Type**: Float32  
**Description**: Logits for the **first character** prediction only
- Single character prediction (not full word)
- 30 character vocabulary classes
- Use with beam search or character-level language model for full words

## Android Integration

### 1. Dependencies

Add ExecuTorch Android dependencies to your `build.gradle`:

```gradle
dependencies {
    implementation 'org.pytorch:executorch:0.7.+'
    implementation 'org.pytorch:executorch-extension-llm:0.7.+'
}
```

### 2. Model Loading

```java
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

public class ExecuTorchSwipeDecoder {
    private Module model;
    
    public void loadModel(String modelPath) {
        try {
            model = Module.load(modelPath);
        } catch (Exception e) {
            Log.e("SwipeDecoder", "Failed to load ExecuTorch model", e);
        }
    }
}
```

### 3. Preprocessing Pipeline

```java
public class SwipePreprocessor {
    
    // Extract trajectory features with velocities and accelerations
    public float[][][] extractTrajectoryFeatures(List<TouchPoint> points) {
        int seqLen = Math.min(points.size(), 299); // Clamp to max length
        float[][][] features = new float[seqLen][1][6]; // [seq_len, batch_size, features]
        
        for (int i = 0; i < seqLen; i++) {
            TouchPoint curr = points.get(i);
            
            // Normalize coordinates to [0,1] based on keyboard bounds
            features[i][0][0] = normalizeX(curr.x);  // x
            features[i][0][1] = normalizeY(curr.y);  // y
            
            // Calculate velocities
            if (i > 0) {
                TouchPoint prev = points.get(i-1);
                float dt = (curr.timestamp - prev.timestamp) / 1000.0f; // seconds
                features[i][0][2] = (curr.x - prev.x) / dt;  // dx/dt
                features[i][0][3] = (curr.y - prev.y) / dt;  // dy/dt
            }
            
            // Calculate accelerations
            if (i > 1) {
                TouchPoint prev = points.get(i-1);
                TouchPoint prev2 = points.get(i-2);
                float dt1 = (curr.timestamp - prev.timestamp) / 1000.0f;
                float dt2 = (prev.timestamp - prev2.timestamp) / 1000.0f;
                float vx_curr = (curr.x - prev.x) / dt1;
                float vy_curr = (curr.y - prev.y) / dt1;
                float vx_prev = (prev.x - prev2.x) / dt2;
                float vy_prev = (prev.y - prev2.y) / dt2;
                
                features[i][0][4] = (vx_curr - vx_prev) / dt1;  // d²x/dt²
                features[i][0][5] = (vy_curr - vy_prev) / dt1;  // d²y/dt²
            }
        }
        
        return features;
    }
    
    // Find nearest keyboard keys
    public long[][] findNearestKeys(List<TouchPoint> points) {
        int seqLen = Math.min(points.size(), 299);
        long[][] keyIndices = new long[seqLen][1];
        
        for (int i = 0; i < seqLen; i++) {
            TouchPoint point = points.get(i);
            keyIndices[i][0] = findClosestKey(point.x, point.y);
        }
        
        return keyIndices;
    }
    
    private int findClosestKey(float x, float y) {
        // Implement keyboard layout distance calculation
        // Return character vocabulary index (0-29)
        // This depends on your specific keyboard layout
    }
}
```

### 4. Inference

```java
public class ExecuTorchSwipeDecoder {
    
    public char predictFirstCharacter(List<TouchPoint> touchPoints) {
        if (touchPoints.size() < 2) {
            return '\0'; // Need at least 2 points
        }
        
        // 1. Preprocess inputs
        float[][][] trajFeats = preprocessor.extractTrajectoryFeatures(touchPoints);
        long[][] kbFeatures = preprocessor.findNearestKeys(touchPoints);
        
        // 2. Create ExecuTorch tensors
        Tensor trajTensor = Tensor.fromBlob(flatten3D(trajFeats), 
            new long[]{trajFeats.length, 1, 6});
        Tensor kbTensor = Tensor.fromBlob(flatten2D(kbFeatures), 
            new long[]{kbFeatures.length, 1});
        
        // 3. Run inference
        Tensor[] outputs = model.forward(trajTensor, kbTensor);
        float[] logits = outputs[0].getDataAsFloatArray(); // [30] logits
        
        // 4. Get predicted character
        int bestIndex = argmax(logits);
        return indexToChar(bestIndex);
    }
    
    private float[] flatten3D(float[][][] array) {
        // Flatten 3D array to 1D for tensor creation
        int totalSize = array.length * array[0].length * array[0][0].length;
        float[] flattened = new float[totalSize];
        int idx = 0;
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                for (int k = 0; k < array[i][j].length; k++) {
                    flattened[idx++] = array[i][j][k];
                }
            }
        }
        return flattened;
    }
    
    private long[] flatten2D(long[][] array) {
        int totalSize = array.length * array[0].length;
        long[] flattened = new long[totalSize];
        int idx = 0;
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                flattened[idx++] = array[i][j];
            }
        }
        return flattened;
    }
}
```

### 5. Character Vocabulary

```java
// Character index mapping
private static final char[] VOCAB = {
    0,    // 0: PAD
    0,    // 1: SOS  
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    ' ', '.', '\''
};

private char indexToChar(int index) {
    return (index < VOCAB.length && index >= 2) ? VOCAB[index] : '\0';
}

private int argmax(float[] array) {
    int maxIndex = 0;
    for (int i = 1; i < array.length; i++) {
        if (array[i] > array[maxIndex]) {
            maxIndex = i;
        }
    }
    return maxIndex;
}
```

## Multi-Character Word Prediction

Since this model only predicts the **first character**, you have several options for complete words:

### Option 1: Character-Level Language Model
```java
public String predictWord(List<TouchPoint> touchPoints) {
    char firstChar = predictFirstCharacter(touchPoints);
    if (firstChar == '\0') return "";
    
    // Use a character-level language model or dictionary
    // to complete the word based on first character + context
    return completeWordWithLanguageModel(firstChar, touchPoints);
}
```

### Option 2: Dictionary Matching
```java
public List<String> getCandidateWords(List<TouchPoint> touchPoints) {
    char firstChar = predictFirstCharacter(touchPoints);
    if (firstChar == '\0') return new ArrayList<>();
    
    // Get dictionary words starting with predicted character
    List<String> candidates = dictionary.getWordsStartingWith(firstChar);
    
    // Score candidates based on swipe trajectory similarity
    return scoreAndRankCandidates(candidates, touchPoints);
}
```

### Option 3: Ensemble with ONNX Model
```java
public String predictWordEnsemble(List<TouchPoint> touchPoints) {
    // Use ExecuTorch for fast first-character prediction
    char firstChar = execuTorchDecoder.predictFirstCharacter(touchPoints);
    
    // Use ONNX model for full word prediction
    String fullWord = onnxDecoder.predictFullWord(touchPoints);
    
    // Combine predictions (e.g., prefer full word if it starts with first char)
    if (fullWord.length() > 0 && fullWord.charAt(0) == firstChar) {
        return fullWord;
    }
    
    // Fallback to dictionary matching
    return findBestDictionaryMatch(firstChar, touchPoints);
}
```

## Performance Optimization

### 1. Model Quantization
The exported model uses XNNPACK optimization. For further optimization:
- Model is already CPU-optimized via XNNPACK backend
- Consider INT8 quantization for even faster inference (if supported)

### 2. Preprocessing Optimization
```java
// Cache expensive calculations
private static final float KEYBOARD_WIDTH = 1080f;
private static final float KEYBOARD_HEIGHT = 400f;

// Pre-allocate arrays
private float[][][] trajFeatBuffer = new float[299][1][6];
private long[][] kbFeatBuffer = new long[299][1];

public void optimizedPreprocess(List<TouchPoint> points) {
    // Reuse pre-allocated buffers
    int len = Math.min(points.size(), 299);
    // ... fill buffers efficiently
}
```

### 3. Threading
```java
// Run inference on background thread
CompletableFuture<Character> prediction = CompletableFuture.supplyAsync(() -> {
    return predictFirstCharacter(touchPoints);
});
```

## Error Handling

```java
public char predictFirstCharacterSafe(List<TouchPoint> touchPoints) {
    try {
        if (touchPoints == null || touchPoints.size() < 2) {
            Log.w("SwipeDecoder", "Insufficient touch points");
            return '\0';
        }
        
        if (touchPoints.size() > 299) {
            Log.w("SwipeDecoder", "Truncating long trajectory");
            touchPoints = touchPoints.subList(0, 299);
        }
        
        return predictFirstCharacter(touchPoints);
        
    } catch (Exception e) {
        Log.e("SwipeDecoder", "Prediction failed", e);
        return '\0';
    }
}
```

## Performance Expectations

**ExecuTorch Model Benefits**:
- ⚡ **Faster inference**: Single forward pass vs. autoregressive generation
- 🧠 **Lower memory**: Simplified model architecture
- 📱 **Mobile optimized**: XNNPACK backend for ARM CPUs
- 🔋 **Energy efficient**: Reduced computation per prediction

**Trade-offs**:
- Single character prediction (requires additional logic for full words)
- Limited to first character accuracy
- May need ensemble with dictionary/language model

## Troubleshooting

**Common Issues**:

1. **Model loading fails**
   - Check ExecuTorch version compatibility
   - Ensure .pte file is in assets or accessible path
   - Verify Android app has file read permissions

2. **Tensor shape mismatches**
   - Verify input tensors match expected shapes exactly
   - Check seq_len is within [10, 299] range
   - Ensure batch_size = 1

3. **Poor predictions**
   - Verify coordinate normalization to keyboard bounds
   - Check velocity/acceleration calculations
   - Ensure nearest key lookup is accurate

4. **Performance issues**
   - Run inference on background thread
   - Pre-allocate tensor buffers
   - Cache keyboard layout calculations

## Integration Examples

See the repository for:
- Complete Android demo app
- Keyboard layout definitions
- Dictionary integration examples
- Performance benchmarking code

The ExecuTorch approach provides **fast, efficient first-character prediction** ideal for mobile keyboard applications with additional language modeling.