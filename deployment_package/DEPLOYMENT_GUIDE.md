# Character-Level Swipe Typing Model Deployment Guide

## Model Overview
- **Architecture**: Transformer-based character-level generation
- **Accuracy**: 70.1% word accuracy
- **Parameters**: 8.97M
- **Input**: 6D trajectory features (x, y, vx, vy, ax, ay) + nearest keys
- **Output**: Character sequence with beam search

## Exported Formats

### 1. ONNX (Web/Browser)
- **Encoder**: `swipe_model_character.onnx` (18880.2 KB)
- **Decoder**: `swipe_decoder_character.onnx` (16962.3 KB)
- **Runtime**: ONNX Runtime Web (onnxruntime-web)
- **Supported Platforms**: Chrome, Firefox, Safari, Edge

### 2. ExecuTorch (Mobile)
- **Model**: `swipe_model_character.pte` (0.0 KB)
- **Runtime**: ExecuTorch with XNNPACK backend
- **Supported Platforms**: Android, iOS

## Web Deployment (ONNX)

### Installation
```bash
npm install onnxruntime-web
```

### TypeScript Integration
```typescript
import * as ort from 'onnxruntime-web';

class SwipePredictor {
    private encoderSession: ort.InferenceSession;
    private decoderSession: ort.InferenceSession;
    private tokenizer: Tokenizer;
    
    async loadModels(encoderUrl: string, decoderUrl: string) {
        this.encoderSession = await ort.InferenceSession.create(encoderUrl);
        this.decoderSession = await ort.InferenceSession.create(decoderUrl);
        this.tokenizer = new Tokenizer(); // Load from tokenizer_config.json
    }
    
    async predictWord(swipePoints: SwipePoint[]): Promise<string> {
        // 1. Extract features
        const features = this.extractFeatures(swipePoints);
        const nearestKeys = this.findNearestKeys(swipePoints);
        
        // 2. Run encoder
        const encoderInputs = {
            trajectory_features: new ort.Tensor('float32', features.data, features.shape),
            nearest_keys: new ort.Tensor('int64', nearestKeys.data, nearestKeys.shape),
            src_mask: new ort.Tensor('bool', maskData, maskShape)
        };
        
        const encoderOutputs = await this.encoderSession.run(encoderInputs);
        const memory = encoderOutputs.encoder_output;
        
        // 3. Run beam search with decoder
        return await this.beamSearch(memory, 5);
    }
    
    private async beamSearch(memory: ort.Tensor, beamSize: number): Promise<string> {
        // Implement beam search using decoder
        // See web_integration_full.ts for complete implementation
    }
}
```

## Android Deployment (ExecuTorch)

### Setup
1. Add ExecuTorch to your Android project:
```gradle
dependencies {
    implementation 'org.pytorch:executorch-android:0.1.0'
}
```

2. Load and run the model:
```kotlin
class SwipePredictor(context: Context) {
    private lateinit var module: Module
    
    init {
        val modelPath = getAssetFilePath(context, "swipe_model_character.pte")
        module = Module.load(modelPath)
    }
    
    fun predictWord(swipePoints: List<SwipePoint>): String {
        // Extract features
        val features = extractFeatures(swipePoints)
        val nearestKeys = findNearestKeys(swipePoints)
        
        // Create input tensors
        val inputTensor = Tensor.fromBlob(
            features,
            longArrayOf(1, swipePoints.size.toLong(), 6)
        )
        val keysTensor = Tensor.fromBlob(
            nearestKeys,
            longArrayOf(1, swipePoints.size.toLong(), 3)
        )
        
        // Run inference
        val outputTensor = module.forward(
            IValue.from(inputTensor),
            IValue.from(keysTensor)
        ).toTensor()
        
        // Decode output
        return decodeOutput(outputTensor)
    }
}
```

## iOS Deployment (ExecuTorch)

### Setup
1. Add ExecuTorch to your iOS project via CocoaPods:
```ruby
pod 'ExecuTorch', '~> 0.1.0'
```

2. Swift implementation:
```swift
import ExecuTorch

class SwipePredictor {
    private var module: ETModule!
    
    init() {
        let modelPath = Bundle.main.path(forResource: "swipe_model_character", ofType: "pte")!
        module = try! ETModule(contentsOfFile: modelPath)
    }
    
    func predictWord(swipePoints: [SwipePoint]) -> String {
        // Extract features
        let features = extractFeatures(from: swipePoints)
        let nearestKeys = findNearestKeys(for: swipePoints)
        
        // Create tensors
        let inputTensor = try! ETTensor(
            data: features,
            shape: [1, swipePoints.count, 6]
        )
        let keysTensor = try! ETTensor(
            data: nearestKeys,
            shape: [1, swipePoints.count, 3]
        )
        
        // Run inference
        let output = try! module.forward([inputTensor, keysTensor])
        
        // Decode
        return decodeOutput(output[0])
    }
}
```

## Feature Extraction

All platforms must implement consistent feature extraction:

```python
def extract_features(points):
    '''Extract 6D features from swipe points.'''
    features = []
    
    for i, point in enumerate(points):
        # Normalize coordinates
        x = point.x / 360  # keyboard width
        y = point.y / 215  # keyboard height
        
        # Calculate velocity
        if i > 0:
            dt = max(point.t - points[i-1].t, 1)
            vx = (point.x - points[i-1].x) / dt
            vy = (point.y - points[i-1].y) / dt
        else:
            vx = vy = 0
        
        # Calculate acceleration
        if i > 1:
            # ... (see complete implementation)
        else:
            ax = ay = 0
        
        # Normalize and clip
        vx = clip(vx / 1000, -1, 1)
        vy = clip(vy / 1000, -1, 1)
        ax = clip(ax / 500, -1, 1)
        ay = clip(ay / 500, -1, 1)
        
        features.append([x, y, vx, vy, ax, ay])
    
    return features
```

## Beam Search Implementation

The model uses beam search for better accuracy:

```javascript
async function beamSearch(memory, beamSize = 5) {
    let beams = [{
        tokens: [SOS_TOKEN],
        score: 0,
        finished: false
    }];
    
    for (let step = 0; step < MAX_LENGTH; step++) {
        let allCandidates = [];
        
        for (const beam of beams) {
            if (beam.finished) {
                allCandidates.push(beam);
                continue;
            }
            
            // Run decoder
            const logits = await runDecoder(memory, beam.tokens);
            const probs = softmax(logits);
            
            // Get top k tokens
            const topK = getTopK(probs, beamSize);
            
            for (const [token, prob] of topK) {
                const newBeam = {
                    tokens: [...beam.tokens, token],
                    score: beam.score + Math.log(prob),
                    finished: token === EOS_TOKEN
                };
                allCandidates.push(newBeam);
            }
        }
        
        // Keep top beams
        beams = allCandidates
            .sort((a, b) => b.score - a.score)
            .slice(0, beamSize);
        
        // Check if all beams are finished
        if (beams.every(b => b.finished)) break;
    }
    
    // Return best beam
    return decodeTokens(beams[0].tokens);
}
```

## Performance Optimization

### Web (ONNX)
- Use WebAssembly SIMD for 2-3x speedup
- Enable WebGL backend for GPU acceleration
- Cache model sessions between predictions
- Use Web Workers for non-blocking inference

### Mobile (ExecuTorch)
- Use XNNPACK backend for CPU optimization
- Enable GPU delegation where available
- Implement model quantization for smaller size
- Use batch processing for multiple predictions

## Testing

### Unit Tests
```javascript
describe('SwipePredictor', () => {
    it('should predict "hello" correctly', async () => {
        const points = [
            {x: 96, y: 167, t: 0},   // h
            {x: 124, y: 167, t: 50}, // e
            {x: 152, y: 167, t: 100}, // l
            {x: 152, y: 167, t: 150}, // l
            {x: 208, y: 167, t: 200}  // o
        ];
        
        const prediction = await predictor.predictWord(points);
        expect(prediction).toBe('hello');
    });
});
```

### Integration Tests
- Test with real swipe data from `data/combined_dataset/`
- Verify 70% accuracy on test set
- Test edge cases (short words, long words, unusual patterns)

## Troubleshooting

### Common Issues

1. **Low accuracy**: Ensure feature extraction matches training
2. **Slow inference**: Enable hardware acceleration
3. **Memory issues**: Use model quantization
4. **Crashes**: Check tensor shapes and data types

### Debug Mode
Enable verbose logging to debug issues:
```javascript
predictor.setDebugMode(true);
```

## Resources

- Model weights: `checkpoints/full_character_model/`
- Training code: `train_character_model.py`
- Evaluation: `evaluate_character_model.py`
- Dataset: `data/combined_dataset/`

## Support

For issues or questions:
- GitHub: [repository-url]
- Documentation: See `docs/` folder
- Examples: See `examples/` folder

---
Generated with 70.1% word accuracy on 68,848 training samples.
