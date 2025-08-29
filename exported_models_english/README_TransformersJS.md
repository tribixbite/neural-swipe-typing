# Neural Swipe Typing for transformer.js

This directory contains the neural swipe typing model export for use with the Hugging Face transformer.js library, enabling AI-powered swipe gesture recognition directly in web browsers.

## 🎯 Overview

The neural swipe typing system converts touch gestures on virtual keyboards into word predictions using transformer neural networks. This export makes the model available for web applications through transformer.js.

## 📁 Files Structure

```
exported_models_english/
├── scripts/
│   └── convert_to_transformersjs.py     # Model conversion script
├── transformers_js_integration.js       # Main JavaScript integration
├── web_demo.html                        # Interactive web demo
└── README_TransformersJS.md            # This file
```

## 🚀 Quick Start

### 1. Convert Model to ONNX

First, convert your PyTorch Lightning checkpoint to transformer.js compatible format:

```bash
cd /path/to/neural-swipe-typing
python scripts/convert_to_transformersjs.py \
    checkpoints_english/english-epoch=51-val_loss=1.248-val_word_acc=0.659.ckpt \
    --output-dir ./transformers_model \
    --create-demo
```

This creates:
- `huggingface_model/`: HuggingFace-compatible model files
- `onnx/`: ONNX model for transformer.js
- `web_demo/`: Web demo files

### 2. Use in Web Applications

```javascript
import { NeuralSwipeDecoder } from './transformers_js_integration.js';

// Initialize decoder
const decoder = new NeuralSwipeDecoder({
    modelPath: './onnx/',  // Path to ONNX model
    decoding: {
        maxLength: 30,
        topK: 5,
        temperature: 0.8
    }
});

await decoder.initialize();

// Process swipe gesture
const swipePoints = [
    {x: 0.1, y: 0.5, t: 0},     // Start position
    {x: 0.2, y: 0.5, t: 50},   // Intermediate point
    {x: 0.3, y: 0.5, t: 100},  // End position
];

const predictions = await decoder.decode(swipePoints);
console.log(predictions);
// Output: [{word: 'hello', score: 0.95, rank: 1}, ...]
```

### 3. Try the Demo

Open `web_demo.html` in a web browser to see the interactive demo. The demo includes:
- Virtual QWERTY keyboard
- Real-time swipe gesture tracking
- Word prediction display
- Visual feedback and animations

## 🏗️ Architecture

### Model Pipeline

1. **Input Processing**: Raw swipe coordinates (x, y, t)
2. **Feature Extraction**: 
   - Trajectory features: position, velocity, acceleration
   - Keyboard features: distance-weighted key probabilities
3. **Neural Network**: Encoder-decoder transformer
4. **Output**: Character-level predictions → word suggestions

### JavaScript Integration

The `NeuralSwipeDecoder` class handles:
- Model loading via transformer.js
- Real-time feature extraction
- Keyboard layout management
- Prediction post-processing

## 📊 Model Details

- **Architecture**: EncoderDecoderTransformerLike
- **Input Features**: 
  - Trajectory: 6 dimensions (x, y, vx, vy, ax, ay)
  - Keyboard: 30 dimensions (key weight distribution)
- **Vocabulary**: 30 characters (letters, space, punctuation)
- **Max Sequence Length**: 299 (encoder) / 35 (decoder)
- **Model Size**: ~15MB (quantized ONNX)

## 🔧 Configuration Options

### Model Configuration

```javascript
const config = {
    modelPath: './onnx/',
    keyboard: {
        keys: [...],  // Custom keyboard layout
    },
    decoding: {
        maxLength: 35,
        topK: 10,
        topP: 0.9,
        temperature: 1.0,
        beamSize: 5
    }
};
```

### Feature Extraction Parameters

- **Trajectory smoothing**: Built-in velocity/acceleration calculation
- **Keyboard weighting**: Exponential distance falloff
- **Normalization**: Automatic coordinate normalization

## 🌐 Browser Compatibility

- **Modern browsers** with ES6 module support
- **WebAssembly** support (for ONNX Runtime)
- **Canvas/Touch API** for gesture tracking
- **Optional WebGPU** for accelerated inference

### Performance Notes

- **CPU inference**: ~50-100ms per prediction
- **Memory usage**: ~100MB including model
- **WebGPU**: 5-10x speedup when available

## 🛠️ Development

### Prerequisites

```bash
# Python dependencies
pip install torch transformers optimum[onnxruntime]

# JavaScript dependencies (for web integration)
npm install @huggingface/transformers
```

### Model Conversion Process

1. **Load PyTorch model** from Lightning checkpoint
2. **Wrap for HuggingFace** compatibility
3. **Export to ONNX** using Optimum
4. **Generate tokenizer** and config files
5. **Create web integration** files

### Custom Models

To use your own trained model:

1. Ensure checkpoint follows the expected format
2. Update model name in conversion script
3. Adjust feature extraction if needed
4. Re-run conversion process

## 📝 API Reference

### NeuralSwipeDecoder

#### Methods

- `initialize(modelPath, options)`: Load model and setup
- `decode(swipePoints, options)`: Process gesture → predictions
- `extractTrajectoryFeatures(points)`: Get motion features
- `extractKeyboardFeatures(points)`: Get keyboard proximity
- `updateConfig(config)`: Update decoder settings

#### Events

- Model loading progress
- Prediction updates (streaming)
- Error handling

## 🔍 Troubleshooting

### Common Issues

1. **ONNX conversion fails**:
   - Model too complex for automatic export
   - Use model simplification or manual export
   - Consider alternative backends (TensorRT, OpenVINO)

2. **Large model size**:
   - Apply quantization (fp16, int8)
   - Use model distillation
   - Implement progressive loading

3. **Performance issues**:
   - Enable WebGPU acceleration
   - Reduce sequence length
   - Optimize feature extraction

4. **CORS errors**:
   - Serve files from web server (not file://)
   - Configure proper headers for WASM/model files

### Debug Mode

Enable debug logging:

```javascript
const decoder = new NeuralSwipeDecoder({
    debug: true
});

// Or double-click status bar in web demo
```

## 🤝 Contributing

1. **Model improvements**: Train better architectures
2. **Web integration**: Enhance JavaScript library
3. **Performance**: Optimize inference and features
4. **Documentation**: Improve guides and examples

## 📄 License

This neural swipe typing implementation is provided as-is for research and development purposes. Please ensure compliance with your use case requirements.

---

## 🔗 Related Resources

- [Hugging Face transformer.js](https://huggingface.co/docs/transformers.js)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html)
- [Neural Swipe Paper](link-to-research-paper)
- [Live Demo](link-to-hosted-demo)

For questions and support, please refer to the main project documentation or open an issue in the repository.