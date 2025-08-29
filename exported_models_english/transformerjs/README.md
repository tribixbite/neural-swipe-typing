# Neural Swipe Decoder for Web Applications

**Note**: The PyTorch model is too complex for direct ONNX export to transformer.js. This package provides the JavaScript interface and configuration files, but requires a PyTorch backend for inference.

## Files

- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer configuration  
- `neural_swipe_decoder.js`: JavaScript interface (requires PyTorch backend)

## Usage Options

### Option 1: Server-Side Inference (Recommended)
Set up a Python FastAPI/Flask server with the PyTorch model:

```python
# server.py
from fastapi import FastAPI
import torch
import numpy as np

app = FastAPI()

# Load your PyTorch model here
model = load_your_model()

@app.post("/api/swipe/decode")
async def decode_swipe(data: dict):
    trajectory_features = np.array(data['trajectory_features'])
    keyboard_features = np.array(data['keyboard_features'])
    
    # Convert to tensors and run inference
    with torch.no_grad():
        predictions = model(trajectory_features, keyboard_features)
    
    return {"predictions": predictions.tolist()}
```

### Option 2: Model Simplification
Consider creating a simplified version of the model that can be exported to ONNX:
- Remove complex transformer components that don't export well
- Use simpler attention mechanisms
- Pre-compute certain operations

### Option 3: Use TensorFlow.js
Convert the model to TensorFlow format first, then to TensorFlow.js.

## Client Usage

```javascript
import { NeuralSwipeDecoder } from './neural_swipe_decoder.js';

const decoder = new NeuralSwipeDecoder();
await decoder.initialize();

const swipePoints = [{x: 0.1, y: 0.5, t: 0}, ...];
const result = await decoder.decode(swipePoints);

// Send result.features to your PyTorch backend
const response = await fetch('/api/swipe/decode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({trajectory_features: result.features})
});
```
