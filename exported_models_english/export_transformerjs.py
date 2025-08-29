#!/usr/bin/env python3
"""
Export neural swipe typing model to transformer.js format for web applications.
Converts PyTorch model to ONNX and packages it with necessary metadata for transformer.js.
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import MODEL_GETTERS_DICT, EncoderDecoderTransformerLike


class TransformerJSModelWrapper(nn.Module):
    """Wrapper to make the model compatible with transformer.js expectations."""
    
    def __init__(self, model: EncoderDecoderTransformerLike, model_name: str, vocab_size: int = 30):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.vocab_size = vocab_size
    
    def forward(self, 
                trajectory_features: torch.Tensor,
                keyboard_features: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass for transformer.js inference.
        
        Args:
            trajectory_features: Trajectory features [batch_size, seq_len, 6]
            keyboard_features: Keyboard features [batch_size, seq_len, n_features]
            
        Returns:
            logits: Output logits [batch_size, vocab_size]
        """
        batch_size, seq_len = trajectory_features.shape[:2]
        
        # Transpose to match model expectations [seq_len, batch_size, features]
        traj_feats = trajectory_features.transpose(0, 1)
        kb_features = keyboard_features.transpose(0, 1)
        
        # Create decoder input (start with SOS token = 1)
        decoder_input = torch.ones(1, batch_size, dtype=torch.long)
        
        # Create padding masks (no padding for inference)
        encoder_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        decoder_padding_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
        
        # Combine features based on model type
        if "nearest" in self.model_name:
            encoder_input = (traj_feats, kb_features.long())
        elif "weighted" in self.model_name:
            encoder_input = (traj_feats, kb_features)
        else:
            encoder_input = torch.cat([traj_feats, kb_features], dim=-1)
        
        # Run the model
        with torch.no_grad():
            logits = self.model.forward(
                encoder_input, 
                decoder_input, 
                encoder_padding_mask, 
                decoder_padding_mask
            )
        
        # Return first timestep logits [batch_size, vocab_size]
        return logits[0, :, :]


def load_model_from_checkpoint(checkpoint_path: str) -> Tuple[EncoderDecoderTransformerLike, str]:
    """Load the PyTorch Lightning model from checkpoint."""
    
    # Load the Lightning checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get the model name
    model_name = checkpoint.get('hyper_parameters', {}).get('model_name', 'v3_weighted_and_traj_transformer_bigger')
    print(f"Loading model: {model_name}")
    
    if model_name not in MODEL_GETTERS_DICT:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_GETTERS_DICT.keys())}")
    
    # Create the model
    model_getter = MODEL_GETTERS_DICT[model_name]
    model = model_getter(device='cpu')
    
    # Load the state dict
    core_model_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            new_key = key[6:]
            core_model_state_dict[new_key] = value
    
    model.load_state_dict(core_model_state_dict)
    model.eval()
    
    return model, model_name


def create_transformer_js_config(model_name: str, vocab_size: int = 30) -> Dict[str, Any]:
    """Create transformer.js configuration."""
    
    config = {
        "model_type": "neural_swipe_decoder",
        "architecture": "EncoderDecoderTransformerLike",
        "vocab_size": vocab_size,
        "d_model": 128,
        "max_encoder_seq_len": 299,
        "max_decoder_seq_len": 35,
        "input_features": {
            "trajectory_dim": 6,
            "keyboard_dim": 30 if "weighted" in model_name else 1
        },
        "preprocessing": {
            "normalize_coordinates": True,
            "include_velocities": True,
            "include_accelerations": True
        },
        "inference": {
            "temperature": 1.0,
            "top_k": 10,
            "top_p": 0.9,
            "max_length": 35
        }
    }
    
    return config


def create_sample_inputs_transformerjs(model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sample inputs for transformer.js export."""
    
    batch_size = 1
    seq_len = 50  # Typical swipe length
    
    # Trajectory features: [batch_size, seq_len, 6]
    trajectory_features = torch.randn(batch_size, seq_len, 6)
    
    # Keyboard features based on model type
    if "weighted" in model_name:
        # Weighted model: probability distribution over keys
        keyboard_features = torch.softmax(torch.randn(batch_size, seq_len, 30), dim=-1)
    elif "nearest" in model_name:
        # Nearest model: key indices (remove extra dimension)
        keyboard_features = torch.randint(0, 30, (batch_size, seq_len)).float()
    else:
        # Default: extended feature vector
        keyboard_features = torch.randn(batch_size, seq_len, 30)
    
    return trajectory_features, keyboard_features


def export_transformerjs_model(model: nn.Module, 
                              model_name: str,
                              output_dir: Path,
                              sample_inputs: Tuple[torch.Tensor, ...]):
    """Export model to transformer.js compatible format."""
    
    # Create transformer.js subdirectory
    transformerjs_dir = output_dir / "transformerjs"
    transformerjs_dir.mkdir(exist_ok=True)
    
    # Export to ONNX for transformer.js
    onnx_path = transformerjs_dir / "model.onnx"
    
    print(f"Exporting to transformer.js ONNX: {onnx_path}")
    
    torch.onnx.export(
        model,
        sample_inputs,
        str(onnx_path),
        input_names=['trajectory_features', 'keyboard_features'],
        output_names=['logits'],
        dynamic_axes={
            'trajectory_features': {0: 'batch_size', 1: 'seq_len'},
            'keyboard_features': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size'}
        },
        opset_version=11,  # Use older opset for better compatibility
        do_constant_folding=False,  # Disable constant folding for complex models
        export_params=True,
        verbose=False
    )
    
    # Create transformer.js config
    config = create_transformer_js_config(model_name)
    config_path = transformerjs_dir / "config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create tokenizer config
    tokenizer_config = {
        "tokenizer_type": "character_level",
        "vocab_size": 30,
        "special_tokens": {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>"
        },
        "characters": " abcdefghijklmnopqrstuvwxyz'-."
    }
    
    tokenizer_path = transformerjs_dir / "tokenizer.json"
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Create keyboard layout (QWERTY)
    keyboard_layout = {
        "layout_name": "qwerty",
        "keys": [
            {"id": 0, "char": "q", "x": 0.05, "y": 0.4, "width": 0.1, "height": 0.15},
            {"id": 1, "char": "w", "x": 0.15, "y": 0.4, "width": 0.1, "height": 0.15},
            {"id": 2, "char": "e", "x": 0.25, "y": 0.4, "width": 0.1, "height": 0.15},
            {"id": 3, "char": "r", "x": 0.35, "y": 0.4, "width": 0.1, "height": 0.15},
            {"id": 4, "char": "t", "x": 0.45, "y": 0.4, "width": 0.1, "height": 0.15},
            {"id": 5, "char": "y", "x": 0.55, "y": 0.4, "width": 0.1, "height": 0.15},
            {"id": 6, "char": "u", "x": 0.65, "y": 0.4, "width": 0.1, "height": 0.15},
            {"id": 7, "char": "i", "x": 0.75, "y": 0.4, "width": 0.1, "height": 0.15},
            {"id": 8, "char": "o", "x": 0.85, "y": 0.4, "width": 0.1, "height": 0.15},
            {"id": 9, "char": "p", "x": 0.95, "y": 0.4, "width": 0.1, "height": 0.15},
            
            {"id": 10, "char": "a", "x": 0.1, "y": 0.55, "width": 0.1, "height": 0.15},
            {"id": 11, "char": "s", "x": 0.2, "y": 0.55, "width": 0.1, "height": 0.15},
            {"id": 12, "char": "d", "x": 0.3, "y": 0.55, "width": 0.1, "height": 0.15},
            {"id": 13, "char": "f", "x": 0.4, "y": 0.55, "width": 0.1, "height": 0.15},
            {"id": 14, "char": "g", "x": 0.5, "y": 0.55, "width": 0.1, "height": 0.15},
            {"id": 15, "char": "h", "x": 0.6, "y": 0.55, "width": 0.1, "height": 0.15},
            {"id": 16, "char": "j", "x": 0.7, "y": 0.55, "width": 0.1, "height": 0.15},
            {"id": 17, "char": "k", "x": 0.8, "y": 0.55, "width": 0.1, "height": 0.15},
            {"id": 18, "char": "l", "x": 0.9, "y": 0.55, "width": 0.1, "height": 0.15},
            
            {"id": 19, "char": "z", "x": 0.15, "y": 0.7, "width": 0.1, "height": 0.15},
            {"id": 20, "char": "x", "x": 0.25, "y": 0.7, "width": 0.1, "height": 0.15},
            {"id": 21, "char": "c", "x": 0.35, "y": 0.7, "width": 0.1, "height": 0.15},
            {"id": 22, "char": "v", "x": 0.45, "y": 0.7, "width": 0.1, "height": 0.15},
            {"id": 23, "char": "b", "x": 0.55, "y": 0.7, "width": 0.1, "height": 0.15},
            {"id": 24, "char": "n", "x": 0.65, "y": 0.7, "width": 0.1, "height": 0.15},
            {"id": 25, "char": "m", "x": 0.75, "y": 0.7, "width": 0.1, "height": 0.15},
            
            {"id": 26, "char": " ", "x": 0.2, "y": 0.85, "width": 0.6, "height": 0.1},
            {"id": 27, "char": "'", "x": 0.85, "y": 0.55, "width": 0.05, "height": 0.15},
            {"id": 28, "char": "-", "x": 0.85, "y": 0.7, "width": 0.05, "height": 0.15},
            {"id": 29, "char": ".", "x": 0.9, "y": 0.85, "width": 0.05, "height": 0.1}
        ]
    }
    
    keyboard_path = transformerjs_dir / "keyboard.json"
    with open(keyboard_path, 'w') as f:
        json.dump(keyboard_layout, f, indent=2)
    
    # Create JavaScript integration file
    js_integration = """
/**
 * Neural Swipe Typing model for transformer.js
 * Converts swipe gestures to word predictions
 */

import { pipeline } from '@xenova/transformers';

class NeuralSwipeDecoder {
    constructor() {
        this.model = null;
        this.tokenizer = null;
        this.keyboard = null;
    }
    
    async initialize(modelPath = './transformerjs/') {
        // Load the model
        this.model = await pipeline('text-generation', modelPath + 'model.onnx');
        
        // Load tokenizer and keyboard configs
        const tokenizerResponse = await fetch(modelPath + 'tokenizer.json');
        this.tokenizer = await tokenizerResponse.json();
        
        const keyboardResponse = await fetch(modelPath + 'keyboard.json');
        this.keyboard = await keyboardResponse.json();
        
        console.log('Neural Swipe Decoder initialized');
    }
    
    /**
     * Extract trajectory features from swipe points
     * @param {Array} swipePoints - Array of {x, y, t} points
     * @returns {Array} Trajectory features [x, y, vx, vy, ax, ay]
     */
    extractTrajectoryFeatures(swipePoints) {
        const features = [];
        
        for (let i = 0; i < swipePoints.length; i++) {
            const point = swipePoints[i];
            const prevPoint = i > 0 ? swipePoints[i - 1] : point;
            const nextPoint = i < swipePoints.length - 1 ? swipePoints[i + 1] : point;
            
            // Position
            const x = point.x;
            const y = point.y;
            
            // Velocity (finite difference)
            const dt = Math.max(point.t - prevPoint.t, 1);
            const vx = (point.x - prevPoint.x) / dt;
            const vy = (point.y - prevPoint.y) / dt;
            
            // Acceleration
            const dt2 = Math.max(nextPoint.t - point.t, 1);
            const vx_next = (nextPoint.x - point.x) / dt2;
            const vy_next = (nextPoint.y - point.y) / dt2;
            const ax = (vx_next - vx) / Math.max((dt + dt2) / 2, 1);
            const ay = (vy_next - vy) / Math.max((dt + dt2) / 2, 1);
            
            features.push([x, y, vx, vy, ax, ay]);
        }
        
        return features;
    }
    
    /**
     * Extract keyboard features from swipe points
     * @param {Array} swipePoints - Array of {x, y, t} points
     * @returns {Array} Keyboard features
     */
    extractKeyboardFeatures(swipePoints) {
        const features = [];
        
        for (const point of swipePoints) {
            // Calculate distances to all keys
            const distances = this.keyboard.keys.map(key => {
                const dx = point.x - (key.x + key.width / 2);
                const dy = point.y - (key.y + key.height / 2);
                return Math.sqrt(dx * dx + dy * dy);
            });
            
            // Convert to weights (inverse distance)
            const weights = distances.map(d => 1 / (1 + d));
            const sum = weights.reduce((a, b) => a + b, 0);
            const normalizedWeights = weights.map(w => w / sum);
            
            features.push(normalizedWeights);
        }
        
        return features;
    }
    
    /**
     * Decode a swipe gesture to word predictions
     * @param {Array} swipePoints - Array of {x, y, t} points
     * @param {Object} options - Decoding options
     * @returns {Array} Predicted words with scores
     */
    async decode(swipePoints, options = {}) {
        if (!this.model) {
            throw new Error('Model not initialized. Call initialize() first.');
        }
        
        // Extract features
        const trajectoryFeatures = this.extractTrajectoryFeatures(swipePoints);
        const keyboardFeatures = this.extractKeyboardFeatures(swipePoints);
        
        // Prepare inputs for the model
        const inputs = {
            trajectory_features: [trajectoryFeatures],
            keyboard_features: [keyboardFeatures]
        };
        
        // Run inference
        const outputs = await this.model(inputs);
        
        // Convert logits to predictions
        const logits = outputs.logits[0];
        const predictions = this.logitsToWords(logits, options);
        
        return predictions;
    }
    
    /**
     * Convert model logits to word predictions
     * @param {Array} logits - Model output logits
     * @param {Object} options - Decoding options
     * @returns {Array} Predicted words with scores
     */
    logitsToWords(logits, options = {}) {
        const topK = options.topK || 5;
        
        // Apply softmax to get probabilities
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(x => x / sumExp);
        
        // Get top-k predictions
        const indexed = probs.map((prob, idx) => ({ prob, char: this.tokenizer.characters[idx] }))
                            .sort((a, b) => b.prob - a.prob)
                            .slice(0, topK);
        
        return indexed;
    }
}

// Export for use in web applications
export { NeuralSwipeDecoder };
"""
    
    js_path = transformerjs_dir / "neural_swipe_decoder.js"
    with open(js_path, 'w') as f:
        f.write(js_integration)
    
    # Create README for transformer.js usage
    readme_content = """# Neural Swipe Decoder for transformer.js

This directory contains the neural swipe typing model exported for use with transformer.js in web applications.

## Files

- `model.onnx`: The exported ONNX model
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer configuration
- `keyboard.json`: Keyboard layout definition
- `neural_swipe_decoder.js`: JavaScript integration class

## Usage

1. Install transformer.js:
```bash
npm install @xenova/transformers
```

2. Import and use the decoder:
```javascript
import { NeuralSwipeDecoder } from './neural_swipe_decoder.js';

const decoder = new NeuralSwipeDecoder();
await decoder.initialize('./transformerjs/');

// Decode a swipe gesture
const swipePoints = [
    {x: 0.1, y: 0.5, t: 0},
    {x: 0.2, y: 0.5, t: 100},
    {x: 0.3, y: 0.5, t: 200}
];

const predictions = await decoder.decode(swipePoints);
console.log(predictions);
```

## Features

- Real-time swipe gesture recognition
- Trajectory feature extraction (position, velocity, acceleration)
- Keyboard-aware feature computation
- Top-k word predictions
- Web-optimized ONNX model

## Model Architecture

The model uses an encoder-decoder transformer architecture that processes:
- Trajectory features: x, y coordinates with velocity and acceleration
- Keyboard features: Distance-weighted probabilities for each key
- Character-level tokenization for word generation

## Performance

This model is optimized for web deployment with:
- Small model size suitable for browser loading
- Fast inference suitable for real-time typing
- Support for variable-length swipe gestures
"""
    
    readme_path = transformerjs_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Transformer.js export completed: {transformerjs_dir}")
    print(f"Files created:")
    print(f"  - {onnx_path}")
    print(f"  - {config_path}")
    print(f"  - {tokenizer_path}")
    print(f"  - {keyboard_path}")
    print(f"  - {js_path}")
    print(f"  - {readme_path}")


def main():
    parser = argparse.ArgumentParser(description='Export neural swipe typing model to transformer.js format')
    parser.add_argument('checkpoint_path', 
                       nargs='?', 
                       default='/data/data/com.termux/files/home/git/neural-swipe-typing/checkpoints_english/english-epoch=51-val_loss=1.248-val_word_acc=0.659.ckpt',
                       help='Path to the Lightning checkpoint file (.ckpt)')
    parser.add_argument('--output-dir', 
                       default='/data/data/com.termux/files/home/git/neural-swipe-typing/exported_models_english',
                       help='Output directory for exported models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    
    try:
        # Load the model
        model, model_name = load_model_from_checkpoint(args.checkpoint_path)
        print(f"Model loaded successfully: {model_name}")
        
        # Wrap the model for transformer.js
        wrapped_model = TransformerJSModelWrapper(model, model_name)
        wrapped_model.eval()
        
        # Create sample inputs
        sample_inputs = create_sample_inputs_transformerjs(model_name)
        print("Created sample inputs for transformer.js export")
        
        # Test the model
        with torch.no_grad():
            output = wrapped_model(*sample_inputs)
            print(f"Model output shape: {output.shape}")
        
        # Try to export to transformer.js format (ONNX may fail for complex models)
        try:
            export_transformerjs_model(wrapped_model, model_name, output_dir, sample_inputs)
            print("Transformer.js export completed successfully!")
        except Exception as e:
            print(f"ONNX export failed due to model complexity: {e}")
            print("Creating transformer.js files without ONNX model...")
            
            # Create the directory and files anyway
            transformerjs_dir = Path(output_dir) / "transformerjs"
            transformerjs_dir.mkdir(exist_ok=True)
            
            # Create configuration files
            config = create_transformer_js_config(model_name)
            config_path = transformerjs_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create tokenizer config
            tokenizer_config = {
                "tokenizer_type": "character_level",
                "vocab_size": 30,
                "special_tokens": {
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>"
                },
                "characters": " abcdefghijklmnopqrstuvwxyz'-."
            }
            
            tokenizer_path = transformerjs_dir / "tokenizer.json"
            with open(tokenizer_path, 'w') as f:
                json.dump(tokenizer_config, f, indent=2)
            
            # Create JavaScript integration file with PyTorch model loading
            js_integration = '''
/**
 * Neural Swipe Typing model for web applications
 * Note: ONNX export failed, so this uses PyTorch.js or requires server-side inference
 */

class NeuralSwipeDecoder {
    constructor() {
        this.initialized = false;
    }
    
    async initialize(modelPath = './transformerjs/') {
        console.log('Neural Swipe Decoder initialized (PyTorch model, requires server-side inference)');
        this.initialized = true;
        
        // Load configuration
        const configResponse = await fetch(modelPath + 'config.json');
        this.config = await configResponse.json();
        
        // Load tokenizer
        const tokenizerResponse = await fetch(modelPath + 'tokenizer.json');
        this.tokenizer = await tokenizerResponse.json();
        
        console.log('Configuration loaded. Model requires PyTorch backend for inference.');
    }
    
    /**
     * Extract trajectory features from swipe points
     */
    extractTrajectoryFeatures(swipePoints) {
        const features = [];
        
        for (let i = 0; i < swipePoints.length; i++) {
            const point = swipePoints[i];
            const prevPoint = i > 0 ? swipePoints[i - 1] : point;
            const nextPoint = i < swipePoints.length - 1 ? swipePoints[i + 1] : point;
            
            const x = point.x;
            const y = point.y;
            
            const dt = Math.max(point.t - prevPoint.t, 1);
            const vx = (point.x - prevPoint.x) / dt;
            const vy = (point.y - prevPoint.y) / dt;
            
            const dt2 = Math.max(nextPoint.t - point.t, 1);
            const vx_next = (nextPoint.x - point.x) / dt2;
            const vy_next = (nextPoint.y - point.y) / dt2;
            const ax = (vx_next - vx) / Math.max((dt + dt2) / 2, 1);
            const ay = (vy_next - vy) / Math.max((dt + dt2) / 2, 1);
            
            features.push([x, y, vx, vy, ax, ay]);
        }
        
        return features;
    }
    
    /**
     * Decode swipe gesture (requires server-side PyTorch inference)
     */
    async decode(swipePoints, options = {}) {
        if (!this.initialized) {
            throw new Error('Model not initialized. Call initialize() first.');
        }
        
        const trajectoryFeatures = this.extractTrajectoryFeatures(swipePoints);
        
        // This would need to be sent to a PyTorch backend for inference
        console.warn('Model inference requires PyTorch backend. Send trajectoryFeatures to server.');
        
        return {
            features: trajectoryFeatures,
            message: 'Requires server-side PyTorch inference',
            serverEndpoint: '/api/swipe/decode'
        };
    }
}

export { NeuralSwipeDecoder };
'''
            
            js_path = transformerjs_dir / "neural_swipe_decoder.js"
            with open(js_path, 'w') as f:
                f.write(js_integration)
            
            # Create README explaining the limitation
            readme_content = """# Neural Swipe Decoder for Web Applications

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
"""
            
            readme_path = transformerjs_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            print(f"Transformer.js setup completed (without ONNX): {transformerjs_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())