#!/usr/bin/env python3
"""
Convert neural swipe typing model to transformer.js format using Hugging Face Optimum.
This creates ONNX models compatible with the @huggingface/transformers library.
"""

import argparse
import json
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import HfApi, ModelCard, ModelCardData

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import MODEL_GETTERS_DICT, EncoderDecoderTransformerLike


class HuggingFaceModelWrapper(nn.Module):
    """
    Wrapper to make the neural swipe model compatible with Hugging Face transformers.
    This creates a structure that can be converted to ONNX via Optimum.
    """
    
    def __init__(self, swipe_model: EncoderDecoderTransformerLike, model_name: str, vocab_size: int = 30):
        super().__init__()
        self.swipe_model = swipe_model
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.config = self._create_config()
    
    def _create_config(self) -> Dict[str, Any]:
        """Create a configuration dict compatible with HuggingFace."""
        return {
            "model_type": "neural_swipe_decoder",
            "architectures": ["NeuralSwipeModel"],
            "vocab_size": self.vocab_size,
            "d_model": 128,
            "max_encoder_seq_len": 299,
            "max_decoder_seq_len": 35,
            "is_encoder_decoder": True,
            "model_name": self.model_name,
            "task_specific_params": {
                "swipe_decoding": {
                    "temperature": 1.0,
                    "top_k": 10,
                    "top_p": 0.9,
                    "max_length": 35
                }
            }
        }
    
    def forward(self, 
                trajectory_features: torch.Tensor,
                keyboard_features: torch.Tensor,
                decoder_input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass compatible with HuggingFace format.
        
        Args:
            trajectory_features: [batch_size, seq_len, 6]
            keyboard_features: [batch_size, seq_len, n_features] 
            decoder_input_ids: [batch_size, decoder_seq_len]
            
        Returns:
            logits: [batch_size, decoder_seq_len, vocab_size]
        """
        batch_size, seq_len = trajectory_features.shape[:2]
        
        # Transpose to model's expected format [seq_len, batch_size, features]
        traj_feats = trajectory_features.transpose(0, 1)
        kb_features = keyboard_features.transpose(0, 1)
        
        # Handle decoder input
        if decoder_input_ids is None:
            decoder_seq_len = 1
            decoder_input = torch.ones(decoder_seq_len, batch_size, dtype=torch.long)
        else:
            decoder_seq_len = decoder_input_ids.shape[1]
            decoder_input = decoder_input_ids.transpose(0, 1)  # [decoder_seq_len, batch_size]
        
        # Create padding masks
        encoder_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        decoder_padding_mask = torch.zeros(batch_size, decoder_seq_len, dtype=torch.bool)
        
        # Prepare encoder input based on model type
        if "nearest" in self.model_name:
            encoder_input = (traj_feats, kb_features.long())
        elif "weighted" in self.model_name:
            encoder_input = (traj_feats, kb_features)
        else:
            encoder_input = torch.cat([traj_feats, kb_features], dim=-1)
        
        # Run the model
        logits = self.swipe_model.forward(
            encoder_input,
            decoder_input,
            encoder_padding_mask,
            decoder_padding_mask
        )
        
        # Transpose back to HuggingFace format [batch_size, seq_len, vocab_size]
        return logits.transpose(0, 1)


def load_model_from_checkpoint(checkpoint_path: str) -> Tuple[EncoderDecoderTransformerLike, str]:
    """Load model from PyTorch Lightning checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_name = checkpoint.get('hyper_parameters', {}).get('model_name', 'v3_weighted_and_traj_transformer_bigger')
    
    print(f"Model architecture: {model_name}")
    
    if model_name not in MODEL_GETTERS_DICT:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_GETTERS_DICT.keys())}")
    
    # Create model on CPU
    model_getter = MODEL_GETTERS_DICT[model_name]
    model = model_getter(device='cpu')
    
    # Load state dict
    core_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            core_state_dict[key[6:]] = value
    
    model.load_state_dict(core_state_dict)
    model.eval()
    
    return model, model_name


def create_transformers_config(model_name: str, vocab_size: int = 30) -> Dict[str, Any]:
    """Create config.json for transformers.js."""
    return {
        "model_type": "neural_swipe_decoder",
        "architectures": ["NeuralSwipeModel"],
        "vocab_size": vocab_size,
        "d_model": 128,
        "max_encoder_seq_len": 299,
        "max_decoder_seq_len": 35,
        "is_encoder_decoder": True,
        "task_specific_params": {
            "swipe_decoding": {
                "temperature": 1.0,
                "top_k": 10,
                "top_p": 0.9,
                "max_length": 35
            }
        },
        "transformers_version": "4.36.0",
        "use_cache": True,
        "torch_dtype": "float32"
    }


def create_tokenizer_config(vocab_size: int = 30) -> Dict[str, Any]:
    """Create tokenizer.json for transformers.js."""
    # Character vocabulary for swipe typing
    vocab = [
        "<pad>", "<s>", "</s>", "<unk>",  # Special tokens
        " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "-", "."
    ]
    
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<pad>", "special": True},
            {"id": 1, "content": "<s>", "special": True},  
            {"id": 2, "content": "</s>", "special": True},
            {"id": 3, "content": "<unk>", "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "CharDelimiterSplit",
            "delimiter": ""
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": "<s> $A </s>",
            "pair": "<s> $A </s> $B </s>",
            "special_tokens": {
                "<s>": {"id": 1, "type_id": 0},
                "</s>": {"id": 2, "type_id": 0}
            }
        },
        "decoder": {
            "type": "CharDecoder"
        },
        "model": {
            "type": "WordPiece",
            "vocab": {char: i for i, char in enumerate(vocab)},
            "unk_token": "<unk>",
            "continuing_subword_prefix": "",
            "max_input_chars_per_word": 50
        }
    }


def create_tokenizer_config_json() -> Dict[str, Any]:
    """Create tokenizer_config.json for transformers.js."""
    return {
        "tokenizer_class": "PreTrainedTokenizer",
        "model_max_length": 35,
        "padding_side": "right",
        "truncation_side": "right",
        "special_tokens": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>"
        },
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "clean_up_tokenization_spaces": True
    }


def create_sample_inputs(model_name: str, batch_size: int = 1) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Create sample inputs for ONNX conversion."""
    seq_len = 50
    
    # Trajectory features
    trajectory_features = torch.randn(batch_size, seq_len, 6)
    
    # Keyboard features based on model type
    if "weighted" in model_name:
        keyboard_features = torch.softmax(torch.randn(batch_size, seq_len, 30), dim=-1)
    elif "nearest" in model_name:
        keyboard_features = torch.randint(0, 30, (batch_size, seq_len)).float()
    else:
        keyboard_features = torch.randn(batch_size, seq_len, 30)
    
    # Decoder input (optional for generation)
    decoder_input_ids = torch.ones(batch_size, 1, dtype=torch.long)  # Start with <s> token
    
    inputs = {
        "trajectory_features": trajectory_features,
        "keyboard_features": keyboard_features, 
        "decoder_input_ids": decoder_input_ids
    }
    
    input_names = ["trajectory_features", "keyboard_features", "decoder_input_ids"]
    dynamic_axes = {
        "trajectory_features": {0: "batch_size", 1: "sequence_length"},
        "keyboard_features": {0: "batch_size", 1: "sequence_length"},
        "decoder_input_ids": {0: "batch_size", 1: "decoder_length"},
        "logits": {0: "batch_size", 1: "decoder_length"}
    }
    
    return inputs, {"input_names": input_names, "dynamic_axes": dynamic_axes}


def save_as_huggingface_model(wrapped_model: nn.Module, 
                             model_name: str,
                             output_dir: Path,
                             sample_inputs: Dict[str, torch.Tensor]):
    """Save model in HuggingFace format."""
    
    # Create model directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = output_dir / "pytorch_model.bin"
    torch.save(wrapped_model.state_dict(), model_path)
    
    # Save config
    config_path = output_dir / "config.json"
    config = create_transformers_config(model_name)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer_config = create_tokenizer_config()
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Save tokenizer config
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config_json = create_tokenizer_config_json()
    with open(tokenizer_config_path, 'w') as f:
        json.dump(tokenizer_config_json, f, indent=2)
    
    print(f"HuggingFace model saved to: {output_dir}")


def convert_to_onnx_with_optimum(model_dir: Path, onnx_dir: Path, sample_inputs: Dict[str, torch.Tensor]):
    """Convert to ONNX using Hugging Face Optimum."""
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from optimum.exporters.onnx import export
        from transformers import AutoConfig, PreTrainedModel
        
        print("Converting to ONNX using Optimum...")
        
        # Create ONNX directory
        onnx_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to ONNX (this may fail for complex models)
        try:
            export(
                model=str(model_dir),
                output=str(onnx_dir),
                task="seq2seq-lm",  # Closest task type
                opset=14
            )
            print(f"ONNX model exported to: {onnx_dir}")
            
        except Exception as e:
            print(f"Optimum export failed: {e}")
            print("Falling back to manual ONNX export...")
            
            # Manual ONNX export as fallback
            from pathlib import Path
            import torch
            
            # Load the saved model
            model_path = model_dir / "pytorch_model.bin"
            config_path = model_dir / "config.json"
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # This is a simplified export - may need adjustment for complex models
            print("Manual ONNX export is complex for custom architectures.")
            print("Consider using the TensorRT or OpenVINO backends instead.")
            return False
            
    except ImportError:
        print("Optimum not installed. Please install: pip install optimum[onnxruntime]")
        return False
    
    return True


def create_web_demo(output_dir: Path, model_name: str):
    """Create a web demo for the transformer.js model."""
    
    demo_dir = output_dir / "web_demo"
    demo_dir.mkdir(exist_ok=True)
    
    # HTML demo
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Swipe Typing Demo</title>
    <script type="module">
        import {{ pipeline }} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.0';
        
        class SwipeDecoder {{
            constructor() {{
                this.pipeline = null;
                this.ready = false;
            }}
            
            async initialize() {{
                try {{
                    // Load the model from the onnx directory
                    this.pipeline = await pipeline('text-generation', './onnx/', {{
                        dtype: 'fp32'
                    }});
                    this.ready = true;
                    console.log('Model loaded successfully');
                    document.getElementById('status').textContent = 'Ready';
                }} catch (error) {{
                    console.error('Failed to load model:', error);
                    document.getElementById('status').textContent = 'Error loading model';
                }}
            }}
            
            extractTrajectoryFeatures(swipePoints) {{
                const features = [];
                
                for (let i = 0; i < swipePoints.length; i++) {{
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
                }}
                
                return features;
            }}
            
            async decodeSwipe(swipePoints) {{
                if (!this.ready) {{
                    throw new Error('Model not ready');
                }}
                
                const trajectoryFeatures = this.extractTrajectoryFeatures(swipePoints);
                // Note: This is a simplified example - actual implementation would
                // need to properly format inputs for the custom model
                
                console.log('Trajectory features:', trajectoryFeatures);
                return {{ message: 'Demo mode - custom model integration needed' }};
            }}
        }}
        
        // Initialize when page loads
        window.addEventListener('DOMContentLoaded', async () => {{
            const decoder = new SwipeDecoder();
            await decoder.initialize();
            
            // Demo swipe points
            const demoSwipe = [
                {{ x: 0.1, y: 0.5, t: 0 }},
                {{ x: 0.2, y: 0.5, t: 100 }},
                {{ x: 0.3, y: 0.5, t: 200 }},
                {{ x: 0.4, y: 0.5, t: 300 }}
            ];
            
            document.getElementById('demo-btn').onclick = async () => {{
                try {{
                    const result = await decoder.decodeSwipe(demoSwipe);
                    document.getElementById('result').textContent = JSON.stringify(result, null, 2);
                }} catch (error) {{
                    document.getElementById('result').textContent = 'Error: ' + error.message;
                }}
            }};
        }});
    </script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .status {{
            font-weight: bold;
            margin: 10px 0;
        }}
        button {{
            padding: 10px 20px;
            font-size: 16px;
            margin: 10px 0;
        }}
        #result {{
            background: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            white-space: pre-wrap;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <h1>Neural Swipe Typing Demo</h1>
    <div class="status">Status: <span id="status">Loading...</span></div>
    
    <p>This demo shows how to integrate the neural swipe typing model with transformer.js.</p>
    
    <button id="demo-btn">Test Demo Swipe</button>
    
    <div id="result"></div>
    
    <h2>Notes</h2>
    <ul>
        <li>The model architecture is: {model_name}</li>
        <li>This requires the ONNX model to be successfully converted</li>
        <li>Custom preprocessing may be needed for trajectory and keyboard features</li>
        <li>The actual implementation would integrate with a virtual keyboard interface</li>
    </ul>
</body>
</html>'''
    
    html_path = demo_dir / "index.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    # README for the demo
    readme_content = f'''# Neural Swipe Typing Web Demo

This demo shows how to use the converted neural swipe typing model with transformer.js.

## Setup

1. Ensure the ONNX model was successfully converted
2. Serve this directory with a web server (due to CORS restrictions)
3. Open `index.html` in a browser

## Usage

```bash
# Serve with Python
python -m http.server 8000

# Or with Node.js
npx http-server .

# Then open http://localhost:8000
```

## Model Details

- Architecture: {model_name}
- Input: Trajectory features (x, y, velocity, acceleration) + keyboard features
- Output: Character probabilities for word prediction

## Integration Notes

The actual integration would require:
1. Virtual keyboard component with touch tracking
2. Real-time trajectory feature extraction
3. Keyboard-aware distance calculations
4. Beam search for word generation
'''
    
    readme_path = demo_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Web demo created: {demo_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert neural swipe model to transformer.js format')
    parser.add_argument('checkpoint_path', 
                       help='Path to PyTorch Lightning checkpoint')
    parser.add_argument('--output-dir', 
                       default='./transformers_model',
                       help='Output directory for converted model')
    parser.add_argument('--model-name',
                       help='Override model name from checkpoint')
    parser.add_argument('--create-demo', 
                       action='store_true',
                       help='Create web demo')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the original model
        print("Step 1: Loading PyTorch model...")
        original_model, model_name = load_model_from_checkpoint(args.checkpoint_path)
        if args.model_name:
            model_name = args.model_name
        
        # Wrap for HuggingFace compatibility
        print("Step 2: Creating HuggingFace-compatible wrapper...")
        wrapped_model = HuggingFaceModelWrapper(original_model, model_name)
        wrapped_model.eval()
        
        # Create sample inputs
        print("Step 3: Preparing sample inputs...")
        sample_inputs, onnx_config = create_sample_inputs(model_name)
        
        # Test the wrapped model
        print("Step 4: Testing wrapped model...")
        with torch.no_grad():
            output = wrapped_model(**sample_inputs)
            print(f"Model output shape: {output.shape}")
        
        # Save as HuggingFace model
        print("Step 5: Saving as HuggingFace model...")
        hf_model_dir = output_dir / "huggingface_model"
        save_as_huggingface_model(wrapped_model, model_name, hf_model_dir, sample_inputs)
        
        # Convert to ONNX
        print("Step 6: Converting to ONNX...")
        onnx_dir = output_dir / "onnx"
        success = convert_to_onnx_with_optimum(hf_model_dir, onnx_dir, sample_inputs)
        
        if not success:
            print("ONNX conversion failed. Model may be too complex for automatic conversion.")
            print("Consider:")
            print("1. Simplifying the model architecture")
            print("2. Using custom ONNX export logic")
            print("3. Using a different backend (TensorRT, OpenVINO)")
        
        # Create web demo
        if args.create_demo:
            print("Step 7: Creating web demo...")
            create_web_demo(output_dir, model_name)
        
        print("\\nConversion completed!")
        print(f"Output directory: {output_dir}")
        print(f"HuggingFace model: {hf_model_dir}")
        print(f"ONNX model: {onnx_dir}")
        
        if args.create_demo:
            print(f"Web demo: {output_dir / 'web_demo'}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())