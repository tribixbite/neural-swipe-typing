#!/usr/bin/env python3
"""
Export mobile model to ONNX only (for now) and create deployment package
"""

import torch
import pytorch_lightning as pl
from mobile_optimized_model import create_mobile_model
from train_mobile_model import MobileSwipeTrainer
import json
import os
import onnx
import onnxruntime as ort

def main():
    print("=== Mobile Model ONNX Export ===")
    
    # Load best checkpoint
    checkpoint_path = "checkpoints/mobile_model/mobile-swipe-epoch=06-val_acc=1.000-v1.ckpt"
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the Lightning model from checkpoint
    lightning_model = MobileSwipeTrainer.load_from_checkpoint(checkpoint_path, map_location='cpu')
    lightning_model.eval()
    
    # Extract the PyTorch model and move to CPU for export
    pytorch_model = lightning_model.model
    pytorch_model.cpu()
    pytorch_model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in pytorch_model.parameters()):,} parameters")
    
    # Create output directory
    output_dir = 'mobile_deployment_package'
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'swipe_model_web.onnx')
    sample_input = torch.randn(1, 100, 6)  # batch=1, seq_len=100, features=6
    
    print(f"Exporting to ONNX: {onnx_path}")
    
    # Create dynamic axes for variable sequence lengths
    dynamic_axes = {
        'trajectory_input': {0: 'batch_size', 1: 'sequence_length'},
        'character_output': {0: 'batch_size', 1: 'word_length'}
    }
    
    # Export model with updated opset for compatibility
    torch.onnx.export(
        pytorch_model,
        sample_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['trajectory_input'],
        output_names=['character_output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    # Validate exported model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Test inference
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX export successful. Output shape: {ort_outputs[0].shape}")
    
    # Create deployment metadata
    metadata_path = os.path.join(output_dir, 'deployment_metadata.json')
    metadata = {
        'model_architecture': 'mobile_swipe_typing',
        'input_shape': [1, 'variable', 6],
        'output_shape': [1, 'variable', 30],
        'max_sequence_length': 150,
        'feature_extraction': {
            'coordinates': True,
            'velocities': True,
            'accelerations': True,
            'normalization': [360, 215]
        },
        'inference': {
            'mode': 'greedy',
            'max_steps': 20
        },
        'performance': {
            'model_size_mb': os.path.getsize(onnx_path) / 1024 / 1024,
            'parameters': sum(p.numel() for p in pytorch_model.parameters()),
            'avg_inference_ms': 50  # Estimated
        },
        'vocabulary': {
            'size': 30,
            'special_tokens': {'<pad>': 0, '<eos>': 1, '<unk>': 2, '<sos>': 3},
            'characters': {chr(ord('a') + i): i + 4 for i in range(26)}
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create vocabulary file
    vocab_path = os.path.join(output_dir, 'vocabulary.json')
    vocab_data = {
        'char_to_idx': {'<pad>': 0, '<eos>': 1, '<unk>': 2, '<sos>': 3},
        'size': 30
    }
    vocab_data['char_to_idx'].update({chr(ord('a') + i): i + 4 for i in range(26)})
    vocab_data['idx_to_char'] = {v: k for k, v in vocab_data['char_to_idx'].items()}
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    # Create integration examples
    examples_dir = os.path.join(output_dir, 'integration_examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    # Web TypeScript example
    web_example = '''// Web TypeScript integration example
import * as ort from 'onnxruntime-web';

class MobileSwipePredictor {
    private session: ort.InferenceSession | null = null;
    private charToIdx = {
        '<pad>': 0, '<eos>': 1, '<unk>': 2, '<sos>': 3,
        'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10,
        'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17,
        'o': 18, 'p': 19, 'q': 20, 'r': 21, 's': 22, 't': 23, 'u': 24,
        'v': 25, 'w': 26, 'x': 27, 'y': 28, 'z': 29
    };
    private idxToChar = Object.fromEntries(
        Object.entries(this.charToIdx).map(([k, v]) => [v, k])
    );
    
    async loadModel(modelUrl: string) {
        this.session = await ort.InferenceSession.create(modelUrl);
        console.log('Mobile swipe model loaded');
    }
    
    async predictWord(swipePoints: {x: number, y: number, t: number}[]): Promise<string> {
        if (!this.session) throw new Error('Model not loaded');
        
        // Extract 6D features: x, y, vx, vy, ax, ay
        const features = this.extractFeatures(swipePoints);
        const inputTensor = new ort.Tensor('float32', features, [1, features.length / 6, 6]);
        
        // Run inference
        const feeds = { trajectory_input: inputTensor };
        const results = await this.session.run(feeds);
        
        // Decode character probabilities to word
        return this.decodeOutput(results.character_output);
    }
    
    private extractFeatures(points: {x: number, y: number, t: number}[]): Float32Array {
        if (points.length < 2) return new Float32Array(0);
        
        const features = [];
        
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            
            // Normalize coordinates to [0, 1]
            const x = p.x / 360;
            const y = p.y / 215;
            
            // Calculate velocities
            let vx = 0, vy = 0;
            if (i > 0) {
                const prev = points[i - 1];
                const dt = Math.max(p.t - prev.t, 1); // Avoid division by zero
                vx = (p.x - prev.x) / dt;
                vy = (p.y - prev.y) / dt;
            }
            
            // Calculate accelerations
            let ax = 0, ay = 0;
            if (i > 1) {
                const prev = points[i - 1];
                const prev2 = points[i - 2];
                const dt1 = Math.max(p.t - prev.t, 1);
                const dt2 = Math.max(prev.t - prev2.t, 1);
                const vx_prev = (prev.x - prev2.x) / dt2;
                const vy_prev = (prev.y - prev2.y) / dt2;
                ax = (vx - vx_prev) / dt1;
                ay = (vy - vy_prev) / dt1;
            }
            
            // Clip velocities and accelerations
            vx = Math.max(-1000, Math.min(1000, vx));
            vy = Math.max(-1000, Math.min(1000, vy));
            ax = Math.max(-500, Math.min(500, ax));
            ay = Math.max(-500, Math.min(500, ay));
            
            features.push(x, y, vx / 1000, vy / 1000, ax / 500, ay / 500);
        }
        
        return new Float32Array(features);
    }
    
    private decodeOutput(tensor: ort.Tensor): string {
        const data = tensor.data as Float32Array;
        const [batchSize, seqLen, vocabSize] = tensor.dims;
        
        let word = '';
        for (let i = 0; i < seqLen; i++) {
            let maxIdx = 0;
            let maxVal = -Infinity;
            
            for (let j = 0; j < vocabSize; j++) {
                const val = data[i * vocabSize + j];
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = j;
                }
            }
            
            const char = this.idxToChar[maxIdx];
            if (char === '<eos>') break;
            if (char && char !== '<pad>' && char !== '<sos>') {
                word += char;
            }
        }
        
        return word;
    }
}

// Usage example
async function main() {
    const predictor = new MobileSwipePredictor();
    await predictor.loadModel('swipe_model_web.onnx');
    
    // Example swipe points for "hello"
    const swipePoints = [
        {x: 96, y: 167, t: 0},    // h
        {x: 124, y: 167, t: 50},  // e
        {x: 152, y: 167, t: 100}, // l
        {x: 152, y: 167, t: 150}, // l
        {x: 208, y: 167, t: 200}  // o
    ];
    
    const prediction = await predictor.predictWord(swipePoints);
    console.log('Predicted word:', prediction);
}
'''
    
    with open(os.path.join(examples_dir, 'web_integration.ts'), 'w') as f:
        f.write(web_example)
    
    print("\n=== Export Summary ===")
    print(f"ONNX model: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")
    print(f"Metadata: {metadata_path}")
    print(f"Vocabulary: {vocab_path}")
    print(f"Integration examples: {examples_dir}/")
    
    print(f"\nModel parameters: {metadata['performance']['parameters']:,}")
    print(f"Model size: {metadata['performance']['model_size_mb']:.2f} MB")
    
    print("\n✅ ONNX export completed successfully!")
    print("🔧 ExecuTorch export skipped (requires model modifications for control flow)")
    print("\nNext steps:")
    print("1. Test ONNX model in web browser with ONNX.js")
    print("2. Integrate with virtual keyboard interface")
    print("3. For Android: modify model to remove dynamic control flow for ExecuTorch")

if __name__ == "__main__":
    main()