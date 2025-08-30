#!/usr/bin/env python3
"""
Export trained mobile models to ONNX and ExecuTorch formats for deployment.
Optimized for Android keyboard apps and web applications.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from mobile_optimized_model import MobileSwipeTypingModel, create_mobile_model
import json
from typing import Tuple, Dict, Any
import os

class MobileModelExporter:
    def __init__(self, model: nn.Module, model_config: Dict[str, Any]):
        self.model = model
        self.config = model_config
        self.model.eval()
        
    def export_to_onnx(self, output_path: str, sample_input: torch.Tensor) -> str:
        """Export model to ONNX format for web deployment"""
        print(f"Exporting to ONNX: {output_path}")
        
        # Create dynamic axes for variable sequence lengths
        dynamic_axes = {
            'trajectory_input': {0: 'batch_size', 1: 'sequence_length'},
            'character_output': {0: 'batch_size', 1: 'word_length'}
        }
        
        # Export model with updated opset for compatibility
        torch.onnx.export(
            self.model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=13,  # Updated to support required operators
            do_constant_folding=True,
            input_names=['trajectory_input'],
            output_names=['character_output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Validate exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"ONNX export successful. Output shape: {ort_outputs[0].shape}")
        return output_path
    
    def export_to_executorch(self, onnx_path: str, output_path: str) -> str:
        """Export ONNX model to ExecuTorch format for Android deployment"""
        try:
            # Import ExecuTorch (if available)
            from executorch.exir import ExecutorchProgram, to_edge
            from torch.export import export
            from executorch.exir.backend.backend_api import to_backend
            
            print(f"Exporting to ExecuTorch: {output_path}")
            
            # Create sample input for tracing
            sample_input = torch.randn(1, 100, 6)
            
            # Export using torch.export
            exported_program = export(self.model, (sample_input,))
            
            # Convert to Edge format
            edge_program = to_edge(exported_program)
            
            # Convert to ExecuTorch
            executorch_program = edge_program.to_executorch()
            
            # Save to file
            with open(output_path, "wb") as f:
                f.write(executorch_program.buffer)
            
            print(f"ExecuTorch export successful: {output_path}")
            return output_path
            
        except ImportError:
            print("ExecuTorch not available. Creating placeholder export script.")
            
            # Create shell script for ExecuTorch export when tools are available
            script_content = f"""#!/bin/bash
# ExecuTorch export script for Android deployment
# Requires ExecuTorch installation

echo "Exporting model to ExecuTorch format..."

python -c "
import torch
from mobile_optimized_model import create_mobile_model

# Load trained model
model = create_mobile_model()
model.load_state_dict(torch.load('mobile_swipe_model.pth', map_location='cpu'))
model.eval()

# Export to ExecuTorch
from executorch.exir import to_edge
from torch.export import export

sample_input = torch.randn(1, 100, 6)
exported_program = export(model, (sample_input,))
edge_program = to_edge(exported_program)
executorch_program = edge_program.to_executorch()

with open('{output_path}', 'wb') as f:
    f.write(executorch_program.buffer)

print('ExecuTorch export completed: {output_path}')
"
"""
            
            script_path = output_path.replace('.pte', '_export.sh')
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            os.chmod(script_path, 0o755)
            print(f"ExecuTorch export script created: {script_path}")
            return script_path
    
    def optimize_for_mobile(self, model_path: str) -> str:
        """Apply mobile-specific optimizations"""
        print("Applying mobile optimizations...")
        
        # Load and optimize model
        model = torch.jit.load(model_path) if model_path.endswith('.pt') else self.model
        
        # Apply torch.jit optimizations
        if hasattr(torch.jit, 'optimize_for_inference'):
            model = torch.jit.optimize_for_inference(model)
        
        # Apply quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # Save optimized model
        optimized_path = model_path.replace('.pth', '_mobile_optimized.pth')
        torch.save(quantized_model.state_dict(), optimized_path)
        
        print(f"Mobile optimized model saved: {optimized_path}")
        return optimized_path
    
    def create_deployment_package(self, output_dir: str) -> Dict[str, str]:
        """Create complete deployment package for mobile apps"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample input for export
        sample_input = torch.randn(1, 100, 6)  # batch=1, seq_len=100, features=6
        
        package_files = {}
        
        # Export to ONNX for web deployment
        onnx_path = os.path.join(output_dir, 'swipe_model_web.onnx')
        package_files['onnx_web'] = self.export_to_onnx(onnx_path, sample_input)
        
        # Export to ExecuTorch for Android
        executorch_path = os.path.join(output_dir, 'swipe_model_android.pte')
        package_files['executorch_android'] = self.export_to_executorch(onnx_path, executorch_path)
        
        # Create vocabulary file
        vocab_path = os.path.join(output_dir, 'vocabulary.json')
        self.create_mobile_vocabulary(vocab_path)
        package_files['vocabulary'] = vocab_path
        
        # Create deployment metadata
        metadata_path = os.path.join(output_dir, 'deployment_metadata.json')
        metadata = {
            'model_architecture': 'mobile_swipe_typing',
            'input_shape': [1, 'variable', 6],
            'output_shape': [1, 'variable', 28],
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
                'model_size_mb': 0.7,  # Estimated
                'avg_inference_ms': 30,  # Estimated
                'parameters': 171548
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        package_files['metadata'] = metadata_path
        
        # Create integration examples
        self.create_integration_examples(output_dir)
        package_files['examples'] = os.path.join(output_dir, 'integration_examples')
        
        print(f"Deployment package created in: {output_dir}")
        return package_files
    
    def create_mobile_vocabulary(self, output_path: str):
        """Create mobile-optimized vocabulary subset"""
        # Load full vocabulary
        full_vocab_path = './data/data_preprocessed/voc.txt'
        
        if os.path.exists(full_vocab_path):
            with open(full_vocab_path, 'r') as f:
                full_vocab = [line.strip() for line in f.readlines()]
            
            # Select top 3000 most common words for mobile
            mobile_vocab = full_vocab[:3000]
        else:
            # Fallback vocabulary
            mobile_vocab = ['the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they']
        
        # Save mobile vocabulary
        vocab_data = {
            'words': mobile_vocab,
            'size': len(mobile_vocab),
            'char_to_idx': {chr(ord('a') + i): i + 4 for i in range(26)},  # a-z mapped to 4-29
            'special_tokens': {'<pad>': 0, '<eos>': 1, '<unk>': 2, '<sos>': 3}
        }
        
        with open(output_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Mobile vocabulary saved: {output_path} ({len(mobile_vocab)} words)")
    
    def create_integration_examples(self, output_dir: str):
        """Create integration example code for different platforms"""
        examples_dir = os.path.join(output_dir, 'integration_examples')
        os.makedirs(examples_dir, exist_ok=True)
        
        # Android Kotlin example
        android_example = '''
// Android Kotlin integration example
class SwipeTypingPredictor {
    private lateinit var module: Module
    
    fun loadModel(modelPath: String) {
        module = Module.load(AssetFilePath(context, "swipe_model_android.pte"))
    }
    
    fun predictWord(swipePoints: List<SwipePoint>): String {
        // Extract features: x, y, vx, vy, ax, ay
        val features = extractFeatures(swipePoints)
        val inputTensor = Tensor.fromBlob(features, longArrayOf(1, features.size.toLong() / 6, 6))
        
        // Run inference
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        
        // Decode output to word
        return decodeOutput(outputTensor)
    }
    
    private fun extractFeatures(points: List<SwipePoint>): FloatArray {
        // Implementation for feature extraction
        // Returns [x1,y1,vx1,vy1,ax1,ay1, x2,y2,vx2,vy2,ax2,ay2, ...]
    }
}
'''
        
        # Web TypeScript example
        web_example = '''
// Web TypeScript integration example
import * as ort from 'onnxruntime-web';

class WebSwipePredictor {
    private session: ort.InferenceSession | null = null;
    
    async loadModel(modelUrl: string) {
        this.session = await ort.InferenceSession.create(modelUrl);
    }
    
    async predictWord(swipePoints: SwipePoint[]): Promise<string> {
        if (!this.session) throw new Error('Model not loaded');
        
        // Extract features
        const features = this.extractFeatures(swipePoints);
        const inputTensor = new ort.Tensor('float32', features, [1, features.length / 6, 6]);
        
        // Run inference
        const feeds = { trajectory_input: inputTensor };
        const results = await this.session.run(feeds);
        
        // Decode output
        return this.decodeOutput(results.character_output);
    }
    
    private extractFeatures(points: SwipePoint[]): Float32Array {
        // Implementation for feature extraction
        // Returns normalized coordinates, velocities, accelerations
    }
    
    private decodeOutput(tensor: ort.Tensor): string {
        // Convert character probabilities to word
    }
}
'''
        
        # Save examples
        with open(os.path.join(examples_dir, 'android_integration.kt'), 'w') as f:
            f.write(android_example)
        
        with open(os.path.join(examples_dir, 'web_integration.ts'), 'w') as f:
            f.write(web_example)
        
        print(f"Integration examples created in: {examples_dir}")

def main():
    """Main export function"""
    print("=== Mobile Model Export Pipeline ===")
    
    # Create mobile model
    model = create_mobile_model()
    print(f"Created mobile model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load configuration
    config_path = 'configs/config_mobile_optimized.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create exporter
    exporter = MobileModelExporter(model, config)
    
    # Create deployment package
    output_dir = 'mobile_deployment_package'
    package_files = exporter.create_deployment_package(output_dir)
    
    print("\n=== Export Summary ===")
    for key, path in package_files.items():
        print(f"{key}: {path}")
    
    print("\nMobile deployment package ready!")
    print("Next steps:")
    print("1. Train the model using the cleaned dataset")
    print("2. Replace model weights in deployment package")
    print("3. Test on target mobile devices")
    print("4. Integrate into Android/Web applications")

if __name__ == "__main__":
    main()