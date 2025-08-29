#!/usr/bin/env python3
"""
TorchScript export for Android/Java interface.
More robust than ONNX for complex transformer models.
"""

import torch
import torch.nn as nn
import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import MODEL_GETTERS_DICT


class SwipeTypingModelWrapper(nn.Module):
    """
    Complete wrapper for the swipe typing model that handles:
    - Input preprocessing (combining trajectory and keyboard features)
    - Encoder-decoder architecture
    - Autoregressive decoding
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.vocab_size = 30
        self.sos_token = 29
        self.eos_token = 26
        self.pad_token = 28
        self.max_length = 35
        
    @torch.jit.export
    def encode_swipe(self, swipe_features: torch.Tensor) -> torch.Tensor:
        """
        Encode a swipe gesture into memory representation.
        
        Args:
            swipe_features: [batch_size, seq_len, 7] where last dim is:
                          [x, y, vx, vy, ax, ay, kb_id]
        Returns:
            memory: [seq_len, batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = swipe_features.shape
        
        # Split features
        traj_feats = swipe_features[:, :, :6].transpose(0, 1)  # [seq_len, batch, 6]
        kb_ids = swipe_features[:, :, 6].long().clamp(0, 29).transpose(0, 1)  # [seq_len, batch]
        
        # Create encoder input
        encoder_input = (traj_feats, kb_ids)
        
        # Encode
        x_embedded = self.model.enc_in_emb_model(encoder_input)
        memory = self.model.encoder(x_embedded, src_key_padding_mask=None)
        
        return memory
    
    @torch.jit.export
    def decode_step(self, 
                   tgt_sequence: torch.Tensor,
                   memory: torch.Tensor) -> torch.Tensor:
        """
        Decode one step given current sequence and encoder memory.
        
        Args:
            tgt_sequence: [batch_size, current_length] - tokens generated so far
            memory: [src_len, batch_size, hidden_dim] - encoder output
        Returns:
            next_token_logits: [batch_size, vocab_size] - logits for next token
        """
        # Transpose target to [seq_len, batch]
        tgt = tgt_sequence.transpose(0, 1)
        
        # Embed and decode
        tgt_embedded = self.model.dec_in_emb_model(tgt)
        
        # Create causal mask
        tgt_len = tgt.shape[0]
        tgt_mask = torch.triu(
            torch.ones(tgt_len, tgt_len) * float('-inf'),
            diagonal=1
        ).to(tgt.device)
        
        # Decode
        decoded = self.model.decoder(
            tgt_embedded, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=None,
            tgt_key_padding_mask=None
        )
        
        # Get logits for last position
        last_logits = self.model.out(decoded[-1])  # [batch_size, vocab_size]
        
        return last_logits
    
    @torch.jit.export
    def generate_word(self, 
                     swipe_features: torch.Tensor,
                     max_length: int = 35) -> torch.Tensor:
        """
        Generate complete word from swipe features.
        
        Args:
            swipe_features: [batch_size, seq_len, 7]
            max_length: Maximum word length
        Returns:
            generated_ids: [batch_size, generated_length]
        """
        batch_size = swipe_features.shape[0]
        device = swipe_features.device
        
        # Encode swipe once
        memory = self.encode_swipe(swipe_features)
        
        # Initialize with SOS token
        generated = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Decode next token
            logits = self.decode_step(generated, memory)
            
            # Greedy decoding (can be replaced with beam search)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have produced EOS
            if (next_token == self.eos_token).all():
                break
        
        return generated
    
    def forward(self, swipe_features: torch.Tensor) -> torch.Tensor:
        """
        Main forward pass - generates word from swipe.
        
        Args:
            swipe_features: [batch_size, seq_len, 7]
        Returns:
            generated_ids: [batch_size, generated_length]
        """
        return self.generate_word(swipe_features)


def load_model(checkpoint_path, model_name="v3_nearest_and_traj_transformer_bigger"):
    """Load model from checkpoint."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model
    model = MODEL_GETTERS_DICT[model_name]()
    
    # Fix state dict keys
    state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.', '') if key.startswith('model.') else key
        new_key = new_key.replace('enc_in_emb_key_emb.', 'enc_in_emb_model.key_emb.')
        new_key = new_key.replace('dec_in_emb_0.', 'dec_in_emb_model.0.')
        new_key = new_key.replace('dec_in_emb_2.', 'dec_in_emb_model.2.')
        state_dict[new_key] = value
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model


def export_torchscript(model, output_dir):
    """Export model to TorchScript format."""
    
    # Create wrapper
    wrapped_model = SwipeTypingModelWrapper(model)
    wrapped_model.eval()
    
    # Create example input
    batch_size = 1
    seq_len = 100
    example_input = torch.randn(batch_size, seq_len, 7)
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapped_model, example_input)
    
    # Script the model for better support of control flow
    print("Scripting model...")
    scripted_model = torch.jit.script(wrapped_model)
    
    # Save both versions
    traced_path = output_dir / "swipe_model_traced.pt"
    scripted_path = output_dir / "swipe_model_scripted.pt"
    
    torch.jit.save(traced_model, str(traced_path))
    torch.jit.save(scripted_model, str(scripted_path))
    
    print(f"✅ Traced model saved to {traced_path}")
    print(f"✅ Scripted model saved to {scripted_path}")
    
    # Also save optimized mobile version
    mobile_path = output_dir / "swipe_model_mobile.ptl"
    optimized = torch.jit.optimize_for_mobile(scripted_model)
    optimized._save_for_lite_interpreter(str(mobile_path))
    print(f"✅ Mobile optimized model saved to {mobile_path}")
    
    return scripted_path, mobile_path


def create_java_interface(output_dir):
    """Create Java interface code for Android."""
    
    java_code = '''
package com.example.swipetyping;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class SwipeTypingModel {
    private Module model;
    private static final int FEATURE_DIM = 7;
    private static final int VOCAB_SIZE = 30;
    private static final int MAX_LENGTH = 35;
    
    // Token IDs
    private static final long PAD_TOKEN = 28;
    private static final long SOS_TOKEN = 29;
    private static final long EOS_TOKEN = 26;
    private static final long UNK_TOKEN = 27;
    
    // Character vocabulary
    private static final String[] VOCAB = {
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", 
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z", "<eos>", "<unk>", "<pad>", "<sos>"
    };
    
    public SwipeTypingModel(String modelPath) {
        model = Module.load(modelPath);
    }
    
    /**
     * Preprocess swipe gesture into features.
     * @param points List of swipe points with x, y, timestamp
     * @return Feature tensor [1, seq_len, 7]
     */
    public Tensor preprocessSwipe(float[][] points) {
        int seqLen = points.length;
        float[] features = new float[seqLen * FEATURE_DIM];
        
        for (int i = 0; i < seqLen; i++) {
            float x = points[i][0];
            float y = points[i][1];
            float t = points[i][2];
            
            // Calculate velocities
            float vx = 0, vy = 0;
            if (i > 0 && i < seqLen - 1) {
                vx = (points[i+1][0] - points[i-1][0]) / (points[i+1][2] - points[i-1][2]);
                vy = (points[i+1][1] - points[i-1][1]) / (points[i+1][2] - points[i-1][2]);
            }
            
            // Calculate accelerations
            float ax = 0, ay = 0;
            if (i > 1 && i < seqLen - 1) {
                float vx_prev = (points[i][0] - points[i-2][0]) / (points[i][2] - points[i-2][2]);
                float vy_prev = (points[i][1] - points[i-2][1]) / (points[i][2] - points[i-2][2]);
                ax = (vx - vx_prev) / (t - points[i-1][2]);
                ay = (vy - vy_prev) / (t - points[i-1][2]);
            }
            
            // Get nearest keyboard key (simplified - needs actual implementation)
            float kbId = getNearestKeyId(x, y);
            
            // Pack features
            int offset = i * FEATURE_DIM;
            features[offset] = x;
            features[offset + 1] = y;
            features[offset + 2] = vx;
            features[offset + 3] = vy;
            features[offset + 4] = ax;
            features[offset + 5] = ay;
            features[offset + 6] = kbId;
        }
        
        // Create tensor [1, seq_len, 7]
        long[] shape = {1, seqLen, FEATURE_DIM};
        return Tensor.fromBlob(features, shape);
    }
    
    /**
     * Predict word from swipe gesture.
     * @param swipeFeatures Preprocessed swipe features
     * @return Predicted word string
     */
    public String predict(Tensor swipeFeatures) {
        // Run model inference
        IValue output = model.forward(IValue.from(swipeFeatures));
        Tensor outputTensor = output.toTensor();
        
        // Convert tensor to token IDs
        long[] tokenIds = outputTensor.getDataAsLongArray();
        
        // Decode tokens to string
        StringBuilder word = new StringBuilder();
        for (long tokenId : tokenIds) {
            if (tokenId == EOS_TOKEN || tokenId == PAD_TOKEN) {
                break;
            }
            if (tokenId != SOS_TOKEN && tokenId < VOCAB.length) {
                word.append(VOCAB[(int)tokenId]);
            }
        }
        
        return word.toString();
    }
    
    private float getNearestKeyId(float x, float y) {
        // Simplified - needs actual keyboard layout logic
        // This should map (x,y) coordinates to keyboard key IDs 0-29
        // For now, return a dummy value
        return 0;
    }
}
'''
    
    java_path = output_dir / "SwipeTypingModel.java"
    with open(java_path, 'w') as f:
        f.write(java_code)
    
    print(f"✅ Java interface saved to {java_path}")
    
    # Also create gradle dependencies
    gradle_deps = '''
dependencies {
    implementation 'org.pytorch:pytorch_android_lite:1.13.1'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.13.1'
}
'''
    
    gradle_path = output_dir / "gradle_dependencies.txt"
    with open(gradle_path, 'w') as f:
        f.write(gradle_deps)
    
    print(f"✅ Gradle dependencies saved to {gradle_path}")


def main():
    # Setup paths
    checkpoint_path = "../checkpoints_english/english-epoch=51-val_loss=1.248-val_word_acc=0.659.ckpt"
    output_dir = Path("./torchscript_export/")
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting TorchScript export for Android/Java")
    print("=" * 50)
    
    try:
        # Load model
        model = load_model(checkpoint_path)
        print(f"✅ Model loaded successfully")
        
        # Export to TorchScript
        scripted_path, mobile_path = export_torchscript(model, output_dir)
        
        # Create Java interface
        create_java_interface(output_dir)
        
        # Create config
        config = {
            "model_type": "swipe-typing-transformer",
            "input_features": 7,
            "vocab_size": 30,
            "max_length": 35,
            "architecture": {
                "encoder_layers": 4,
                "decoder_layers": 4,
                "hidden_size": 128,
                "attention_heads": 4
            }
        }
        
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Config saved to {config_path}")
        
        print("=" * 50)
        print("🎉 Export completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print("\n📝 Android integration steps:")
        print("   1. Add PyTorch Android to your app's build.gradle")
        print("   2. Copy swipe_model_mobile.ptl to assets folder")
        print("   3. Use SwipeTypingModel.java for inference")
        print("   4. Implement getNearestKeyId() with your keyboard layout")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()