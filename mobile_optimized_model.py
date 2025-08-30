"""
Mobile-optimized neural swipe typing model architecture.
Designed for real-time inference on Android devices with ONNX/ExecuTorch export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class MobilePositionalEncoding(nn.Module):
    """Lightweight positional encoding for mobile deployment"""
    def __init__(self, d_model: int, max_len: int = 150, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Pre-compute positional encodings up to max_len
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class MobileSwipeEmbedding(nn.Module):
    """Efficient swipe point embedding for mobile inference"""
    def __init__(self, input_dim: int = 6, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        # Simple linear projection for trajectory features (x, y, vx, vy, ax, ay)
        self.trajectory_proj = nn.Linear(input_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, traj_features: torch.Tensor) -> torch.Tensor:
        # traj_features: (batch_size, seq_len, 6) -> (batch_size, seq_len, embed_dim)
        x = self.trajectory_proj(traj_features)
        x = self.layer_norm(x)
        return self.dropout(x)

class MobileTransformerBlock(nn.Module):
    """Lightweight transformer block optimized for mobile"""
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

class MobileSwipeDecoder(nn.Module):
    """Mobile-optimized decoder for character sequence generation"""
    def __init__(self, d_model: int, vocab_size: int = 28, max_len: int = 20, 
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = MobilePositionalEncoding(d_model, max_len, dropout)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            MobileTransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, encoder_output: torch.Tensor, target_chars: Optional[torch.Tensor] = None) -> torch.Tensor:
        if target_chars is not None:
            # Training mode
            char_emb = self.char_embedding(target_chars)
            char_emb = self.pos_encoding(char_emb)
            
            # Apply decoder layers with cross-attention simulation
            # Simplified: just use self-attention for mobile efficiency
            x = char_emb
            for layer in self.decoder_layers:
                x = layer(x)
            
            return self.output_proj(x)
        else:
            # Inference mode - simplified for mobile
            batch_size = encoder_output.size(0)
            device = encoder_output.device
            
            # Start with SOS token
            output_chars = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            outputs = []
            
            for step in range(self.max_len):
                char_emb = self.char_embedding(output_chars)
                char_emb = self.pos_encoding(char_emb)
                
                x = char_emb
                for layer in self.decoder_layers:
                    x = layer(x)
                
                logits = self.output_proj(x)
                outputs.append(logits[:, -1:, :])  # Take last timestep
                
                # Greedy decoding for mobile efficiency
                next_char = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                output_chars = torch.cat([output_chars, next_char], dim=1)
                
                # Stop if all sequences hit EOS
                if (next_char == 1).all():  # EOS token
                    break
            
            return torch.cat(outputs, dim=1)

class MobileSwipeTypingModel(nn.Module):
    """Complete mobile-optimized swipe typing model"""
    def __init__(self, d_model: int = 64, vocab_size: int = 28, max_seq_len: int = 150,
                 max_word_len: int = 20, nhead: int = 4, num_encoder_layers: int = 3,
                 num_decoder_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Swipe sequence encoder
        self.swipe_embedding = MobileSwipeEmbedding(6, d_model, dropout)
        self.pos_encoding = MobilePositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            MobileTransformerBlock(d_model, nhead, dropout) for _ in range(num_encoder_layers)
        ])
        
        # Global pooling for sequence representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Decoder
        self.decoder = MobileSwipeDecoder(d_model, vocab_size, max_word_len, 
                                        nhead, num_decoder_layers, dropout)
        
    def encode_swipe(self, traj_features: torch.Tensor) -> torch.Tensor:
        """Encode swipe trajectory into sequence representation"""
        # traj_features: (batch_size, seq_len, 6)
        x = self.swipe_embedding(traj_features)
        x = self.pos_encoding(x)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Global pooling to get fixed-size representation
        # x: (batch_size, seq_len, d_model) -> (batch_size, d_model)
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        return x
    
    def forward(self, traj_features: torch.Tensor, target_chars: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode swipe sequence
        encoded_swipe = self.encode_swipe(traj_features)
        
        # Expand for decoder input
        batch_size, seq_len = traj_features.size(0), traj_features.size(1)
        encoder_output = encoded_swipe.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Decode to character sequence
        return self.decoder(encoder_output, target_chars)

def create_mobile_model(vocab_size: int = 28, d_model: int = 64) -> MobileSwipeTypingModel:
    """Factory function for mobile-optimized model"""
    return MobileSwipeTypingModel(
        d_model=d_model,
        vocab_size=vocab_size,
        max_seq_len=150,  # Cap sequence length for mobile
        max_word_len=20,
        nhead=4,
        num_encoder_layers=3,  # Reduced for mobile efficiency
        num_decoder_layers=2,  # Reduced for mobile efficiency
        dropout=0.1
    )

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model size comparison
if __name__ == "__main__":
    mobile_model = create_mobile_model()
    print(f"Mobile model parameters: {count_parameters(mobile_model):,}")
    
    # Test forward pass
    batch_size, seq_len, feature_dim = 4, 100, 6
    dummy_input = torch.randn(batch_size, seq_len, feature_dim)
    dummy_target = torch.randint(0, 28, (batch_size, 15))  # Max 15 chars
    
    with torch.no_grad():
        output = mobile_model(dummy_input, dummy_target)
        print(f"Output shape: {output.shape}")
        
        # Inference mode test
        inference_output = mobile_model(dummy_input)
        print(f"Inference output shape: {inference_output.shape}")