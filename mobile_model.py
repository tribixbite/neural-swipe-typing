"""
Mobile-optimized neural swipe typing model for Android deployment.
Designed for ONNX and ExecuTorch export with <10MB size and <50ms latency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MobilePositionalEncoding(nn.Module):
    """Lightweight positional encoding without dropout for mobile inference."""
    
    def __init__(self, d_model: int, max_len: int = 150):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. Shape: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1)]


class DepthwiseSeparableConv1d(nn.Module):
    """Mobile-friendly depthwise separable convolution for sequence processing."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size//2, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shape: (batch, channels, seq_len)"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return F.relu(x)


class EfficientAttention(nn.Module):
    """Efficient attention mechanism with linear complexity for mobile."""
    
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Efficient attention using linear approximation.
        Shape: (batch, seq_len, d_model)"""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        
        # Transpose for attention: (batch, heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Efficient attention using kernel trick (linear complexity)
        # Instead of (Q @ K^T) @ V, compute Q @ (K^T @ V)
        k_cumsum = k.sum(dim=2, keepdim=True)  # Global key statistics
        v_weighted = (k.transpose(-2, -1) @ v) / (seq_len ** 0.5)
        
        # Apply query
        attn_output = q @ v_weighted
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)


class MobileEncoderBlock(nn.Module):
    """Lightweight encoder block for mobile deployment."""
    
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attention = EfficientAttention(d_model, num_heads)
        self.conv = DepthwiseSeparableConv1d(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shape: (batch, seq_len, d_model)"""
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Depthwise separable conv with residual
        # Transpose for conv: (batch, d_model, seq_len)
        conv_in = x.transpose(1, 2)
        conv_out = self.conv(conv_in).transpose(1, 2)
        x = self.norm2(x + conv_out)
        
        return x


class MobileSwipeEncoder(nn.Module):
    """Mobile-optimized encoder for swipe sequences."""
    
    def __init__(self, input_dim: int = 6, d_model: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = MobilePositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            MobileEncoderBlock(d_model) for _ in range(num_layers)
        ])
        
        # Global pooling for sequence representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode swipe sequence.
        Input shape: (batch, seq_len, input_dim)
        Output shape: (batch, d_model)"""
        
        # Project input features
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Global pooling to get fixed-size representation
        # Transpose for pooling: (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        return x


class VocabularyDecoder(nn.Module):
    """Fast vocabulary-based decoder for word prediction."""
    
    def __init__(self, d_model: int = 64, vocab_size: int = 10000, 
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode to vocabulary scores.
        Input shape: (batch, d_model)
        Output shape: (batch, vocab_size)"""
        return self.mlp(encoded)


class MobileSwipeTypingModel(nn.Module):
    """Complete mobile-optimized swipe typing model."""
    
    def __init__(self, 
                 input_dim: int = 6,  # x, y, vx, vy, ax, ay
                 d_model: int = 64,
                 num_layers: int = 2,
                 vocab_size: int = 10000,
                 max_seq_len: int = 150):
        super().__init__()
        
        self.encoder = MobileSwipeEncoder(input_dim, d_model, num_layers)
        self.decoder = VocabularyDecoder(d_model, vocab_size)
        
        # For ONNX export
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for swipe typing.
        Input shape: (batch, seq_len, 6) - trajectory features
        Output shape: (batch, vocab_size) - word probabilities"""
        
        # Encode swipe sequence
        encoded = self.encoder(x, mask)
        
        # Decode to vocabulary
        logits = self.decoder(encoded)
        
        return logits
    
    def predict_top_k(self, x: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k word predictions with scores.
        Returns: (indices, scores) both of shape (batch, k)"""
        
        logits = self.forward(x)
        scores = F.softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        
        return top_indices, top_scores


def create_mobile_model(vocab_size: int = 10000) -> MobileSwipeTypingModel:
    """Factory function to create mobile model with default settings."""
    return MobileSwipeTypingModel(
        input_dim=6,  # x, y, vx, vy, ax, ay
        d_model=64,   # Smaller than original 128
        num_layers=2, # Fewer layers than original 4+3
        vocab_size=vocab_size,
        max_seq_len=150
    )


# Model size calculation
def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation and size
    model = create_mobile_model()
    param_count = count_parameters(model)
    
    print(f"Mobile Model Statistics:")
    print(f"Total parameters: {param_count:,}")
    print(f"Model size (FP32): {param_count * 4 / 1024 / 1024:.2f} MB")
    print(f"Model size (INT8): {param_count / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    dummy_input = torch.randn(batch_size, seq_len, 6)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Test top-k prediction
        top_indices, top_scores = model.predict_top_k(dummy_input, k=5)
        print(f"Top-5 predictions shape: {top_indices.shape}")