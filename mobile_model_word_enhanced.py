"""
Enhanced mobile model with word-level prediction incorporating original model's features.
Combines the efficiency of word prediction with the accuracy of the original approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 150):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. Shape: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EnhancedWordEncoder(nn.Module):
    """
    Enhanced encoder that processes both trajectory features and nearest key information.
    Similar to the original model but optimized for word-level output.
    """
    
    def __init__(self,
                 traj_dim: int = 6,  # x, y, vx, vy, ax, ay
                 d_model: int = 128,  # Match original model size
                 nhead: int = 8,
                 num_layers: int = 4,  # Match original encoder layers
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 kb_vocab_size: int = 29):  # 26 letters + special tokens
        super().__init__()
        
        # Trajectory feature projection (like original's traj_feats)
        self.traj_proj = nn.Sequential(
            nn.Linear(traj_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Keyboard feature embedding (like original's nearest_key features)
        self.kb_embedding = nn.Embedding(kb_vocab_size, d_model)
        self.kb_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder (match original architecture)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Global context aggregation for word prediction
        self.context_attention = nn.MultiheadAttention(
            d_model, 
            num_heads=nhead, 
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self,
                traj_features: torch.Tensor,
                kb_features: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode swipe trajectory with keyboard context.
        
        Args:
            traj_features: (batch, seq_len, 6) - trajectory features
            kb_features: (batch, seq_len) - nearest keyboard key indices  
            src_mask: (batch, seq_len) - padding mask
            
        Returns:
            (batch, d_model) - encoded representation for word prediction
        """
        batch_size, seq_len, _ = traj_features.shape
        
        # Process trajectory features
        traj_encoded = self.traj_proj(traj_features)
        
        # Process keyboard features
        kb_embedded = self.kb_embedding(kb_features)
        kb_encoded = self.kb_proj(kb_embedded)
        
        # Fuse features
        combined = torch.cat([traj_encoded, kb_encoded], dim=-1)
        fused = self.fusion(combined)
        
        # Add positional encoding
        fused = self.pos_encoder(fused)
        
        # Apply transformer encoder
        encoded = self.transformer(fused, src_key_padding_mask=src_mask)
        
        # Aggregate sequence into fixed-size representation
        # Use attention-based pooling for better context capture
        query = encoded.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        context, _ = self.context_attention(
            query, encoded, encoded,
            key_padding_mask=src_mask
        )
        context = context.squeeze(1)  # (batch, d_model)
        
        return context


class WordPredictionHead(nn.Module):
    """
    Prediction head for word-level output.
    More sophisticated than the original mobile model to match accuracy.
    """
    
    def __init__(self,
                 d_model: int = 128,
                 vocab_size: int = 10000,
                 hidden_dim: int = 512,  # Larger hidden layer
                 num_layers: int = 3,  # Deeper MLP
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        input_dim = d_model
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final projection to vocabulary
        layers.append(nn.Linear(hidden_dim, vocab_size))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict word probabilities.
        
        Args:
            x: (batch, d_model) - encoded representation
            
        Returns:
            (batch, vocab_size) - word logits
        """
        return self.mlp(x)


class EnhancedMobileWordModel(nn.Module):
    """
    Enhanced mobile model for word-level swipe typing.
    Incorporates successful features from the original 70% accuracy model
    while maintaining word-level prediction for mobile efficiency.
    """
    
    def __init__(self,
                 traj_dim: int = 6,
                 d_model: int = 128,  # Match original
                 nhead: int = 8,
                 num_encoder_layers: int = 4,  # Match original
                 dim_feedforward: int = 512,
                 hidden_dim: int = 512,
                 num_pred_layers: int = 3,
                 dropout: float = 0.1,
                 kb_vocab_size: int = 29,
                 word_vocab_size: int = 10000):
        super().__init__()
        
        self.encoder = EnhancedWordEncoder(
            traj_dim=traj_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            kb_vocab_size=kb_vocab_size
        )
        
        self.predictor = WordPredictionHead(
            d_model=d_model,
            vocab_size=word_vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_pred_layers,
            dropout=dropout
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self,
                traj_features: torch.Tensor,
                kb_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for word prediction.
        
        Args:
            traj_features: (batch, seq_len, 6) - trajectory features
            kb_features: (batch, seq_len) - nearest key indices
            mask: (batch, seq_len) - padding mask
            
        Returns:
            (batch, word_vocab_size) - word probabilities
        """
        # Encode trajectory with keyboard context
        encoded = self.encoder(traj_features, kb_features, mask)
        
        # Predict word
        logits = self.predictor(encoded)
        
        return logits
    
    def predict_top_k(self, 
                      traj_features: torch.Tensor,
                      kb_features: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k word predictions with scores.
        
        Returns:
            (indices, scores) both of shape (batch, k)
        """
        logits = self.forward(traj_features, kb_features, mask)
        scores = F.softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        
        return top_indices, top_scores


def create_enhanced_word_model(kb_vocab_size: int = 29,
                                word_vocab_size: int = 10000) -> EnhancedMobileWordModel:
    """
    Factory function to create enhanced word-level model.
    Uses architecture insights from the original 70% accuracy model.
    """
    model = EnhancedMobileWordModel(
        traj_dim=6,  # x, y, vx, vy, ax, ay
        d_model=128,  # Match original model size
        nhead=8,
        num_encoder_layers=4,  # Match original encoder
        dim_feedforward=512,
        hidden_dim=512,  # Larger prediction head
        num_pred_layers=3,  # Deeper prediction head
        dropout=0.1,
        kb_vocab_size=kb_vocab_size,
        word_vocab_size=word_vocab_size
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = create_enhanced_word_model()
    param_count = count_parameters(model)
    
    print(f"Enhanced Word-Level Mobile Model:")
    print(f"Total parameters: {param_count:,}")
    print(f"Model size (FP32): {param_count * 4 / 1024 / 1024:.2f} MB")
    print(f"Model size (INT8): {param_count / 1024 / 1024:.2f} MB")
    print(f"Target: Match original model's 70% accuracy with word-level prediction")
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    
    traj_features = torch.randn(batch_size, seq_len, 6)
    kb_features = torch.randint(0, 26, (batch_size, seq_len))
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[:, 80:] = True  # Simulate padding
    
    with torch.no_grad():
        output = model(traj_features, kb_features, mask)
        print(f"\nInput shapes: traj={traj_features.shape}, kb={kb_features.shape}")
        print(f"Output shape: {output.shape}")
        
        # Test top-k prediction
        top_indices, top_scores = model.predict_top_k(traj_features, kb_features, mask, k=5)
        print(f"Top-5 predictions shape: {top_indices.shape}")
        print(f"Top-5 scores shape: {top_scores.shape}")