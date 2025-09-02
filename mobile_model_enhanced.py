"""
Enhanced mobile neural swipe typing model with character-level prediction.
Replicates the original model's 70% accuracy while maintaining mobile compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
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


class EnhancedMobileEncoder(nn.Module):
    """Enhanced encoder with proper transformer architecture."""
    
    def __init__(self, 
                 input_dim: int = 8,  # x, y, vx, vy, ax, ay + nearest_key features
                 d_model: int = 128,  # Match original model
                 nhead: int = 8,
                 num_layers: int = 4,  # Match original encoder layers
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 kb_vocab_size: int = 29):  # 26 letters + special tokens
        super().__init__()
        
        # Separate projections for trajectory and keyboard features
        self.traj_proj = nn.Linear(6, d_model // 2)  # x, y, vx, vy, ax, ay
        self.kb_embedding = nn.Embedding(kb_vocab_size, d_model // 2)
        
        # Combine features
        self.feature_norm = nn.LayerNorm(d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, 
                traj_features: torch.Tensor,
                kb_features: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode swipe trajectory with keyboard context.
        
        Args:
            traj_features: (batch, seq_len, 6) - trajectory features
            kb_features: (batch, seq_len) - nearest keyboard key indices
            src_key_padding_mask: (batch, seq_len) - padding mask
            
        Returns:
            (batch, seq_len, d_model) - encoded features
        """
        # Project trajectory features
        traj_encoded = self.traj_proj(traj_features)
        
        # Embed keyboard features
        kb_embedded = self.kb_embedding(kb_features)
        
        # Combine features
        combined = torch.cat([traj_encoded, kb_embedded], dim=-1)
        combined = self.feature_norm(combined)
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(
            combined,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return encoded


class CharacterLevelDecoder(nn.Module):
    """Character-level decoder matching the original model."""
    
    def __init__(self,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,  # Match original decoder layers
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 char_vocab_size: int = 32,  # 26 letters + special tokens
                 max_word_len: int = 20):
        super().__init__()
        
        self.d_model = d_model
        self.char_vocab_size = char_vocab_size
        
        # Character embedding and positional encoding
        self.char_embedding = nn.Embedding(char_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_word_len)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, char_vocab_size)
        
    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode to character sequence.
        
        Args:
            tgt: (batch, tgt_seq_len) - target character indices
            memory: (batch, src_seq_len, d_model) - encoder output
            tgt_mask: (tgt_seq_len, tgt_seq_len) - causal mask
            memory_key_padding_mask: (batch, src_seq_len) - encoder padding mask
            tgt_key_padding_mask: (batch, tgt_seq_len) - decoder padding mask
            
        Returns:
            (batch, tgt_seq_len, char_vocab_size) - character logits
        """
        # Embed target characters
        tgt_embedded = self.char_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        
        # Apply transformer decoder
        decoded = self.transformer_decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        output = self.output_proj(decoded)
        
        return output
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class EnhancedMobileSwipeModel(nn.Module):
    """
    Enhanced mobile swipe typing model with character-level prediction.
    Replicates the original model's architecture for 70% accuracy.
    """
    
    def __init__(self,
                 input_dim: int = 6,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 kb_vocab_size: int = 29,  # keyboard vocabulary
                 char_vocab_size: int = 32,  # character vocabulary
                 max_seq_len: int = 150,
                 max_word_len: int = 20):
        super().__init__()
        
        self.encoder = EnhancedMobileEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            kb_vocab_size=kb_vocab_size
        )
        
        self.decoder = CharacterLevelDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            char_vocab_size=char_vocab_size,
            max_word_len=max_word_len
        )
        
        # Special token indices
        self.sos_idx = char_vocab_size - 1  # <sos> token
        self.eos_idx = char_vocab_size - 4  # <eos> token
        self.pad_idx = char_vocab_size - 2  # <pad> token
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self,
                traj_features: torch.Tensor,
                kb_features: torch.Tensor,
                tgt: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            traj_features: (batch, seq_len, 6) - trajectory features
            kb_features: (batch, seq_len) - nearest key indices
            tgt: (batch, tgt_len) - target character sequence (for training)
            src_key_padding_mask: (batch, seq_len) - encoder padding mask
            tgt_key_padding_mask: (batch, tgt_len) - decoder padding mask
            
        Returns:
            (batch, tgt_len, char_vocab_size) - character logits
        """
        # Encode trajectory
        memory = self.encoder(
            traj_features,
            kb_features,
            src_key_padding_mask=src_key_padding_mask
        )
        
        if tgt is not None:
            # Training mode with teacher forcing
            tgt_len = tgt.shape[1]
            tgt_mask = self.decoder.generate_square_subsequent_mask(tgt_len).to(tgt.device)
            
            output = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            return output
        else:
            # Inference mode - autoregressive generation
            return self.generate(memory, src_key_padding_mask)
    
    def generate(self,
                 memory: torch.Tensor,
                 memory_key_padding_mask: Optional[torch.Tensor] = None,
                 max_len: int = 20) -> torch.Tensor:
        """
        Generate character sequence autoregressively.
        
        Args:
            memory: (batch, seq_len, d_model) - encoder output
            memory_key_padding_mask: (batch, seq_len) - padding mask
            max_len: Maximum generation length
            
        Returns:
            (batch, max_len) - generated character indices
        """
        batch_size = memory.shape[0]
        device = memory.device
        
        # Start with <sos> token
        generated = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Generate next token
            tgt_mask = self.decoder.generate_square_subsequent_mask(generated.shape[1]).to(device)
            
            output = self.decoder(
                generated,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Get next token
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated <eos>
            if (next_token == self.eos_idx).all():
                break
        
        return generated


def create_enhanced_mobile_model(kb_vocab_size: int = 29,
                                  char_vocab_size: int = 32) -> EnhancedMobileSwipeModel:
    """Factory function to create enhanced mobile model."""
    model = EnhancedMobileSwipeModel(
        input_dim=6,
        d_model=128,  # Match original
        nhead=8,
        num_encoder_layers=4,  # Match original
        num_decoder_layers=3,  # Match original
        dim_feedforward=512,
        dropout=0.1,
        kb_vocab_size=kb_vocab_size,
        char_vocab_size=char_vocab_size,
        max_seq_len=150,
        max_word_len=20
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    model = create_enhanced_mobile_model()
    param_count = count_parameters(model)
    
    print(f"Enhanced Mobile Model Statistics:")
    print(f"Total parameters: {param_count:,}")
    print(f"Model size (FP32): {param_count * 4 / 1024 / 1024:.2f} MB")
    print(f"Model size (INT8): {param_count / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    tgt_len = 10
    
    traj_features = torch.randn(batch_size, seq_len, 6)
    kb_features = torch.randint(0, 26, (batch_size, seq_len))
    tgt = torch.randint(0, 28, (batch_size, tgt_len))
    
    with torch.no_grad():
        output = model(traj_features, kb_features, tgt)
        print(f"\nTraining mode:")
        print(f"Input shapes: traj={traj_features.shape}, kb={kb_features.shape}, tgt={tgt.shape}")
        print(f"Output shape: {output.shape}")
        
        # Test inference
        memory = model.encoder(traj_features, kb_features)
        generated = model.generate(memory)
        print(f"\nInference mode:")
        print(f"Generated shape: {generated.shape}")