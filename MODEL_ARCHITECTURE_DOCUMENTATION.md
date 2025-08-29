# Neural Swipe Typing Model - Complete Architecture Documentation

## Overview
This is an encoder-decoder transformer model specifically designed for gesture typing prediction. It converts swipe gestures on a keyboard into predicted words.

## Model Architecture

### Components
1. **Encoder**: 4-layer transformer encoder
   - Hidden dimension: 128
   - Attention heads: 4
   - Feedforward dimension: 128
   - Dropout: 0.1
   - Activation: ReLU

2. **Decoder**: Similar transformer decoder for autoregressive generation

3. **Embeddings**:
   - Swipe point embedder (SeparateTrajAndNearestEmbeddingWithPos)
   - Word token embedder (character-level with positional encoding)

## Input Pipeline

### Raw Input Format
- **X**: Array of x-coordinates for swipe points
- **Y**: Array of y-coordinates for swipe points  
- **T**: Array of timestamps (milliseconds from swipe start)
- **grid_name**: Keyboard layout identifier ("default" or "extra")
- **tgt_word**: Target word (for training only)

### Feature Extraction

#### Trajectory Features (6 dimensions):
1. **x**: Normalized x-coordinate (0-1)
2. **y**: Normalized y-coordinate (0-1)
3. **vx**: Velocity in x direction (dx/dt)
4. **vy**: Velocity in y direction (dy/dt)
5. **ax**: Acceleration in x direction (d²x/dt²)
6. **ay**: Acceleration in y direction (d²y/dt²)

#### Keyboard Features:
- **Nearest key ID**: Integer 0-29 representing closest keyboard key
- OR **Weighted distances**: Float weights to all 30 keys

### Model Input Format
The model expects a **tuple** of two tensors:
```python
encoder_input = (trajectory_features, keyboard_features)
```

**Dimensions**:
- `trajectory_features`: [seq_len, batch_size, 6]
- `keyboard_features`: [seq_len, batch_size] (for IDs) or [seq_len, batch_size, 30] (for weights)

## Padding and Masking

### Padding Masks
- **x_pad_mask** (encoder): Shape [batch_size, seq_len], dtype=bool
  - True = padded position, False = real data
- **y_pad_mask** (decoder): Shape [batch_size, chars_seq_len], dtype=bool
  - Used for variable-length word sequences

### Padding Values
- Swipe sequences: Padded with 0
- Word sequences: Padded with token ID 28 (`<pad>`)

## Vocabulary and Tokenization

### Keyboard Tokens (30 total):
- 0-25: Letters 'a' to 'z'
- 26: `<eos>` (end of sequence)
- 27: `<unk>` (unknown)
- 28: `<pad>` (padding)
- 29: `<sos>` (start of sequence)

### Character Tokenization:
- Same 30-token vocabulary for word generation
- Words are character sequences terminated by `<eos>`

## Model Forward Pass

### Training Mode:
```python
def forward(self, x, y, x_pad_mask, y_pad_mask):
    x_encoded = self.encode(x, x_pad_mask)
    return self.decode(y, x_encoded, x_pad_mask, y_pad_mask)
```

### Inference Mode:
1. Encode swipe once: `memory = encoder(swipe_features)`
2. Autoregressively generate characters:
   - Start with `<sos>` token
   - Decode next character using memory and generated sequence
   - Stop at `<eos>` or max length

## Key Parameters

### Dimensions:
- Max swipe length: 299 points
- Max word length: 35 characters  
- Batch size (training): 256
- Batch size (validation): 512
- Hidden size: 128
- Vocabulary size: 30

### Training Configuration:
- Optimizer: Adam (lr=1e-4, weight_decay=0)
- Loss: Cross-entropy with label smoothing (0.045)
- Learning rate scheduler: ReduceLROnPlateau

## Export Challenges and Solutions

### Main Issues:
1. **Tuple Input**: Model expects `(traj_feats, kb_ids)` tuple, ONNX prefers single tensor
2. **Padding Masks**: Must be boolean type, incompatible with some ONNX operations
3. **Autoregressive Decoder**: Dynamic looping not supported in static ONNX graph

### Recommended Export Strategy:

#### For Web (ONNX):
1. Create wrapper that concatenates inputs: [batch, seq_len, 7] where dim 7 = 6 trajectory + 1 kb_id
2. Export encoder and decoder separately
3. Implement autoregressive loop in JavaScript
4. Use ONNX opset 14+ for better compatibility

#### For Android/Java:
1. Use TorchScript instead of ONNX for better compatibility
2. Export full model with torch.jit.script
3. Use PyTorch Mobile for Android integration
4. Handle feature extraction in Java

## Implementation Notes

### Critical Constraints:
- Keyboard IDs must be in range [0, 29]
- Padding masks must be boolean tensors
- Sequence lengths are variable but have maximum limits
- Model uses teacher forcing during training

### Performance Considerations:
- Encoder only needs to run once per swipe
- Decoder runs multiple times (once per character)
- Beam search can improve prediction quality
- Vocabulary masking can constrain predictions to valid words