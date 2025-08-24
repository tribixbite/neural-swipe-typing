# Neural Swipe Typing - Architecture Overview

## Executive Summary

This project implements a transformer-based neural network for gesture keyboard input (swipe typing) that converts continuous touch trajectories into word predictions. The key innovation is a novel **Swipe Point Embedding (SPE)** method using weighted sums of keyboard key embeddings, which outperforms traditional nearest-key approaches by **0.67% in Swipe MRR**.

## 1. Model Architecture

### 1.1 Current Production Model: `v3_weighted_and_traj_transformer_bigger`

The architecture follows an encoder-decoder transformer pattern with four main components:

```
Raw Swipe → Feature Extraction → Encoder → Decoder → Word Prediction
```

#### Core Components:

1. **Swipe Point Embedder** (`SeparateTrajAndWeightedEmbeddingWithPos`)
   - Computes weighted sum of ALL keyboard key embeddings for each swipe point
   - Combines trajectory features with keyboard distance weights
   - Embedding dimension: 128 total (122 for keys + 6 for trajectory features)
   - Uses positional encoding with dropout=0.1

2. **Transformer Encoder**
   - 4 layers, 4 attention heads per layer
   - Hidden dimension (d_model): 128
   - Feedforward dimension: 128
   - Dropout: 0.1
   - Layer normalization after encoder stack

3. **Transformer Decoder**
   - 4 layers, 4 attention heads per layer
   - Hidden dimension: 128
   - Feedforward dimension: 128
   - Character-level output tokenization
   - Autoregressive generation with attention to encoder output

4. **Word Token Embedder**
   - Standard embedding layer for target characters
   - Vocabulary size: 37 tokens (including special tokens)
   - Combined with sinusoidal positional encoding

### 1.2 Model Variants Comparison

| Model Variant | SPE Method | Input Features | Use Case | Performance |
|--------------|------------|----------------|----------|-------------|
| **v3_weighted_and_traj** | Weighted sum of all keys | Trajectory + key weights | **Production** | **Best (MRR ~0.81)** |
| v3_nearest_and_traj | Single nearest key | Trajectory + nearest key ID | Baseline comparison | MRR ~0.80 |
| v3_nearest_only | Single nearest key | Only nearest key sequence | Ablation study | Lowest |
| v3_trainable_gaussian | Learnable noise params | Trajectory + learned weights | Experimental | Under development |

### 1.3 Key Innovation: Weighted Sum Embedding

Traditional approaches make "hard" decisions, assigning each swipe point to the single nearest key. This loses information when points are ambiguous (e.g., between 'E' and 'R').

Our approach:
- Maintains a probability distribution over ALL keyboard keys for each swipe point
- Computes embedding as weighted sum: `embedding = Σ(weight_i × key_embedding_i)`
- Preserves uncertainty and provides richer signal to the encoder
- Implemented efficiently as a linear layer: `nn.Linear(n_keys, embedding_dim)`

## 2. Data Pipeline

### 2.1 Input Data Format

**Raw swipe data:**
```json
{
  "x": [120, 125, 130, ...],      // X coordinates
  "y": [89, 91, 95, ...],          // Y coordinates  
  "t": [0, 15, 30, ...],           // Timestamps (ms)
  "grid_name": "qwerty_english",   // Keyboard layout ID
  "target_word": "hello"           // Ground truth
}
```

### 2.2 Feature Extraction Pipeline

1. **Trajectory Features** (6 dimensions when all enabled):
   - Raw coordinates (x, y): 2D
   - Velocities (vx, vy): 2D
   - Accelerations (ax, ay): 2D
   - Time: 1D (optional, currently disabled)

2. **Keyboard Features**:
   - Distance weights to all keys (Gaussian-based)
   - OR nearest key indices (for baseline models)

3. **Tokenization**:
   - Target word → character tokens with `<sos>` and `<eos>`
   - Decoder input: `[<sos>, h, e, l, l, o]`
   - Decoder output: `[h, e, l, l, o, <eos>]`

### 2.3 Model Input/Output Format

**After collation (batch-first=False):**
```python
encoder_input: (seq_len, batch_size, 128)  # Trajectory + key embeddings
decoder_input: (target_len, batch_size)     # Token indices
swipe_pad_mask: (batch_size, seq_len)       # Attention mask
word_pad_mask: (batch_size, target_len)     # Decoder mask
```

**Model output:**
```python
logits: (batch_size, target_len, vocab_size)  # Character probabilities
```

## 3. Training Configuration

### 3.1 Current Settings (English)

```json
{
  "batch_size_train": 256,
  "batch_size_val": 512,
  "learning_rate": 1e-4,
  "optimizer": "Adam",
  "label_smoothing": 0.045,
  "dropout": 0.2,
  "max_epochs": 100,
  "early_stopping_patience": 15,
  "val_check_interval": 1.0,  // Every epoch
  "lr_scheduler": "ReduceLROnPlateau"
}
```

### 3.2 Dataset Statistics

- Total English swipes: 87,166 (after filtering)
- Train: 69,732 (80%)
- Validation: 8,716 (10%)
- Test: 8,718 (10%)

## 4. Inference & Decoding

### 4.1 Decoding Algorithms

1. **Greedy Search**: Fast, single-best decoding
2. **Custom Beam Search** (Production):
   - Beam size: 6
   - Normalization factor: 0.5
   - **Vocabulary masking**: Masks impossible token continuations based on dictionary
   - Significantly faster and more accurate than standard beam search

### 4.2 Performance Metrics

- **Swipe MRR** (Mean Reciprocal Rank): ~0.81
- **Word-level accuracy**: ~73% (top-1)
- **Inference time**: <100ms target for mobile

## 5. Mobile Deployment (ExecutorTorch)

### 5.1 Export Pipeline

```bash
PyTorch Model (.pt) → torch.export → ExecutorTorch (.pte) → Android App
```

**Steps:**
1. Load trained model and set to eval mode
2. Export using `torch.export` to capture computation graph
3. Lower to ExecutorTorch with operator decomposition
4. Apply XNNPACK backend delegation for mobile optimization
5. Save as `.pte` file with test cases for verification

### 5.2 Android Integration

```kotlin
// 1. Load model from assets
val model = ExecutorTorchModule.load("model.pte")

// 2. On swipe complete
val swipeData = collectSwipePoints()
val features = extractFeatures(swipeData)  // Must match training pipeline

// 3. Run inference
val output = model.forward(features)
val candidates = beamSearch(output)

// 4. Display suggestions
showCandidates(candidates)
```

### 5.3 Mobile Constraints

- **Memory budget**: ~60MB (iOS limit, good Android target)
- **Inference speed**: <100ms required
- **Model size**: Currently ~15MB (before optimization)

## 6. Caveats & Limitations

### 6.1 Current Issues

1. **Memory efficiency**: Model needs quantization for mobile deployment
2. **Multiprocessing bug**: Vocabulary masking breaks with `num_workers > 0`
3. **Language support**: Currently single-language, needs architecture for multi-language
4. **Training memory**: DataLoader can consume `dataset_size × num_workers` RAM

### 6.2 Architecture Limitations

1. **Fixed keyboard layouts**: Requires pre-defined grid configurations
2. **Character-level generation**: No word-piece or subword tokenization
3. **No online learning**: Cannot adapt to user-specific patterns
4. **Context limitations**: Doesn't use previous words for prediction

## 7. Future Improvements

### 7.1 In Development

1. **Trainable Gaussian Augmentation**
   - Learn noise distribution from data
   - Model variant exists, needs training
   - Could improve robustness to imprecise swipes

2. **Vocabulary Masking During Training**
   - Currently only at inference
   - Could simplify optimization and improve convergence
   - Planned in refactoring roadmap

### 7.2 Planned Enhancements

1. **Model Optimization**:
   - Quantization (float32 → int8) for 4× size reduction
   - Pruning to remove redundant weights
   - Knowledge distillation to smaller student models

2. **Multi-language Architecture**:
   - Shared encoder with language-specific decoders
   - Dynamic vocabulary and keyboard layout switching
   - Transfer learning from high-resource to low-resource languages

3. **Advanced Features**:
   - Attention visualization for debugging
   - User-specific fine-tuning
   - Context-aware prediction using previous words
   - Conformer-based encoder (replacing transformer)

## 8. Comparison with Industry Solutions

### 8.1 Google GBoard
- Also uses on-device neural models
- Reported to use LSTM-based architecture (older publications)
- Similar memory constraints (~60MB on iOS)

### 8.2 Grammarly Keyboard
- Blog mentions transformer-based approach
- Similar architecture but different feature extraction
- Our weighted-sum SPE is novel contribution

### 8.3 Key Differentiators
- **Novel SPE method**: Weighted-sum embedding vs nearest-key
- **Custom beam search**: With vocabulary masking
- **Open source**: Full implementation available
- **Modular design**: Easy to experiment with components

## 9. Repository Structure

```
neural-swipe-typing/
├── src/
│   ├── model.py                    # Model architectures
│   ├── train_english.py            # Training script
│   ├── dataset.py                  # Data loading
│   ├── feature_extraction/         # Feature extractors
│   ├── ns_tokenizers.py           # Tokenization
│   └── create_english_keyboard_tokenizer.py
├── configs/
│   └── config_english.json        # Training configuration
├── data/
│   ├── swipelogs/                 # Raw English data
│   └── data_preprocessed/         # Processed datasets
└── executorch/                    # Mobile export (branch)
```

## 10. Getting Started

### Training from Scratch
```bash
# Prepare data
python src/prepare_english_dataset.py

# Train model (GPU recommended)
python src/train_english.py --config configs/config_english.json --gpus 1

# Export for mobile
python src/export_executorch.py --checkpoint best_model.ckpt
```

### Using Pretrained Model
```bash
# Download checkpoint
wget [model_url]

# Run inference
python src/predict_v2.py --config configs/config_english.json

# Evaluate
python src/evaluate.py --predictions results/predictions/
```

## References

- Original Yandex Cup 2023 competition (7th place solution)
- [Report on SPE methods](docs_and_assets/report/report.md)
- [Master's thesis (Russian)](https://drive.google.com/file/d/1ad9zlfgfy6kOA-41GxjUQIzr8cWuaqxL/view)
- [GBoard Blog](https://research.google/blog/the-machine-intelligence-behind-gboard/)
- [Grammarly Blog](https://www.grammarly.com/blog/engineering/deep-learning-swipe-typing/)

---

*This architecture overview is current as of the English dataset integration (Phase 3). For the latest updates, check the git history and PM.md.*