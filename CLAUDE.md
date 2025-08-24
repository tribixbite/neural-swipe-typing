# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🔴 CRITICAL: Project Management

**ALWAYS start by reviewing `/memory/pm.md` for current project status and tasks.**

The PM.md file contains:
- Current project goals and status
- Detailed step-by-step implementation plan
- Task checkboxes to track progress
- Critical path items and next actions
- Risk mitigation strategies

**After completing any task, UPDATE the PM.md file with:**
- Checkbox completions
- New discoveries or blockers
- Changes in approach
- Experimental results

## Project Overview

Neural Swipe Typing - A transformer neural network that transduces swipe gestures on a keyboard into word candidates. The project implements a novel method for constructing swipe point embeddings (SPE) that outperforms existing approaches.

## Build and Test Commands

### Training Pipeline
```bash
# Download preprocessed dataset (recommended - takes ~10 minutes)
cd src
python ./data_obtaining_and_preprocessing/download_dataset_preprocessed.py

# OR obtain and preprocess from scratch (takes ~6 hours)
bash ./data_obtaining_and_preprocessing/obtain_and_prepare_data.sh

# Run training (use train.ipynb in Kaggle/Colab for GPU support)
jupyter notebook src/train.ipynb
```

### Prediction and Evaluation
```bash
# Run prediction on validation set
python src/predict_v2.py --config configs/config__my_weighted_features.json --num-workers 0

# Evaluate predictions
python src/evaluate.py --predictions_path ./results/predictions/val/my_weighted_features/*.pkl
```

### Quality Checks
```bash
# Run unit tests (if available)
python -m pytest tests/

# Type checking (install mypy first)
mypy src/

# Linting
pylint src/
```

## Architecture Overview

### Core Model Components

1. **EncoderDecoderTransformerLike** (`src/model.py`)
   - Base class for all swipe decoding models
   - Components:
     - **Swipe Point Embedder**: Processes raw swipe coordinates into embeddings
     - **Word Component Token Embedder**: Character-level tokenization and embedding
     - **Encoder**: Processes swipe trajectories (Transformer/Conformer-based)
     - **Decoder**: Generates word candidates (Transformer-based)

2. **Feature Extraction Pipeline** (`src/feature_extraction/`)
   - Transforms raw swipe data `(x, y, t, grid_name, target_word)` 
   - Extracts trajectory features: coordinates, velocities, accelerations
   - Computes keyboard distance features and nearest key lookups

3. **Dataset Architecture** (`src/dataset.py`)
   - **CurveDataset**: Main dataset class handling swipe trajectories
   - Supports multiple keyboard layouts via `grid_name_to_grid.json`
   - Applies transformations: trajectory features, noise augmentation

### Current Refactoring Status

Based on `docs_and_assets/Refactoring_plan.md`, the major refactoring branches have been completed:
- ✅ **refactor-model-module**: Modularized swipe point embedders into separate components
- ✅ **refactor-feature-extraction**: Extracted feature processing into dedicated module
- ✅ **refactor-eval-and-predict**: Improved prediction and evaluation pipelines

### English Dataset Processing

The `data/swipelogs/` directory contains English swipe data in two formats:
- `.json` files: User metadata (age, gender, device info)
- `.log` files: Touch event sequences with timestamps and coordinates

To process this data for training:
1. Convert log format to training format using dataset conversion scripts
2. Apply keyboard layout mapping for English QWERTY
3. Generate trajectory features and tokenize words

## Key Implementation Details

### Model Configurations
- Models defined in `MODEL_GETTERS_DICT` in `src/model.py`
- Current best model: `v3_weighted_and_traj_transformer_bigger`
- Feature configs: `traj_feats_and_distance_weights`, `traj_feats_and_nearest_key`

### Training Specifics
- Uses PyTorch Lightning for training orchestration
- Batch sizes: Train=256, Val=512 (configurable)
- Learning rate scheduling: ReduceLROnPlateau with patience=20
- Label smoothing: 0.045
- Optimizer: Adam with lr=1e-4

### Decoding Algorithms
- **Greedy Search**: Fast, single-best decoding
- **Beam Search**: Higher accuracy with vocabulary masking
  - Custom implementation masks impossible token continuations
  - Beam size typically 6 with normalization factor 0.5

### Important Configuration Files
- `configs/config__my_weighted_features.json`: Best performing model config
- `data/data_preprocessed/gridname_to_grid.json`: Keyboard layout definitions
- `data/data_preprocessed/voc.txt`: Character vocabulary

## Working with English Dataset

### Current Status
**⚠️ English dataset conversion script needs to be created - see `/memory/pm.md` Phase 1**

The `data/swipelogs/` directory contains raw English swipe data that needs conversion:
- `.json` files: User metadata (device info, demographics)
- `.log` files: Touch events with columns: `sentence timestamp keyb_width keyb_height event x_pos y_pos x_radius y_radius angle word is_err`

### Expected Training Format
```json
{
  "word": "hello",
  "curve": {
    "x": [120, 125, 130, ...],
    "y": [89, 91, 95, ...],
    "t": [0, 15, 30, ...],
    "grid_name": "qwerty_english"
  }
}
```

### Conversion Pipeline (To Be Implemented)
1. Parse touch events from `.log` files
2. Group events by word boundaries (using timestamps and word column)
3. Filter out errors (is_err=1)
4. Map to QWERTY layout (needs to be added to gridname_to_grid.json)
5. Output as JSONL in expected format

### Training on English Data

1. Add English keyboard layout to `gridname_to_grid.json` if not present
2. Update `GRID_NAME_TO_DS_PATHS` in `train.ipynb` with English dataset paths
3. Set `GRID_NAME = "english"` in training configuration
4. Run training pipeline with appropriate hyperparameters

## Development Workflow

### Adding New Features
1. Implement feature extractor in `src/feature_extraction/feature_extractors.py`
2. Register in `get_transforms()` function
3. Update model architecture if needed in `src/model.py`
4. Test with small dataset subset first

### Debugging Tips
- Use `--num-workers 0` for prediction when using vocabulary masking (known bug)
- Memory issues in training: Set DataLoader `num_workers=0` or use `torch.cuda.empty_cache()`
- For reproducibility: Set random seeds and log git commit hash

### Performance Optimization
- Batch processing: Always prefer batched operations over loops
- Caching: Pre-compute keyboard distance lookups and nearest key mappings
- Multi-processing: Use appropriate number of workers (typically 4-8)

## Common Issues and Solutions

1. **Memory overflow during training**: 
   - Reduce batch size or set `num_workers=0`
   - Use gradient accumulation for larger effective batch sizes

2. **Slow prediction with vocabulary masking**:
   - Must use `--num-workers 0` (multiprocessing bug)
   - Consider using greedy search for faster inference

3. **Dataset preprocessing hangs**:
   - Check available disk space
   - Verify file permissions in output directories

## Next Steps for Refactoring

Per the refactoring plan, priority items include:
1. Complete English dataset integration pipeline
2. Implement trainable Gaussian noise augmentation
3. Migrate from Jupyter notebooks to Python scripts for production
4. Add comprehensive unit tests for core components
5. Optimize beam search performance with caching

## Important Notes

- Models require specific dataset transformations used during training
- Keyboard layouts must match between training and inference
- The web demo uses legacy model version - update planned for winter 2024
- Current best Swipe MRR: ~0.81 on validation set