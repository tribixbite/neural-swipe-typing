# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Data Preparation
- **Get preprocessed dataset (recommended)**: `cd src && python data_obtaining_and_preprocessing/download_dataset_preprocessed.py`
- **Full data pipeline from scratch**: `cd src && bash data_obtaining_and_preprocessing/obtain_and_prepare_data.sh` (takes ~6 hours)

### Training
- **Train models**: Open and run `src/train.ipynb` in Jupyter
- **Note**: Use `n_workers=0` in DataLoader to avoid RAM issues with PyTorch Lightning

### Prediction & Evaluation
- **Generate predictions**: `python src/predict_v2.py --config configs/config__my_weighted_features.json --num-workers 0`
- **Evaluate predictions**: `python src/evaluate.py --config configs/config_evaluation.json`
- **Demo prediction**: `src/word_generation_demo.ipynb`

### Testing
- **Unit tests**: `python -m pytest src/unittests/`
- **Single test**: `python src/unittests/test_distance_getter.py`

### Dependencies
- **Install**: `pip install -r requirements.txt`
- **Python version**: 3.10 (tested)

## Architecture Overview

This is a neural swipe typing system that converts touch gestures on a keyboard into word predictions using transformer models.

### Core Components

**Models** (`src/model.py`):
- `EncoderDecoderTransformerLike`: Main model architecture
- Components: swipe point embedder, word token embedder, encoder, decoder
- Model variants available in `MODEL_GETTERS_DICT`

**Feature Extraction** (`src/feature_extraction/`):
- `feature_extractors.py`: Transforms raw swipe data into model inputs
- `nearest_key_lookup.py`: Keyboard key proximity calculations
- `distances_lookup.py`: Distance-based features

**Data Pipeline**:
- Raw format: `(x, y, t, grid_name, tgt_word)`
- Transformed: `(encoder_input, decoder_input), decoder_output`
- Collated: `(packed_model_in, dec_out)` where `packed_model_in = (encoder_input, decoder_input, swipe_pad_mask, word_pad_mask)`

**Key Files**:
- `src/dataset.py`: Dataset loading and processing
- `src/ns_tokenizers.py`: Character and keyboard tokenization
- `src/word_generators_v2.py`: Beam search and word generation
- `src/logit_processors.py`: Custom beam search with vocabulary masking
- `src/metrics.py`: Evaluation metrics (MRR, accuracy)

### Configuration System

Models are configured via JSON files in `configs/`:
- Training dataset, model architecture, feature extraction method
- Decoding algorithm parameters
- Different feature extraction methods: `my_weighted_features`, `nearest_features`, `indiswipe_features`, etc.

### Important Notes

- **Vocabulary masking**: When `use_vocab_for_generation: true` in config, MUST use `--num-workers 0` due to performance issues
- **Memory usage**: Training notebooks can consume significant RAM with multiple workers
- **Grid layouts**: Keyboard layouts stored in `data_preprocessed/gridname_to_grid.json`
- **DVC integration**: Large datasets and models managed via DVC (`.dvc` files)

### Model Definition

A trained swipe decoder requires:
1. Model class and weights
2. Dataset transformation used during training  
3. Decoding algorithm configuration

### Custom Datasets

Must provide items as `(x, y, t, grid_name, tgt_word)` tuples and add keyboard layout to `gridname_to_grid.json`.