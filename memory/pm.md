# Project Memory - Neural Swipe Typing

## Completed Tasks

### ✅ English Adaptation (Complete)

Successfully adapted the neural swipe typing training pipeline from Cyrillic to English in 4 systematic phases:

**Phase 1: Character Set & Tokenizers**
- Updated `src/ns_tokenizers.py`: Changed `ALL_CYRILLIC_LETTERS_ALPHABET_ORD` (33 chars) to `ALL_ENGLISH_LETTERS_ALPHABET_ORD` (26 chars)
- Updated `src/feature_extraction/feature_extractors.py`: Import English alphabet and set as default allowed keys
- Removed Cyrillic character substitutions from `src/grid_processing_utils.py`

**Phase 2: Model Architecture Constants**
- Updated all `CHAR_VOCAB_SIZE` from 37 to 30 in `src/model.py` (26 letters + 4 special tokens)
- Updated all `n_keys` and `n_elements` parameters from 37 to 30
- Updated `src/train.ipynb`: Changed `num_classes` from 35 to 28

**Phase 3: English Keyboard Layout**
- Created `data/data_preprocessed/gridname_to_grid.json` with standard QWERTY layout
- Created `data/data_preprocessed/voc.txt` with ~100 common English words
- Generated proper key positions and hitboxes for English keyboard

**Phase 4: Configuration Updates & Testing**
- Updated all config files in `configs/` directory: `n_classes` from 35 to 28
- Successfully tested model initialization with English configuration
- Verified tokenizer compatibility and vocabulary size consistency

**Key Technical Changes:**
- Vocabulary: 26 English letters + 4 special tokens = 30 total
- Model output: n_classes = 28 (excluding `<sos>` and `<pad>` tokens)
- Keyboard: Standard QWERTY layout with proper coordinates
- All dimension constants updated consistently across codebase

**Testing Status:** ✅ PASSED
- Model initializes correctly with new English parameters
- Tokenizers work with English vocabulary  
- All architecture constants properly aligned

## Recent Completed Tasks

### ✅ English Dataset Integration (Complete)

Successfully integrated new English swipe dataset and adapted pipeline for training:

**Dataset Processing:**
- Processed 24,081 English swipe samples (15,177 train, 759 val, 8,145 test)
- Converted normalized coordinates (0-1 range) to absolute pixel coordinates (360x215)
- Fixed dataset loading to handle float coordinates and timestamps
- Updated vocabulary to 10k English words

**Coordinate System Adaptation:**
- Created coordinate conversion script (`convert_coordinates.py`)
- Updated keyboard layout to match 360x215 dimensions with proper QWERTY positioning
- Fixed feature extraction pipeline for English alphabet
- Updated `TrajFeatsGetter` normalization for new coordinate system

**Training Pipeline Setup:**
- Created `train_english.py` training script optimized for 16GB VRAM
- Configured for CUDA GPU with RTX 4090 (17.2GB VRAM)
- Set up PyTorch Lightning training with proper callbacks
- Fixed PyTorch compatibility issues (scheduler parameters)

**Technical Fixes Applied:**
- `dataset.py:21-24`: Added float-to-int conversion for coordinates and timestamps
- `feature_extractors.py:682`: Fixed Cyrillic→English alphabet reference
- `gridname_to_grid.json`: Created new QWERTY layout with 360x215 dimensions
- Model configuration: 6 coordinate features, batch size 64, 28 output classes

**Current Status:**
- ✅ Data pipeline working correctly
- ✅ Model initialization successful (1.1M parameters)
- ✅ GPU detection and setup working
- ⚠️ Minor tensor dimension issue in positional encoder (sequence length mismatch)

## Next Steps

### 🔧 Remaining Issues
1. **Model Architecture**: Fix positional encoder dimension mismatch for English sequence lengths
2. **Hyperparameter Tuning**: Optimize batch size and learning rate for new dataset scale

### 🔄 Ready for Full Training
Once dimensional issue resolved:
1. **Training**: Run full training pipeline with English dataset
2. **Evaluation**: Test model performance on English swipe typing
3. **Performance Optimization**: Fine-tune for best accuracy

### 📋 Future Enhancements
- Experiment with different model architectures for English
- Add data augmentation specific to English swipe patterns
- Evaluate performance against baseline English typing models