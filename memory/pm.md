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

### ✅ Raw Log Data Processing & Validation (Complete)

Successfully improved and ran the log processing pipeline with comprehensive data validation:

**Data Processing Improvements:**
- Added error flag validation (columns 11/12) filtering out flagged swipe traces
- Implemented touchstart/touchend sequence validation for proper touch events
- Added coordinate array validation ensuring equal x, y, t value counts
- Increased word length filter from >1 to >=3 characters minimum
- Added trajectory similarity analysis for repeated words using interpolation and distance metrics

**Dataset Statistics:**
- Processed 1,052 log files from `/data/swipetraces/`
- Found 976 files (92.8%) containing error-flagged data
- Extracted 37,688 training samples after validation filtering
- Generated clean datasets: `raw_converted_english_swipes_train.jsonl` (30,150), `_val.jsonl` (3,769), `_test.jsonl` (3,769)

**Keyboard Layout Analysis:**
- Detected Y coordinate usage: 71.6% of available 215px height space
- Identified 3 main keyboard rows at Y positions [67.5, 102.1, 136.8]
- Confirmed 4-row layout with spacebar row (25% Y space unused as expected)
- High trajectory similarity for repeated words (e.g., 'the': 0.869 average similarity)

**Technical Implementation:**
- `validate_touch_sequence()`: Ensures proper touchstart→touchend sequences
- `has_error_flag()`: Filters lines with error flag = 1 in columns 11/12
- `analyze_trajectory_similarity()`: Computes similarity metrics for repeated word curves
- `analyze_keyboard_layout()`: Maps coordinate distribution to keyboard structure

### ✅ Synthetic Trace Generation Pipeline (Complete)

Created comprehensive synthetic swipe trace generation system using wordgesturegan.com API:

**Generation Scripts:**
- `generate_synthetic_traces.py`: Main generation script with robust API handling
- `run_synthetic_generation.py`: User-friendly wrapper for different generation modes
- Support for batch processing, error handling, and resumable generation

**API Integration:**
- Full wordgesturegan.com API integration with proper headers and authentication
- Multiple noise levels for data augmentation: std_dev [0.5, 1.0, 1.5, 2.0]
- 1 second delay between requests to respect rate limits
- Comprehensive error handling with retries and timeout management

**Data Generation Features:**
- Batch processing with configurable batch sizes (default: 20-500 traces per file)
- Progress tracking with detailed statistics and logging
- Resumable generation from any starting word index
- Test mode for validation and full production mode for complete dataset
- Generated trace format matches existing pipeline requirements

**Output Format:**
- JSONL files with traces containing: `word_seq` (time, x, y arrays), `word`, `std_dev`, `timestamp`
- Generation summary with statistics, success rates, and failed word tracking
- Comprehensive logging for monitoring long-running generation processes

**Testing Results:**
- Successfully tested with 20 words generating 36 synthetic traces
- 100% success rate with API integration working correctly
- Ready for full 10k vocabulary generation (~40k synthetic traces expected)

**Production Generation (In Progress):**
- **ACTIVE**: Full 10k word generation running in background
- **Target**: 39,896 synthetic traces (9,974 words × 4 noise levels)
- **Filtering**: Includes 2+ character words, excludes only 26 single-character words
- **Optimization**: Exponential backoff, 1s base delay with jitter, 5 retries, 15s timeout
- **Progress**: Running smoothly at ~4-5 seconds per word, 0% failure rate
- **ETA**: 11-14 hours for completion (~40k traces)

**Enhanced Parameters:**
- **Backoff Strategy**: Exponential backoff (1.5x multiplier) with random jitter
- **Rate Limiting**: 0.8-1.2s random delays between requests
- **Reliability**: 5 retry attempts with 15s timeout per request
- **Automation**: `--no-confirm` flag for unattended execution

## Next Steps

### 🔧 Remaining Issues
1. **Model Architecture**: Fix positional encoder dimension mismatch for English sequence lengths
2. **Hyperparameter Tuning**: Optimize batch size and learning rate for new dataset scale

### 🔄 Ready for Full Training
Once dimensional issue resolved:
1. **Training**: Run full training pipeline with cleaned English dataset (37,688 samples)
2. **Evaluation**: Test model performance on English swipe typing
3. **Performance Optimization**: Fine-tune for best accuracy

### 📋 Future Enhancements
- Experiment with different model architectures for English
- Add data augmentation specific to English swipe patterns
- Evaluate performance against baseline English typing models
- Consider using trajectory similarity analysis for data augmentation