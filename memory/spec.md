# Specification: Cyrillic to English Adaptation - Complete Analysis

## CRITICAL ADAPTATION POINTS IDENTIFIED

### 1. CHARACTER SET TRANSFORMATION
**Primary Changes Required:**
- Replace Cyrillic alphabet (33 chars) with English alphabet (26 chars)
- Update ALL_CYRILLIC_LETTERS_ALPHABET_ORD → ALL_ENGLISH_LETTERS_ALPHABET_ORD
- Modify KeyboardTokenizerv1 to use English characters
- Adjust CHAR_VOCAB_SIZE from 37 to 30 (26 + 4 special tokens)
- Update n_classes from 35 to 28 (new_vocab_size - 2)

### 2. MODEL ARCHITECTURE UPDATES  
**Files to Modify:**
- `src/model.py`: Update all CHAR_VOCAB_SIZE constants (lines 427, 474, 785, 820, 855)
- `src/model.py`: Change n_keys from 37 to 30
- All config JSON files: Change "n_classes" from 35 to 28
- `src/train.ipynb`: Update num_classes parameter to 28

### 3. KEYBOARD LAYOUT ADAPTATION
**Required Changes:**
- Create new gridname_to_grid.json with English QWERTY layout
- Update key labels from Cyrillic to English characters
- Adjust keyboard dimensions for QWERTY layout proportions
- Modify character substitutions logic (remove Cyrillic-specific substitutions)

### 4. SEQUENCE LENGTH RECALCULATION
**English Word Considerations:**
- Analyze average English word length vs Cyrillic
- Potentially adjust MAX_OUT_SEQ_LEN (currently 35) based on English word statistics
- Consider MAX_CURVES_SEQ_LEN (299) - may need adjustment for English gesture patterns

### 5. DATA PIPELINE MODIFICATIONS
**Training Data Requirements:**
- Create English swipe gesture dataset
- Generate voc.txt with English vocabulary
- Update data preprocessing scripts for English character validation
- Modify dataset paths in training configurations

### 6. TRAJECTORY FEATURE CALCULATIONS
**No Changes Required for:**
- Timing calculations (get_dx_dt function)
- Velocity calculations (dx_dt, dy_dt)  
- Acceleration calculations (d2x_dt2, d2y_dt2)
- Coordinate normalization logic

### 7. FEATURE EXTRACTION UPDATES
**TrajFeatsGetter:**
- Coordinate normalization remains the same
- Time/velocity/acceleration calculations unchanged
- Only keyboard-specific features need updating

### 8. EMBEDDING LAYER ADJUSTMENTS
**Model Components to Resize:**
- WeightedSumEmbedding: n_elements from 37 to 30
- NearestEmbeddingWithPos: n_elements from 37 to 30  
- All embedding layers referencing character count
- Output linear layers: adjust output dimension to 28

## IMPLEMENTATION PRIORITY

### Phase 1: Core Character Set (Critical)
1. Update ns_tokenizers.py with English alphabet
2. Modify CHAR_VOCAB_SIZE throughout model.py
3. Update all config files with new n_classes

### Phase 2: Keyboard Layout (Critical)  
1. Create English QWERTY keyboard layout JSON
2. Update grid processing utilities
3. Remove Cyrillic character substitutions

### Phase 3: Model Architecture (Critical)
1. Adjust embedding dimensions
2. Update output layer sizes
3. Test model initialization with new dimensions

### Phase 4: Data & Training (Essential)
1. Prepare English training dataset
2. Create English vocabulary file
3. Update training configurations
4. Retrain models with English data

## CONSTRAINTS MAINTAINED
- Gesture recognition algorithm unchanged
- Trajectory feature extraction preserved  
- Model architecture type unchanged
- Training hyperparameters can remain similar