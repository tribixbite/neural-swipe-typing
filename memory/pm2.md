# Project Management

## Current Task: Adapt Neural Swipe Typing from Cyrillic to English

### COMPLETED Analysis - Language-Specific Components Found:

## 1. KEYBOARD DIMENSIONS REFERENCES ✅
**Location**: `src/feature_extraction/feature_extractors.py:147-149`
- `TrajFeatsGetter.__call__()` normalizes coordinates by keyboard width/height
- Uses `self.grid_name_to_wh[grid_name]` to get keyboard dimensions
- X coordinates divided by width, Y coordinates divided by height

## 2. KEYBOARD CHARACTERS USAGE ✅
**Primary Location**: `src/ns_tokenizers.py:4-8`
```python
ALL_CYRILLIC_LETTERS_ALPHABET_ORD = [
    'а', 'б', 'в', 'г', 'д', 'е', 'ë', 'ж', 'з', 'и', 'й',
    'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
    'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я'
]
```
**Secondary Locations**:
- `KeyboardTokenizerv1.i2t` (line 69): Uses Cyrillic alphabet + ['-', '<unk>', '<pad>']
- `src/grid_processing_utils.py:27`: Character substitutions {'ъ': 'ь', 'ё': 'е'}

## 3. GENERATOR CHARACTER LIST ✅
**Model Architecture Constants** - `src/model.py`:
- `CHAR_VOCAB_SIZE = 37` (lines 427, 474, 785, 820, 855)
- `n_classes = 35` (CHAR_VOCAB_SIZE - 2, excluding <sos> and <pad>)
- `n_keys = 37` for keyboard tokenizer size
- Config files: `"n_classes": 35` in all config JSONs

## 4. WORDLISTS (INPUT/OUTPUT) ✅
**Vocabulary File**: `data/data_preprocessed/voc.txt.dvc`
**Tokenizer Build**: `src/ns_tokenizers.py:34-43`
- `CharLevelTokenizerv2._build_vocab()` reads vocabulary from file
- Creates character mappings from all unique characters in vocabulary
- Special tokens: ["<eos>", "<unk>", "<pad>", "<sos>"]

## 5. TIMING MEASUREMENTS/CALCULATIONS ✅
**Location**: `src/feature_extraction/feature_extractors.py:76-105`
- `get_dx_dt()` function calculates velocity: `dx_dt[i] = (X[i+1] - X[i-1]) / (T[i+1] - T[i-1])`
- `TrajFeatsGetter` includes timing features if `include_time=True`
- Time is normalized like other trajectory features

## 6. VELOCITY MEASUREMENTS/CALCULATIONS ✅
**Location**: `src/feature_extraction/feature_extractors.py:132-135`
- `TrajFeatsGetter.__call__()`: Calculates `dx_dt` and `dy_dt` using `get_dx_dt()`
- Velocity included when `include_velocities=True`
- Uses same timing-based derivative calculation

## 7. ACCELERATION MEASUREMENTS/CALCULATIONS ✅
**Location**: `src/feature_extraction/feature_extractors.py:137-140`
- Calculates second derivative: `d2x_dt2 = get_dx_dt(dx_dt, T)`
- Applied to both X and Y velocity components
- Included when `include_accelerations=True`

## 8. MAXIMUM/MINIMUM CHARACTER LIMITS ✅
**Sequence Length Constants**:
- `MAX_CURVES_SEQ_LEN = 299` (input sequence length)
- `MAX_OUT_SEQ_LEN = 35` (word character tokenizer.max_word_len - 1)
- `max_steps_n=35` default in word generators
- Training notebook: `num_classes = 35` (vocab_size - 2)

## 9. ADDITIONAL LANGUAGE-SPECIFIC FINDINGS ✅
**Keyboard Layout Dependencies**:
- `gridname_to_grid.json` contains keyboard layouts with Cyrillic labels
- Key center calculations in various model variants
- Keyboard scalers: `KB_X_SCALER = lambda x: x/1080`, `KB_Y_SCALER = lambda x: x/667`

**Data Paths**:
- Training data paths reference Cyrillic-specific datasets
- Grid substitution logic for missing Cyrillic characters

## Next Steps:
1. Create English adaptation plan
2. Implement character set changes
3. Update keyboard layouts
4. Retrain models with English data