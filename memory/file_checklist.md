# File Modification Checklist for Cyrillic → English Adaptation

## CORE TRAINING & GENERATION FILES

### Critical Code Files (Need Modification)
- [ ] ⭐ `src/ns_tokenizers.py` - **CRITICAL**: Contains Cyrillic alphabet, KeyboardTokenizerv1
- [ ] ⭐ `src/model.py` - **CRITICAL**: CHAR_VOCAB_SIZE constants, embedding dimensions
- [ ] ⭐ `src/grid_processing_utils.py` - **CRITICAL**: Cyrillic substitutions, key processing
- [ ] ⭐ `src/feature_extraction/feature_extractors.py` - Uses Cyrillic alphabet constant
- [ ] ⭐ `src/predict_v2.py` - References n_classes in configs
- [ ] ⭐ `src/train.ipynb` - Training notebook with num_classes=35 hardcoded
- [ ] ⭐ `src/word_generation_demo.ipynb` - Demo notebook with n_classes=35

### Configuration Files (Need Modification) 
- [ ] ⭐ `configs/config__my_weighted_features.json` - n_classes: 35
- [ ] ⭐ `configs/config__google_2015_features.json` - n_classes: 35  
- [ ] ⭐ `configs/config__my_nearest_features.json` - n_classes: 35
- [ ] ⭐ `configs/config__phrase_swipe_features.json` - n_classes: 35
- [ ] ⭐ `configs/config__indiswipe_features.json` - n_classes: 35
- [ ] ⭐ `configs/config-yandex-cup.json` - n_classes: null (needs updating)

### Data Files (Need Replacement)
- [ ] ⭐ `data/data_preprocessed/gridname_to_grid.json.dvc` - **CRITICAL**: Keyboard layout with Cyrillic labels
- [ ] ⭐ `data/data_preprocessed/voc.txt.dvc` - **CRITICAL**: Cyrillic vocabulary file
- [ ] ⭐ Training datasets (will need English equivalents)

### Support Files (Minor/No Changes)
- [ ] `src/dataset.py` - Dataset loading (no language-specific content)
- [ ] `src/word_generators_v2.py` - Word generation logic (language agnostic)
- [ ] `src/logit_processors.py` - Logit processing (language agnostic)  
- [ ] `src/metrics.py` - Evaluation metrics (language agnostic)
- [ ] `src/pl_module.py` - PyTorch Lightning module (minor changes needed)

### Feature Extraction Files (No Changes)
- [ ] `src/feature_extraction/nearest_key_lookup.py` - Key proximity (algorithm only)
- [ ] `src/feature_extraction/distances_lookup.py` - Distance calculations (algorithm only)
- [ ] `src/feature_extraction/nearest_key_lookup_optimized.py` - Optimized lookups (algorithm only)

### Utility Files (No Changes)
- [ ] `src/evaluate.py` - Evaluation scripts (language agnostic)
- [ ] `src/utils/ckpt_to_pt.py` - Model conversion utilities
- [ ] `src/utils/delete_duplicates_stable.py` - Data utilities
- [ ] `src/downloaders/download_weights.py` - Weight download utilities

## ADAPTATION PRIORITY CHECKLIST

### Phase 1: Character Set (CRITICAL - Must Complete First)
- [ ] ⭐ Update `src/ns_tokenizers.py`: Replace ALL_CYRILLIC_LETTERS_ALPHABET_ORD
- [ ] ⭐ Update `src/ns_tokenizers.py`: Modify KeyboardTokenizerv1.i2t for English  
- [ ] ⭐ Update `src/model.py`: Change CHAR_VOCAB_SIZE from 37 to 30
- [ ] ⭐ Update `src/model.py`: Change n_classes references from 35 to 28
- [ ] ⭐ Update all config JSON files: "n_classes": 35 → 28

### Phase 2: Model Architecture (CRITICAL)
- [ ] ⭐ Update `src/model.py`: All embedding n_elements from 37 to 30
- [ ] ⭐ Update `src/model.py`: All n_keys references from 37 to 30
- [ ] ⭐ Update `src/train.ipynb`: num_classes from 35 to 28
- [ ] ⭐ Test model initialization with new dimensions

### Phase 3: Keyboard Layout (CRITICAL)
- [ ] ⭐ Create new English QWERTY `gridname_to_grid.json`
- [ ] ⭐ Update `src/grid_processing_utils.py`: Remove Cyrillic substitutions
- [ ] ⭐ Update key center calculations for English layout

### Phase 4: Data & Vocabulary (ESSENTIAL)
- [ ] ⭐ Create English `voc.txt` vocabulary file
- [ ] ⭐ Generate English training dataset (or use existing if available)
- [ ] ⭐ Update training data paths in configs and notebooks

## VERIFICATION CHECKLIST
- [ ] All hardcoded "37" changed to "30" in model files
- [ ] All hardcoded "35" changed to "28" in config files  
- [ ] All Cyrillic character references removed
- [ ] English alphabet properly integrated
- [ ] QWERTY keyboard layout properly defined
- [ ] Model can initialize without errors
- [ ] Training pipeline can load data without errors