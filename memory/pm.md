# Project Management: English Swipe Typing Transformer

## Project Goal
Apply the best pipeline from this neural-swipe-typing repository to English language to create a working swipe transformer that can predict words from swipe traces (and previous words), then deploy it in an Android keyboard application.

## Current Status Summary

### ✅ Completed
- Repository refactoring is largely complete with modularized components:
  - `refactor-model-module`: Swipe point embedders separated
  - `refactor-feature-extraction`: Feature processing modularized
  - `refactor-eval-and-predict`: Prediction pipeline improved
- Logit processors integrated for vocabulary masking during beam search
- Best model identified: `v3_weighted_and_traj_transformer_bigger`
- Training pipeline established using PyTorch Lightning
- Yandex Cup Russian dataset pipeline working
- ExecutorTorch export pipeline exists in `executorch-investigation` branch (needs merging)

### 🔄 In Progress
- English dataset exists in `data/swipelogs/` but needs conversion
- No English keyboard layout defined in system yet

### ❌ Not Started
- English dataset conversion pipeline
- English keyboard layout configuration
- English vocabulary preparation
- Training on English data
- English model evaluation

## Detailed Step-by-Step Plan

### Phase 1: Data Preparation 🚀 [PRIORITY]

#### 1.1 Analyze English Swipelog Format ✅
- [x] Study structure of `.log` files (touch events with x,y,t coordinates)
- [x] Study structure of `.json` files (user metadata)
- [x] Document the mapping between log events and expected training format
- [x] Identify sentence boundaries and word segmentation in logs

#### 1.2 Create English Keyboard Layout ✅
- [x] Define QWERTY layout in grid format matching existing structure
- [x] Create `add_english_keyboard.py` script
- [x] Specify key positions, dimensions, and characters
- [x] Name it "qwerty_english"
- [x] Created standalone grid file at `data/data_preprocessed/gridname_to_grid_english.json`

#### 1.3 Build Swipelog Conversion Script ✅
- [x] Create `src/data_obtaining_and_preprocessing/convert_swipelogs_to_jsonl.py`
- [x] Parse touch events from `.log` files
- [x] Group events by word boundaries
- [x] Extract x, y, t coordinates for each swipe
- [x] Match target words from sentence context
- [x] Output in JSONL format: `{"word": "target", "curve": {"x": [...], "y": [...], "t": [...], "grid_name": "qwerty_english"}}`
- [x] Test on 5 sample files - extracted 235 swipes successfully

#### 1.4 Process English Dataset ✅
- [x] Run conversion script on all swipelogs (1338 files)
- [x] Split into train/validation/test sets (80/10/10)
- [x] Filter out erroneous swipes (marked with `is_err=1`)
- [x] Extracted 87,166 total swipes
- [x] Save as:
  - `data/data_preprocessed/english_full_train.jsonl` (69,732 samples)
  - `data/data_preprocessed/english_full_valid.jsonl` (8,716 samples)
  - `data/data_preprocessed/english_full_test.jsonl` (8,718 samples)

#### 1.5 Prepare English Vocabulary ✅
- [x] Extract unique words from training data (10,638 unique words)
- [x] Create character vocabulary file (61 unique characters)
- [x] Save as `data/data_preprocessed/voc_english.txt`
- [x] Ensure special tokens included: `<sos>`, `<eos>`, `<pad>`, `<unk>`

### Phase 2: Model Configuration ✅

#### 2.1 Create English Configuration ✅
- [x] Copy `configs/config__my_weighted_features.json`
- [x] Create `configs/config_english.json`
- [x] Update paths to English datasets
- [x] Set `grid_name: "qwerty_english"`
- [x] Configure hyperparameters (lr=1e-4, batch=256/512, smoothing=0.045)

#### 2.2 Update Training Script ✅
- [x] Create standalone `src/train_english.py` script
- [x] Add English dataset paths configuration
- [x] Set appropriate batch sizes (train=256, val=512)
- [x] Configure checkpoint naming for English model
- [x] Include all necessary imports and Lightning setup

### Phase 3: Training

#### 3.1 Initial Training Run
- [ ] Start with small subset (1000 samples) for debugging
- [ ] Verify data loading and feature extraction works
- [ ] Check model forward pass with English data
- [ ] Monitor memory usage and adjust batch size

#### 3.2 Full Training
- [ ] Train on complete English dataset
- [ ] Use best hyperparameters from Russian model as starting point:
  - Learning rate: 1e-4
  - Batch size: 256 (train), 512 (val)
  - Label smoothing: 0.045
  - Dropout: 0.2
- [ ] Enable early stopping (patience=35)
- [ ] Save checkpoints regularly

#### 3.3 Training Monitoring
- [ ] Track metrics via TensorBoard:
  - Train/Val loss
  - Word-level accuracy
  - Token-level accuracy
  - Learning rate schedule
- [ ] Monitor for overfitting
- [ ] Adjust hyperparameters if needed

### Phase 4: Evaluation

#### 4.1 Model Evaluation
- [ ] Run prediction on test set using `predict_v2.py`
- [ ] Calculate metrics:
  - Swipe MRR (Mean Reciprocal Rank)
  - Top-1, Top-3, Top-5 accuracy
  - Character error rate
- [ ] Compare with baseline methods

#### 4.2 Qualitative Analysis
- [ ] Test on common English phrases
- [ ] Analyze failure cases
- [ ] Test different decoding strategies:
  - Greedy search
  - Beam search (various beam sizes)
  - With/without vocabulary masking

### Phase 5: Android Deployment Preparation

#### 5.1 Merge ExecutorTorch Branch ✅
- [x] Review `executorch-investigation` branch compatibility
- [x] Merge branch into main (contains `executorch_export.ipynb`)
- [x] No conflicts - clean merge
- [ ] Test export pipeline with trained English model

#### 5.2 Export Model for Android
- [ ] Install ExecutorTorch 0.5 with XNNPACK backend
- [ ] Export trained English model to .pte format using `executorch_export.ipynb`
- [ ] Optimize model size (target <60MB for Android keyboard)
- [ ] Remove dropout layers from exported model
- [ ] Test exported model inference speed

#### 5.3 Android Integration Requirements
- [ ] Model must fit in ~60MB memory budget (Android keyboard limit)
- [ ] Inference time <50ms per swipe for real-time performance
- [ ] Support for dynamic vocabulary updates
- [ ] Handle multiple keyboard layouts (portrait/landscape)

### Phase 6: Optimization & Final Deployment

#### 6.1 Model Optimization
- [ ] Experiment with different architectures if needed
- [ ] Try different feature combinations
- [ ] Optimize inference speed
- [ ] Consider model quantization for deployment

#### 6.2 Integration Testing
- [ ] Test with `word_generation_demo.ipynb`
- [ ] Verify beam search with English vocabulary
- [ ] Test real-time inference performance
- [ ] Test .pte model in Android environment

#### 6.3 Documentation
- [ ] Update README with English model results
- [ ] Document English dataset statistics
- [ ] Create usage examples for English
- [ ] Update CLAUDE.md with English-specific instructions

## Critical Path Items (Must Do First)

1. **Create swipelog conversion script** - Without this, no training data
2. **Define English keyboard layout** - Required for feature extraction
3. **Convert at least 1000 samples** - Minimum for testing pipeline
4. **Verify data format** - Ensure compatibility with existing code
5. **Merge executorch-investigation branch** - Needed for Android deployment

## Risk Mitigation

### Potential Issues & Solutions

1. **Swipelog format complexity**
   - Solution: Start with simplest cases, incrementally handle edge cases
   - Fallback: Use only clean, unambiguous swipes initially

2. **Memory constraints during training**
   - Solution: Use gradient accumulation for larger effective batch sizes
   - Solution: Set DataLoader workers=0 as documented

3. **Different keyboard dimensions**
   - Solution: Normalize coordinates to standard range
   - Solution: Use relative positions instead of absolute

4. **Word boundary detection in logs**
   - Solution: Use timestamp gaps between swipes
   - Solution: Cross-reference with sentence structure

5. **Android memory limits (60MB for keyboard)**
   - Solution: Use model quantization (int8 instead of float32)
   - Solution: Reduce model size (fewer layers/dimensions)
   - Solution: Use ExecutorTorch's XNNPACK optimizations

6. **Inference speed on mobile**
   - Solution: Cache frequent predictions
   - Solution: Use smaller beam size for mobile
   - Solution: Implement early stopping in beam search

## Success Metrics

- [ ] Successfully convert >10,000 English swipes
- [ ] Achieve >70% top-1 word accuracy on validation set
- [ ] Achieve >85% top-3 word accuracy on validation set
- [ ] Inference speed <50ms per swipe on Android device
- [ ] Model size <60MB (Android keyboard memory limit)
- [ ] Successful .pte export via ExecutorTorch
- [ ] Working proof-of-concept in Android app

## Next Immediate Actions

1. **TODAY**: Create and test swipelog parser for 10 sample files
2. **TODAY**: Define English QWERTY layout in grid format
3. **TOMORROW**: Complete conversion script and process 1000 samples
4. **THIS WEEK**: Complete Phase 1 (Data Preparation)
5. **NEXT WEEK**: Complete Phase 2-3 (Configuration & Training)

## Notes for Future Sessions

- Always check this file first for current status
- Update checkboxes as tasks complete
- Document any blockers or changes in approach
- Keep track of experiment results and hyperparameters
- Note any discovered issues with English data

## Command Reference

```bash
# Merge executorch branch
git checkout main
git merge origin/executorch-investigation

# Data conversion (to be created)
python src/data_obtaining_and_preprocessing/convert_swipelogs_to_jsonl.py \
  --input_dir data/swipelogs/ \
  --output_path data/data_preprocessed/train_english.jsonl \
  --keyboard_layout qwerty_english

# Training
python src/train.py --config configs/config_english.json

# Prediction
python src/predict_v2.py --config configs/config_english.json --num-workers 0

# Evaluation
python src/evaluate.py --predictions_path results/predictions/test/english/*.pkl

# Export to ExecutorTorch (after training)
jupyter notebook src/executorch_export.ipynb
# Follow instructions in notebook to export .pte file
```

## Android Deployment Details (from executorch-investigation branch)

### ExecutorTorch Export Process
The `executorch-investigation` branch contains `src/executorch_export.ipynb` which:
- Exports PyTorch models to .pte format for mobile deployment
- Uses ExecutorTorch 0.5 with XNNPACK backend for optimization
- Handles model conversion for Android runtime

### Android App Requirements
Based on the notebook documentation:
1. ExecutorTorch 0.5 installation with XNNPACK
2. Model in .pte format placed in `app/src/main/assets/`
3. ExecutorTorch.aar library in `app/libs/`
4. Dependencies in `build.gradle.kts`:
   - `com.facebook.fbjni:fbjni:0.5.1`
   - `com.facebook.soloader:soloader:0.10.5`

### TODO from ExecutorTorch notebook
- Remove dropout from exported models
- Create FullEncoder modules
- Verify unused model fields don't affect export

---
Last Updated: 2024-11-24
Status: Phases 1-2 COMPLETE! Dataset processed (87K swipes), configuration ready, training script created. Ready to start training!