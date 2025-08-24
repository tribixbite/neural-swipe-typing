# Analysis and Recommendations for English Adaptation

## 1. Executive Summary

The project to adapt the Cyrillic-based swipe model for English is on a solid trajectory. The phased approach outlined in `pm.md` is logical, and the data preparation and filtering steps undertaken were critical for success. The core strategy of reusing the transformer architecture while replacing data-specific components (keyboard, vocabulary) is sound.

However, a review of the project documents and configuration files reveals several critical issues and inconsistencies that need to be addressed. The most significant concern is a fundamental mismatch between the model's expected input size and the actual English keyboard definition.

This analysis provides specific, actionable recommendations to correct these issues, improve data quality, and ensure project documentation is synchronized.

## 2. Critical Concern: Keyboard Definition and Model Input Size

There is a major discrepancy regarding the number of keyboard keys, which directly impacts the model's architecture.

- **The Finding:** `ARCHITECTURE_OVERVIEW.md` states the model's `Swipe Point Embedder` handles 122 keys. However, the `data/data_preprocessed/gridname_to_grid_english.json` file defines only **34 keys**.
- **The Impact:** If the model architecture has a hardcoded input layer sized for 122 keys, it is severely oversized for the 34-key English layout. This leads to a large number of unused parameters, increasing model size and memory consumption, slowing down training and inference, and potentially hindering learning.
- **Recommendation:**
    1.  **Verify Model Configuration:** Immediately check the model's input layer definition in `src/model.py` and the parameters within `configs/config_english.json` and `configs/config_english_filtered.json`.
    2.  **Correct Input Dimension:** Ensure the model's input dimension for key embeddings matches the actual number of keys defined in `gridname_to_grid_english.json` (i.e., 34).
    3.  **Update Documentation:** Update the `ARCHITECTURE_OVERVIEW.md` to reflect the correct number of keys for the English model.

## 3. Data Integrity Issue: Overlapping Keys in Keyboard Layout

The English keyboard definition itself contains errors that will confuse the model.

- **The Finding:** In `gridname_to_grid_english.json`, several distinct characters share the exact same hitbox coordinates, effectively making them indistinguishable to the model based on spatial features.
    - `'m'` and `.`
    - `'n'` and `,`
    - `'p'` and `-`
- **The Impact:** The model will receive identical swipe information for different target characters, making it impossible to learn to differentiate between them correctly. This will degrade accuracy for words containing these characters.
- **Recommendation:**
    1.  **Fix Generation Script:** Identify and fix the script used to generate `gridname_to_grid_english.json`.
    2.  **Regenerate Layout:** Regenerate the file with corrected, non-overlapping hitboxes for all keys.
    3.  **Retrain if Necessary:** If training has already commenced with the faulty file, the model should be retrained once the corrected layout is in place.

## 4. Documentation Inconsistencies

The project's two main documents, `pm.md` and `ARCHITECTURE_OVERVIEW.md`, are out of sync.

- **The Finding:** There are conflicting statistics for key project metrics:
    - **Dataset Size:** `pm.md` states the filtered dataset has **74,021** swipes, while `ARCHITECTURE_OVERVIEW.md` claims **87,166** (which is the *unfiltered* size).
    - **Vocabulary Size:** `pm.md` reports **40** tokens in the filtered vocabulary, while `ARCHITECTURE_OVERVIEW.md` reports **37**.
- **The Impact:** Inconsistent documentation leads to confusion and makes it difficult to track progress or understand the current state of the project.
- **Recommendation:**
    1.  **Establish Source of Truth:** Use `pm.md` as the source of truth, as it appears to be more frequently updated with detailed progress.
    2.  **Synchronize Documents:** Update `ARCHITECTURE_OVERVIEW.md` to reflect the latest, post-filtering statistics for dataset size and vocabulary.

## 5. Suggestion: Improve Vocabulary and Dataset Filtering

The current method for filtering the English dataset can be significantly improved.

- **The Finding:** The dataset was filtered using `en.txt`, a list of 10,000 common English words. While this was a good step to clean the initial noisy data, it is not comprehensive.
- **The Impact:** Many valid but less common English words were likely discarded, unnecessarily reducing the size and richness of the training data from 87k samples to 74k.
- **Recommendation:**
    1.  **Use a Larger Dictionary:** Re-filter the original, unfiltered `english_full_*.jsonl` datasets using a more comprehensive English word frequency list.
    2.  **Suggested Sources:** Consider using resources like SCOWL (Spell-Checking Oriented Word Lists), Google's Web Trillion Word Corpus, or other large-scale linguistic datasets to create a more robust reference dictionary. This will help retain more valid training samples and improve the model's vocabulary.

## 6. Suggestion: Clarify Strategy for Contextual Prediction

The project goal includes contextual prediction, but this is not reflected in the current architecture or near-term plan.

- **The Finding:** `pm.md` states a goal to predict words from "swipe traces (and previous words)," but `ARCHITECTURE_OVERVIEW.md` correctly lists the absence of this feature as a current limitation.
- **The Impact:** Contextual prediction is a major feature that requires significant architectural changes. It is not a trivial addition and needs to be planned for.
- **Recommendation:**
    1.  **Update Project Plan:** Add a new, distinct phase to `pm.md` for "Context-Aware Prediction."
    2.  **Define Tasks:** This new phase should include tasks such as:
        - Researching methods for incorporating context (e.g., modifying the decoder, adding a context encoder).
        - Updating the model architecture in `src/model.py`.
        - Modifying the `Dataset` and data pipeline to provide previous word context.
        - Training and evaluating the new context-aware model.
