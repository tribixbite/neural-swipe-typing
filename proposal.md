# Proposal to Address Training Warnings

This document outlines a plan to address the three warnings identified during the startup of the training script. These changes aim to improve performance and align the project with modern PyTorch best practices.

---

### 1. Fix for `enable_nested_tensor` Warning

- **Warning:** `UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True`

- **Analysis:** The model currently uses a `(sequence_length, batch_size, features)` tensor format, but the Transformer layers can be optimized for a `(batch_size, sequence_length, features)` format by setting `batch_first=True`. This allows the use of a more efficient "nested tensor" representation.

- **Proposed Fix:** I will modify the model's architecture to use `batch_first=True` for better performance, which involves the following steps:
    1.  **Update Model Architecture:** In `src/model.py`, I will modify the `get_transformer_encoder_backbone_bigger__v3` and `get_transformer_decoder_backbone_bigger__v3` functions. I will add `batch_first=True` to the `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer` constructors.
    2.  **Update Data Collation:** In `src/dataset.py`, I will change the `CollateFnV2` to produce batch-first tensors by setting `batch_first=True` in its constructor.
    3.  **Verify Tensor Shapes:** I will review the `LitNeuroswipeModel` in `src/train_english.py` to ensure all tensor manipulations (reshaping, masking, etc.) correctly handle the new `(batch_size, seq_len, ...)` shape.

---

### 2. Fix for Tensor Cores `matmul_precision` Warning

- **Warning:** `You are using a CUDA device... that has Tensor Cores. To properly utilize them, you should set torch.set_float32_matmul_precision('medium' | 'high')`

- **Analysis:** This is a standard performance recommendation from PyTorch Lightning. The GPU has specialized hardware (Tensor Cores) that can dramatically speed up model training with a minor trade-off in floating-point precision.

- **Proposed Fix:** This is a straightforward, one-line change.
    1.  **Update Training Script:** At the beginning of the `main` function in `src/train_english.py`, I will add the following line:
        ```python
        torch.set_float32_matmul_precision('high')
        ```
        This enables Tensor Core acceleration with a good balance of performance and precision.

---

### 3. Fix for Deprecated `verbose` Parameter

- **Warning:** `The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.`

- **Analysis:** The `ReduceLROnPlateau` learning rate scheduler is being created with a `verbose=True` argument, which is no longer the recommended practice.

- **Proposed Fix:** This is a simple removal.
    1.  **Update Scheduler Creation:** In `src/train_english.py`, I will locate the `get_lr_scheduler` function.
    2.  I will remove the `verbose=True` argument from the `torch.optim.lr_scheduler.ReduceLROnPlateau(...)` call. PyTorch Lightning's logging system already provides learning rate visibility, making this parameter redundant.
