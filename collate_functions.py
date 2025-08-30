"""
Custom collate functions for handling variable-length sequences in mobile swipe training.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple

def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    features, targets = zip(*batch)
    
    # Get maximum sequence length in this batch
    max_seq_len = max(f.size(0) for f in features)
    max_target_len = max(t.size(0) for t in targets)
    
    # Pad feature sequences
    padded_features = []
    for f in features:
        seq_len, feature_dim = f.size()
        if seq_len < max_seq_len:
            # Pad with zeros
            padding = torch.zeros(max_seq_len - seq_len, feature_dim)
            f_padded = torch.cat([f, padding], dim=0)
        else:
            f_padded = f
        padded_features.append(f_padded)
    
    # Pad target sequences  
    padded_targets = []
    for t in targets:
        target_len = t.size(0)
        if target_len < max_target_len:
            # Pad with 0 (padding token)
            padding = torch.zeros(max_target_len - target_len, dtype=t.dtype)
            t_padded = torch.cat([t, padding], dim=0)
        else:
            t_padded = t
        padded_targets.append(t_padded)
    
    # Stack into batch tensors
    features_batch = torch.stack(padded_features, dim=0)
    targets_batch = torch.stack(padded_targets, dim=0)
    
    return features_batch, targets_batch