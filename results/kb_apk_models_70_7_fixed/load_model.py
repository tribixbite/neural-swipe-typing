#!/usr/bin/env python3
"""Load the exported 0.707 accuracy model."""

import torch
from model import MODEL_GETTERS_DICT

def load_model(device='cpu'):
    """Load the 0.707 accuracy neural swipe typing model."""
    
    model_params = {'n_coord_feats': 6, 'n_keys': 29, 'vocab_size': 32, 'max_word_len': 30}
    model = MODEL_GETTERS_DICT["v3_weighted_and_traj_transformer_bigger"](**model_params)
    
    state_dict = torch.load("model_state_dict.pt", map_location=device)
    model.load_state_dict(state_dict)
    
    return model.eval()

def predict(model, trajectory_features, keyboard_weights):
    """Run prediction on swipe data."""
    with torch.no_grad():
        encoder_input = (trajectory_features, keyboard_weights)
        encoded = model.encode(encoder_input, None)
        return encoded

if __name__ == "__main__":
    model = load_model()
    print(f"Model loaded successfully!")
    print(f"Accuracy: 0.707")
    print(f"Vocab size: 32")
    print(f"Model size: 5.1MB")
