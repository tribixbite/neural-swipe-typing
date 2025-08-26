
# Model Reconstruction Script
import torch
from model import MODEL_GETTERS_DICT

def load_neural_swipe_model(state_dict_path, device='cpu'):
    """Load the neural swipe typing model from state dict."""
    
    # Model parameters (from training)
    model_params = {'n_coord_feats': 6, 'n_keys': 29, 'vocab_size': 32, 'max_word_len': 30}
    
    # Create model architecture
    model = MODEL_GETTERS_DICT["v3_weighted_and_traj_transformer_bigger"](**model_params)
    
    # Load trained weights
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model.eval()

def predict_swipe(model, trajectory_features, keyboard_weights):
    """
    Run inference on swipe data.
    
    Args:
        model: Loaded neural swipe model
        trajectory_features: [seq_len, 1, 6] tensor 
        keyboard_weights: [seq_len, 1, 29] tensor
        
    Returns:
        encoded: [seq_len, 1, 128] encoded sequence
    """
    with torch.no_grad():
        encoder_input = (trajectory_features, keyboard_weights)
        encoded = model.encode(encoder_input, src_key_padding_mask=None)
        return encoded

# Usage example:
if __name__ == "__main__":
    # Load model
    model = load_neural_swipe_model("model_state_dict.pt")
    
    # Create sample input
    seq_len = 10
    traj_feats = torch.randn(seq_len, 1, 6)
    kb_weights = torch.softmax(torch.randn(seq_len, 1, 29), dim=-1)
    
    # Run prediction
    encoded = predict_swipe(model, traj_feats, kb_weights)
    print(f"Encoded shape: {encoded.shape}")
