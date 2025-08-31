#!/usr/bin/env python3
"""
Evaluate the trained mobile model using ONLY autoregressive generation.
This provides realistic accuracy metrics for actual deployment conditions.
"""

import torch
import json
import os
from torch.utils.data import DataLoader
from train_mobile_model import MobileSwipeTrainer, SwipeDataset
from mobile_optimized_model import create_mobile_model
from collate_functions import pad_collate_fn

def load_vocabulary():
    """Load character vocabulary - should match training config (30 tokens)"""
    # Use the exact vocabulary that was used during training
    # Based on config: vocab_size = 30
    chars = list("abcdefghijklmnopqrstuvwxyz")  # 26 letters
    
    # Add special tokens (4 tokens)
    special_tokens = ['<pad>', '<eos>', '<unk>', '<sos>']
    vocab = special_tokens + chars  # 4 + 26 = 30 total
    
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    return char_to_idx, idx_to_char, len(vocab)

def evaluate_model_realistic(model_path: str, data_path: str, batch_size: int = 32):
    """
    Evaluate model using only autoregressive generation (no teacher forcing).
    This gives realistic deployment accuracy.
    """
    print("Loading model...")
    # Create model instance with correct vocab size from config
    model = create_mobile_model(vocab_size=30)
    
    # Load the trained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            # Remove 'model.' prefix from keys if present
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    state_dict[k[6:]] = v
                else:
                    state_dict[k] = v
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: {model_path} not found, using random weights")
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Loading dataset...")
    # Create dataset (SwipeDataset creates its own vocabulary)
    try:
        dataset = SwipeDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)
        print(f"Loaded {len(dataset)} samples from {data_path}")
        
        # Use the dataset's vocabulary (should match the training vocab)
        char_to_idx = dataset.char_to_idx
        idx_to_char = dataset.idx_to_char
        print(f"Dataset vocabulary size: {len(char_to_idx)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Evaluation metrics
    total_samples = 0
    correct_words = 0
    correct_tokens = 0
    total_tokens = 0
    
    print("Starting realistic evaluation (autoregressive generation only)...")
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(dataloader):
            features = features.to(device)
            targets = targets.to(device)
            
            # Generate predictions using autoregressive generation (no teacher forcing)
            generated = generate_autoregressive(model, features, char_to_idx, max_len=20)
            
            # Calculate word-level accuracy
            for i in range(features.size(0)):
                pred_word = tokens_to_word(generated[i], idx_to_char, char_to_idx)
                target_word = tokens_to_word(targets[i], idx_to_char, char_to_idx)
                
                if pred_word == target_word:
                    correct_words += 1
                
                # Token-level accuracy (CORRECTED)
                # 1. Get the ground truth tokens, excluding SOS and any PAD/EOS
                true_target_tokens = []
                for token in targets[i, 1:]: # Skip SOS
                    token_item = token.item()
                    if token_item in (char_to_idx['<pad>'], char_to_idx['<eos>']):
                        break
                    true_target_tokens.append(token_item)

                # 2. Get the predicted tokens, excluding SOS
                pred_tokens = generated[i, 1:].tolist()

                # 3. Compare against the ground truth length
                num_correct_in_sample = 0
                for j in range(len(true_target_tokens)):
                    if j < len(pred_tokens) and pred_tokens[j] == true_target_tokens[j]:
                        num_correct_in_sample += 1
                
                correct_tokens += num_correct_in_sample
                total_tokens += len(true_target_tokens)
                
                total_samples += 1
            
            if batch_idx % 10 == 0:
                word_acc = correct_words / total_samples if total_samples > 0 else 0
                token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0
                print(f"Batch {batch_idx}: Word Acc: {word_acc:.3f}, Token Acc: {token_acc:.3f}")
    
    # Final results
    final_word_acc = correct_words / total_samples if total_samples > 0 else 0
    final_token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0
    
    print("\n" + "="*50)
    print("REALISTIC EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {total_samples}")
    print(f"Word accuracy: {final_word_acc:.4f} ({correct_words}/{total_samples})")
    print(f"Token accuracy: {final_token_acc:.4f} ({correct_tokens}/{total_tokens})")
    print("="*50)
    
    return final_word_acc, final_token_acc

def generate_autoregressive(model, features, char_to_idx, max_len=20):
    """Generate word sequences autoregressively (no teacher forcing)"""
    batch_size = features.size(0)
    device = features.device
    
    # Start with SOS token
    generated = torch.full((batch_size, 1), char_to_idx['<sos>'], 
                          dtype=torch.long, device=device)
    
    for _ in range(max_len - 1):
        with torch.no_grad():
            logits = model(features, generated)
            next_token_logits = logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Stop if all sequences hit EOS
            if (next_tokens.squeeze() == char_to_idx['<eos>']).all():
                break
    
    return generated

def tokens_to_word(tokens, idx_to_char, char_to_idx):
    """Convert token sequence to word string"""
    word_chars = []
    for token_id in tokens:
        token_id = token_id.item()
        if token_id == char_to_idx['<eos>'] or token_id == char_to_idx['<pad>']:
            break
        elif token_id == char_to_idx['<sos>'] or token_id == char_to_idx['<unk>']:
            continue
        else:
            if token_id in idx_to_char:
                word_chars.append(idx_to_char[token_id])
    return ''.join(word_chars)

def main():
    # Find the best checkpoint
    checkpoint_dir = "checkpoints/mobile_model"
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.ckpt'):
                checkpoints.append(os.path.join(checkpoint_dir, f))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    # Use the most recent checkpoint
    model_path = max(checkpoints, key=os.path.getmtime)
    print(f"Using checkpoint: {model_path}")
    
    # Evaluate on validation set
    val_data_path = "data/combined_dataset/cleaned_english_swipes_val.jsonl"
    if not os.path.exists(val_data_path):
        print(f"Validation data not found at {val_data_path}")
        print("Please make sure the dataset is prepared correctly.")
        return
    
    print("Evaluating model with REALISTIC metrics (autoregressive generation only)...")
    word_acc, token_acc = evaluate_model_realistic(model_path, val_data_path, batch_size=16)
    
    if word_acc < 0.7:  # 70% threshold
        print(f"\n⚠️  WARNING: Word accuracy {word_acc:.1%} is below 70% target")
        print("The model may need more training or architecture improvements.")
    else:
        print(f"\n✅ SUCCESS: Word accuracy {word_acc:.1%} exceeds 70% target")

if __name__ == "__main__":
    main()