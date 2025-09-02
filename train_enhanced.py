#!/usr/bin/env python3
"""
Enhanced training script replicating the original model's 70% accuracy.
Uses character-level prediction with nearest key features.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

# Import enhanced model
from mobile_model_enhanced import create_enhanced_mobile_model
from src.ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizerv1
from src.feature_extraction.nearest_key_lookup import NearestKeyLookup


class EnhancedSwipeDataset(Dataset):
    """Dataset with character-level targets and nearest key features."""
    
    def __init__(self,
                 data_path: str,
                 char_tokenizer: CharLevelTokenizerv2,
                 kb_tokenizer: KeyboardTokenizerv1,
                 nearest_key_lookup: Optional[NearestKeyLookup] = None,
                 max_seq_len: int = 150,
                 max_word_len: int = 20):
        self.data_path = data_path
        self.char_tokenizer = char_tokenizer
        self.kb_tokenizer = kb_tokenizer
        self.nearest_key_lookup = nearest_key_lookup
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        
        # Fixed keyboard dimensions for English
        self.width = 360
        self.height = 215
        
        # Load data
        self.data = []
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Only include samples with words for training
                if 'word' in item and item['word']:
                    self.data.append(item)
        
        print(f"Loaded {len(self.data)} samples")
        
        # If no lookup provided, create a simple one
        if self.nearest_key_lookup is None:
            print("Creating nearest key lookup...")
            self._create_simple_nearest_key_lookup()
    
    def _create_simple_nearest_key_lookup(self):
        """Create a simple nearest key lookup based on keyboard layout."""
        # Simple QWERTY layout approximation
        self.key_positions = {
            'q': (30, 50), 'w': (66, 50), 'e': (102, 50), 'r': (138, 50), 't': (174, 50),
            'y': (210, 50), 'u': (246, 50), 'i': (282, 50), 'o': (318, 50), 'p': (354, 50),
            'a': (48, 107), 's': (84, 107), 'd': (120, 107), 'f': (156, 107), 'g': (192, 107),
            'h': (228, 107), 'j': (264, 107), 'k': (300, 107), 'l': (336, 107),
            'z': (84, 164), 'x': (120, 164), 'c': (156, 164), 'v': (192, 164), 'b': (228, 164),
            'n': (264, 164), 'm': (300, 164)
        }
    
    def _get_nearest_key(self, x: float, y: float) -> str:
        """Get nearest keyboard key for a coordinate."""
        if hasattr(self, 'nearest_key_lookup') and self.nearest_key_lookup:
            return self.nearest_key_lookup.get_nearest_kb_label(int(x), int(y))
        
        # Simple distance-based lookup
        min_dist = float('inf')
        nearest_key = 'a'
        
        for key, (kx, ky) in self.key_positions.items():
            dist = (x - kx) ** 2 + (y - ky) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_key = key
        
        return nearest_key
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract coordinates and timestamps
        x_coords = np.array(item['curve']['x'], dtype=np.float32)
        y_coords = np.array(item['curve']['y'], dtype=np.float32)
        t_coords = np.array(item['curve']['t'], dtype=np.float32)
        
        # Get nearest key for each point
        nearest_keys = []
        for x, y in zip(x_coords, y_coords):
            key = self._get_nearest_key(x, y)
            nearest_keys.append(self.kb_tokenizer.get_token(key))
        
        # Normalize coordinates
        x_norm = x_coords / self.width
        y_norm = y_coords / self.height
        
        # Compute velocities and accelerations
        if len(x_coords) > 1:
            # Time differences in seconds
            dt = np.diff(t_coords) / 1000.0
            dt[dt == 0] = 0.001  # Avoid division by zero
            
            # Velocities
            vx = np.diff(x_norm) / dt
            vy = np.diff(y_norm) / dt
            
            # Clip to reasonable range
            vx = np.clip(vx, -5, 5)
            vy = np.clip(vy, -5, 5)
            
            # Pad to match position length
            vx = np.concatenate([[0], vx])
            vy = np.concatenate([[0], vy])
            
            # Accelerations
            if len(vx) > 1:
                dt_v = np.diff(t_coords) / 1000.0
                dt_v[dt_v == 0] = 0.001
                
                ax = np.diff(vx) / dt_v
                ay = np.diff(vy) / dt_v
                
                ax = np.clip(ax, -5, 5)
                ay = np.clip(ay, -5, 5)
                
                ax = np.concatenate([[0], ax])
                ay = np.concatenate([[0], ay])
            else:
                ax = np.zeros_like(x_norm)
                ay = np.zeros_like(y_norm)
        else:
            vx = np.zeros_like(x_norm)
            vy = np.zeros_like(y_norm)
            ax = np.zeros_like(x_norm)
            ay = np.zeros_like(y_norm)
        
        # Stack trajectory features
        traj_features = np.stack([x_norm, y_norm, vx, vy, ax, ay], axis=1)
        
        # Handle NaN values
        if np.any(np.isnan(traj_features)):
            traj_features = np.nan_to_num(traj_features, 0.0)
        
        # Pad or truncate sequences
        seq_len = traj_features.shape[0]
        if seq_len > self.max_seq_len:
            traj_features = traj_features[:self.max_seq_len]
            nearest_keys = nearest_keys[:self.max_seq_len]
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        else:
            # Pad trajectory features
            padding = np.zeros((self.max_seq_len - seq_len, 6), dtype=np.float32)
            traj_features = np.concatenate([traj_features, padding], axis=0)
            
            # Pad nearest keys with <pad> token
            pad_token = self.kb_tokenizer.get_token('<pad>')
            nearest_keys = nearest_keys + [pad_token] * (self.max_seq_len - seq_len)
            
            # Create mask (True for padded positions)
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            mask[seq_len:] = True
        
        # Convert to tensors
        traj_tensor = torch.from_numpy(traj_features).float()
        kb_tensor = torch.tensor(nearest_keys, dtype=torch.long)
        
        # Tokenize target word (character-level)
        target_word = item.get('word', '').lower()
        char_tokens = self.char_tokenizer.encode(target_word)
        
        # Prepare decoder input/output
        # Input: <sos> + word chars
        # Output: word chars + <eos>
        if len(char_tokens) > self.max_word_len:
            char_tokens = char_tokens[:self.max_word_len]
        
        # Pad character sequence
        pad_idx = self.char_tokenizer.char_to_idx['<pad>']
        while len(char_tokens) < self.max_word_len:
            char_tokens.append(pad_idx)
        
        decoder_input = torch.tensor(char_tokens[:-1], dtype=torch.long)
        decoder_output = torch.tensor(char_tokens[1:], dtype=torch.long)
        
        # Create decoder mask (True for padded positions)
        decoder_mask = decoder_input == pad_idx
        
        return {
            'traj_features': traj_tensor,
            'kb_features': kb_tensor,
            'encoder_mask': mask,
            'decoder_input': decoder_input,
            'decoder_output': decoder_output,
            'decoder_mask': decoder_mask
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    traj_features = torch.stack([item['traj_features'] for item in batch])
    kb_features = torch.stack([item['kb_features'] for item in batch])
    encoder_masks = torch.stack([item['encoder_mask'] for item in batch])
    decoder_inputs = torch.stack([item['decoder_input'] for item in batch])
    decoder_outputs = torch.stack([item['decoder_output'] for item in batch])
    decoder_masks = torch.stack([item['decoder_mask'] for item in batch])
    
    return {
        'traj_features': traj_features,
        'kb_features': kb_features,
        'encoder_mask': encoder_masks,
        'decoder_input': decoder_inputs,
        'decoder_output': decoder_outputs,
        'decoder_mask': decoder_masks
    }


class EnhancedSwipeTrainer(pl.LightningModule):
    """PyTorch Lightning module for enhanced model training."""
    
    def __init__(self,
                 kb_vocab_size: int = 29,
                 char_vocab_size: int = 32,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Create enhanced model
        self.model = create_enhanced_mobile_model(
            kb_vocab_size=kb_vocab_size,
            char_vocab_size=char_vocab_size
        )
        
        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.pad_idx)
        
        print(f"Enhanced model initialized with {self._count_parameters():,} parameters")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward(self, traj_features, kb_features, decoder_input=None, 
                encoder_mask=None, decoder_mask=None):
        return self.model(
            traj_features,
            kb_features,
            tgt=decoder_input,
            src_key_padding_mask=encoder_mask,
            tgt_key_padding_mask=decoder_mask
        )
    
    def training_step(self, batch, batch_idx):
        # Forward pass with teacher forcing
        logits = self(
            batch['traj_features'],
            batch['kb_features'],
            batch['decoder_input'],
            batch['encoder_mask'],
            batch['decoder_mask']
        )
        
        # Calculate loss
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            batch['decoder_output'].reshape(-1)
        )
        
        # Calculate accuracy (for non-padded positions)
        mask = ~batch['decoder_mask']
        predictions = logits.argmax(dim=-1)
        correct = (predictions == batch['decoder_output']) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass WITHOUT teacher forcing for true validation
        memory = self.model.encoder(
            batch['traj_features'],
            batch['kb_features'],
            batch['encoder_mask']
        )
        
        # Generate predictions autoregressively
        generated = self.model.generate(memory, batch['encoder_mask'])
        
        # Compare with ground truth
        target = batch['decoder_output']
        
        # Calculate character-level accuracy
        min_len = min(generated.size(1) - 1, target.size(1))  # -1 for <sos> token
        gen_chars = generated[:, 1:min_len+1]  # Skip <sos>
        tgt_chars = target[:, :min_len]
        
        # Mask out padding
        mask = tgt_chars != self.model.pad_idx
        correct = (gen_chars == tgt_chars) & mask
        char_accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        # Also calculate with teacher forcing for comparison
        logits = self(
            batch['traj_features'],
            batch['kb_features'],
            batch['decoder_input'],
            batch['encoder_mask'],
            batch['decoder_mask']
        )
        
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            batch['decoder_output'].reshape(-1)
        )
        
        # Teacher forcing accuracy
        mask_tf = ~batch['decoder_mask']
        predictions_tf = logits.argmax(dim=-1)
        correct_tf = (predictions_tf == batch['decoder_output']) & mask_tf
        accuracy_tf = correct_tf.sum().float() / mask_tf.sum().float()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_char_accuracy', char_accuracy, prog_bar=True)  # True accuracy
        self.log('val_tf_accuracy', accuracy_tf, prog_bar=False)  # Teacher forcing accuracy
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Test without teacher forcing
        memory = self.model.encoder(
            batch['traj_features'],
            batch['kb_features'],
            batch['encoder_mask']
        )
        
        generated = self.model.generate(memory, batch['encoder_mask'])
        target = batch['decoder_output']
        
        # Character-level accuracy
        min_len = min(generated.size(1) - 1, target.size(1))
        gen_chars = generated[:, 1:min_len+1]
        tgt_chars = target[:, :min_len]
        
        mask = tgt_chars != self.model.pad_idx
        correct = (gen_chars == tgt_chars) & mask
        char_accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        self.log('test_char_accuracy', char_accuracy)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


def main():
    # Set seed
    pl.seed_everything(42)
    
    # Initialize tokenizers
    print("Initializing tokenizers...")
    char_tokenizer = CharLevelTokenizerv2("data/data_preprocessed/voc.txt")
    kb_tokenizer = KeyboardTokenizerv1()
    
    print(f"Character vocabulary size: {len(char_tokenizer.char_to_idx)}")
    print(f"Keyboard vocabulary size: {len(kb_tokenizer.i2t)}")
    
    # Load nearest key lookup if available
    nearest_key_lookup = None
    lookup_path = "data/data_preprocessed/nearest_key_lookup.pkl"
    if os.path.exists(lookup_path):
        print(f"Loading nearest key lookup from {lookup_path}...")
        try:
            from src.feature_extraction.nearest_key_lookup import ExtendedNearestKeyLookup
            nearest_key_lookup = ExtendedNearestKeyLookup.from_state_dict(lookup_path)
        except:
            print("Failed to load nearest key lookup, will use simple approximation")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EnhancedSwipeDataset(
        "data/combined_dataset/combined_english_swipes_train.jsonl",
        char_tokenizer,
        kb_tokenizer,
        nearest_key_lookup
    )
    
    val_dataset = EnhancedSwipeDataset(
        "data/combined_dataset/combined_english_swipes_val.jsonl",
        char_tokenizer,
        kb_tokenizer,
        nearest_key_lookup
    )
    
    test_dataset = EnhancedSwipeDataset(
        "data/combined_dataset/combined_english_swipes_test.jsonl",
        char_tokenizer,
        kb_tokenizer,
        nearest_key_lookup
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  # Smaller batch size for larger model
        shuffle=True,
        num_workers=0,  # As requested, use 0 workers
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    model = EnhancedSwipeTrainer(
        kb_vocab_size=len(kb_tokenizer.i2t),
        char_vocab_size=len(char_tokenizer.char_to_idx),
        learning_rate=5e-4,
        weight_decay=1e-5
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_char_accuracy',  # Monitor true accuracy
        dirpath='checkpoints/enhanced',
        filename='enhanced-swipe-{epoch:02d}-{val_char_accuracy:.3f}',
        save_top_k=3,
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_char_accuracy',
        patience=10,
        mode='max'
    )
    
    # Logger
    logger = TensorBoardLogger('logs', name='enhanced_swipe')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size of 128
        precision=16  # Mixed precision
    )
    
    # Train
    print("\nStarting enhanced model training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Character vocabulary: {len(char_tokenizer.char_to_idx)}")
    print(f"Keyboard vocabulary: {len(kb_tokenizer.i2t)}")
    print("\nThis model replicates the original architecture for 70% accuracy")
    
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, test_loader)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()