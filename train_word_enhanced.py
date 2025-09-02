#!/usr/bin/env python3
"""
Training script for enhanced word-level model with nearest key features.
Targets 70% accuracy like the original model while maintaining mobile efficiency.
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import enhanced model
from mobile_model_word_enhanced import create_enhanced_word_model


class VocabularyManager:
    """Manages word vocabulary for prediction."""
    
    def __init__(self, vocab_path: str = "data/data_preprocessed/voc.txt", max_size: int = 10000):
        self.vocab_path = vocab_path
        self.max_size = max_size
        self.word_to_idx = {"<unk>": 0, "<pad>": 1}
        self.idx_to_word = {0: "<unk>", 1: "<pad>"}
        self._load_vocabulary()
    
    def _load_vocabulary(self):
        """Load vocabulary from file."""
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'r') as f:
                words = [line.strip().lower() for line in f.readlines()][:self.max_size - 2]
                for idx, word in enumerate(words, start=2):
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
        print(f"Loaded vocabulary with {len(self.word_to_idx)} words")
    
    def encode(self, word: str) -> int:
        """Convert word to index."""
        return self.word_to_idx.get(word.lower(), 0)
    
    def decode(self, idx: int) -> str:
        """Convert index to word."""
        return self.idx_to_word.get(idx, "<unk>")
    
    @property
    def size(self) -> int:
        return len(self.word_to_idx)


class KeyboardTokenizer:
    """Simple keyboard tokenizer for nearest key features."""
    
    def __init__(self):
        self.chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                      'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                      'w', 'x', 'y', 'z', '-', '<unk>', '<pad>']
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
    
    def encode(self, char: str) -> int:
        """Convert character to index."""
        return self.char_to_idx.get(char.lower(), self.char_to_idx['<unk>'])
    
    @property
    def vocab_size(self) -> int:
        return len(self.chars)


class EnhancedWordDataset(Dataset):
    """Dataset with nearest key features for word prediction."""
    
    def __init__(self,
                 data_path: str,
                 vocab_manager: VocabularyManager,
                 kb_tokenizer: KeyboardTokenizer,
                 max_seq_len: int = 150,
                 include_test: bool = False):
        self.data_path = data_path
        self.vocab_manager = vocab_manager
        self.kb_tokenizer = kb_tokenizer
        self.max_seq_len = max_seq_len
        
        # Fixed keyboard dimensions
        self.width = 360
        self.height = 215
        
        # Simple QWERTY layout for nearest key approximation
        self.key_positions = self._create_qwerty_layout()
        
        # Load data
        self.data = []
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Include samples with words for training/val, or all for test
                if include_test or ('word' in item and item['word']):
                    self.data.append(item)
        
        print(f"Loaded {len(self.data)} samples")
    
    def _create_qwerty_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create QWERTY keyboard layout positions."""
        layout = {}
        
        # Row 1: QWERTYUIOP
        row1 = 'qwertyuiop'
        for i, char in enumerate(row1):
            layout[char] = (36 * (i + 0.5), 50)
        
        # Row 2: ASDFGHJKL
        row2 = 'asdfghjkl'
        for i, char in enumerate(row2):
            layout[char] = (36 * (i + 0.5) + 18, 107)  # Offset by half key
        
        # Row 3: ZXCVBNM
        row3 = 'zxcvbnm'
        for i, char in enumerate(row3):
            layout[char] = (36 * (i + 1.5) + 18, 164)  # More offset
        
        return layout
    
    def _get_nearest_key(self, x: float, y: float) -> str:
        """Get nearest keyboard key for a coordinate."""
        min_dist = float('inf')
        nearest_key = 'a'
        
        for key, (kx, ky) in self.key_positions.items():
            dist = (x - kx) ** 2 + (y - ky) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_key = key
        
        return nearest_key
    
    def _compute_features(self, x_coords, y_coords, t_coords):
        """Compute trajectory features with velocities and accelerations."""
        # Normalize coordinates
        x_norm = x_coords / self.width
        y_norm = y_coords / self.height
        
        # Initialize arrays
        n = len(x_coords)
        vx = np.zeros(n, dtype=np.float32)
        vy = np.zeros(n, dtype=np.float32)
        ax = np.zeros(n, dtype=np.float32)
        ay = np.zeros(n, dtype=np.float32)
        
        if n > 1:
            # Compute velocities using central differences
            for i in range(1, n - 1):
                dt_prev = (t_coords[i] - t_coords[i-1]) / 1000.0
                dt_next = (t_coords[i+1] - t_coords[i]) / 1000.0
                
                if dt_prev > 0 and dt_next > 0:
                    # Central difference
                    vx[i] = (x_norm[i+1] - x_norm[i-1]) / (dt_prev + dt_next)
                    vy[i] = (y_norm[i+1] - y_norm[i-1]) / (dt_prev + dt_next)
            
            # Handle endpoints
            if n > 1:
                dt = (t_coords[1] - t_coords[0]) / 1000.0
                if dt > 0:
                    vx[0] = (x_norm[1] - x_norm[0]) / dt
                    vy[0] = (y_norm[1] - y_norm[0]) / dt
                
                dt = (t_coords[-1] - t_coords[-2]) / 1000.0
                if dt > 0:
                    vx[-1] = (x_norm[-1] - x_norm[-2]) / dt
                    vy[-1] = (y_norm[-1] - y_norm[-2]) / dt
            
            # Clip velocities
            vx = np.clip(vx, -5, 5)
            vy = np.clip(vy, -5, 5)
            
            # Compute accelerations
            if n > 2:
                for i in range(1, n - 1):
                    dt_prev = (t_coords[i] - t_coords[i-1]) / 1000.0
                    dt_next = (t_coords[i+1] - t_coords[i]) / 1000.0
                    
                    if dt_prev > 0 and dt_next > 0:
                        ax[i] = (vx[i+1] - vx[i-1]) / (dt_prev + dt_next)
                        ay[i] = (vy[i+1] - vy[i-1]) / (dt_prev + dt_next)
                
                # Clip accelerations
                ax = np.clip(ax, -5, 5)
                ay = np.clip(ay, -5, 5)
        
        return x_norm, y_norm, vx, vy, ax, ay
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract coordinates
        x_coords = np.array(item['curve']['x'], dtype=np.float32)
        y_coords = np.array(item['curve']['y'], dtype=np.float32)
        t_coords = np.array(item['curve']['t'], dtype=np.float32)
        
        # Compute features
        x_norm, y_norm, vx, vy, ax, ay = self._compute_features(x_coords, y_coords, t_coords)
        
        # Get nearest keys
        nearest_keys = []
        for x, y in zip(x_coords, y_coords):
            key = self._get_nearest_key(x, y)
            nearest_keys.append(self.kb_tokenizer.encode(key))
        
        # Stack trajectory features
        traj_features = np.stack([x_norm, y_norm, vx, vy, ax, ay], axis=1)
        
        # Handle NaN values
        if np.any(np.isnan(traj_features)):
            traj_features = np.nan_to_num(traj_features, 0.0)
        
        # Pad or truncate
        seq_len = traj_features.shape[0]
        if seq_len > self.max_seq_len:
            traj_features = traj_features[:self.max_seq_len]
            nearest_keys = nearest_keys[:self.max_seq_len]
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        else:
            # Pad trajectory
            padding = np.zeros((self.max_seq_len - seq_len, 6), dtype=np.float32)
            traj_features = np.concatenate([traj_features, padding], axis=0)
            
            # Pad nearest keys
            pad_token = self.kb_tokenizer.encode('<pad>')
            nearest_keys = nearest_keys + [pad_token] * (self.max_seq_len - seq_len)
            
            # Create mask
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            mask[seq_len:] = True
        
        # Convert to tensors
        traj_tensor = torch.from_numpy(traj_features).float()
        kb_tensor = torch.tensor(nearest_keys, dtype=torch.long)
        
        # Get target word
        target_word = item.get('word', '').lower()
        target_idx = self.vocab_manager.encode(target_word) if target_word else 0
        
        return traj_tensor, kb_tensor, mask, target_idx


def collate_fn(batch):
    """Custom collate function."""
    traj_features, kb_features, masks, targets = zip(*batch)
    
    traj_features = torch.stack(traj_features)
    kb_features = torch.stack(kb_features)
    masks = torch.stack(masks)
    targets = torch.tensor(targets, dtype=torch.long)
    
    return traj_features, kb_features, masks, targets


class EnhancedWordTrainer(pl.LightningModule):
    """PyTorch Lightning module for enhanced word-level training."""
    
    def __init__(self,
                 kb_vocab_size: int = 29,
                 word_vocab_size: int = 10000,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-5,
                 warmup_epochs: int = 5):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = create_enhanced_word_model(
            kb_vocab_size=kb_vocab_size,
            word_vocab_size=word_vocab_size
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=1)  # Ignore <pad>
        
        # Metrics
        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0
        
        print(f"Enhanced model initialized with {self._count_parameters():,} parameters")
        print(f"Model size: {self._count_parameters() * 4 / 1024 / 1024:.2f} MB")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward(self, traj_features, kb_features, mask):
        return self.model(traj_features, kb_features, mask)
    
    def training_step(self, batch, batch_idx):
        traj_features, kb_features, masks, targets = batch
        
        # Forward pass
        logits = self(traj_features, kb_features, masks)
        loss = self.criterion(logits, targets)
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets).sum()
        total = targets.size(0)
        accuracy = correct.float() / total
        
        # Update running metrics
        self.train_correct += correct
        self.train_total += total
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        traj_features, kb_features, masks, targets = batch
        
        # Forward pass (no teacher forcing needed for word prediction)
        logits = self(traj_features, kb_features, masks)
        loss = self.criterion(logits, targets)
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets).sum()
        total = targets.size(0)
        accuracy = correct.float() / total
        
        # Calculate top-5 accuracy
        top5_preds = logits.topk(5, dim=-1)[1]
        top5_correct = (top5_preds == targets.unsqueeze(1)).any(dim=1).sum()
        top5_accuracy = top5_correct.float() / total
        
        # Update running metrics
        self.val_correct += correct
        self.val_total += total
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_top5_acc', top5_accuracy, prog_bar=False, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        traj_features, kb_features, masks, targets = batch
        
        # Forward pass
        logits = self(traj_features, kb_features, masks)
        
        # Calculate metrics
        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets).sum()
        total = targets.size(0)
        accuracy = correct.float() / total
        
        # Top-5 accuracy
        top5_preds = logits.topk(5, dim=-1)[1]
        top5_correct = (top5_preds == targets.unsqueeze(1)).any(dim=1).sum()
        top5_accuracy = top5_correct.float() / total
        
        self.log('test_acc', accuracy)
        self.log('test_top5_acc', top5_accuracy)
    
    def on_train_epoch_end(self):
        # Calculate epoch accuracy
        if self.train_total > 0:
            epoch_acc = self.train_correct.float() / self.train_total
            print(f"\nTrain Epoch Accuracy: {epoch_acc:.4f}")
        self.train_correct = 0
        self.train_total = 0
    
    def on_validation_epoch_end(self):
        # Calculate epoch accuracy
        if self.val_total > 0:
            epoch_acc = self.val_correct.float() / self.val_total
            print(f"Val Epoch Accuracy: {epoch_acc:.4f}")
        self.val_correct = 0
        self.val_total = 0
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Initial restart period
            T_mult=2,  # Increase period after each restart
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }


def main():
    # Set seed
    pl.seed_everything(42)
    
    # Initialize tokenizers
    print("Initializing vocabulary and tokenizers...")
    vocab_manager = VocabularyManager(max_size=10000)
    kb_tokenizer = KeyboardTokenizer()
    
    print(f"Word vocabulary size: {vocab_manager.size}")
    print(f"Keyboard vocabulary size: {kb_tokenizer.vocab_size}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EnhancedWordDataset(
        "data/combined_dataset/combined_english_swipes_train.jsonl",
        vocab_manager,
        kb_tokenizer
    )
    
    val_dataset = EnhancedWordDataset(
        "data/combined_dataset/combined_english_swipes_val.jsonl",
        vocab_manager,
        kb_tokenizer
    )
    
    test_dataset = EnhancedWordDataset(
        "data/combined_dataset/combined_english_swipes_test.jsonl",
        vocab_manager,
        kb_tokenizer,
        include_test=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,  # As requested
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    model = EnhancedWordTrainer(
        kb_vocab_size=kb_tokenizer.vocab_size,
        word_vocab_size=vocab_manager.size,
        learning_rate=5e-4,
        weight_decay=1e-5,
        warmup_epochs=5
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/enhanced_word',
        filename='enhanced-word-{epoch:02d}-{val_acc:.3f}',
        save_top_k=3,
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=15,
        mode='max',
        min_delta=0.001
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger('logs', name='enhanced_word_model')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size of 256
        precision=16,  # Mixed precision
        val_check_interval=0.5  # Check validation twice per epoch
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Enhanced Word-Level Model Training")
    print("="*60)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Word vocabulary: {vocab_manager.size}")
    print(f"Keyboard vocabulary: {kb_tokenizer.vocab_size}")
    print("\nModel features:")
    print("- Trajectory features: x, y, vx, vy, ax, ay")
    print("- Nearest key features for each point")
    print("- 4-layer transformer encoder (matching original)")
    print("- Attention-based sequence aggregation")
    print("- Deep prediction head for word output")
    print("\nTarget: 70% accuracy (matching original model)")
    print("="*60 + "\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\n" + "="*60)
    print("Running Test Evaluation")
    print("="*60)
    trainer.test(model, test_loader)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()