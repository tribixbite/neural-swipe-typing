#!/usr/bin/env python3
"""
Training script for mobile-optimized neural swipe typing model.
Uses vocabulary-based prediction instead of character-by-character generation.
Includes proper validation without teacher forcing.
"""

import os
import sys
# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
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

# Import mobile model
from mobile_model import create_mobile_model


class VocabularyManager:
    """Manages word vocabulary for direct prediction."""
    
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
                words = [line.strip() for line in f.readlines()][:self.max_size - 2]
                for idx, word in enumerate(words, start=2):
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
        print(f"Loaded vocabulary with {len(self.word_to_idx)} words")
    
    def encode(self, word: str) -> int:
        """Convert word to index."""
        return self.word_to_idx.get(word, 0)  # Return <unk> for OOV
    
    def decode(self, idx: int) -> str:
        """Convert index to word."""
        return self.idx_to_word.get(idx, "<unk>")
    
    @property
    def size(self) -> int:
        return len(self.word_to_idx)


class MobileSwipeDataset(Dataset):
    """Dataset for mobile swipe typing with vocabulary-based targets."""
    
    def __init__(self, 
                 data_path: str,
                 vocab_manager: VocabularyManager,
                 max_seq_len: int = 150):
        self.data_path = data_path
        self.vocab_manager = vocab_manager
        self.max_seq_len = max_seq_len
        
        # Get keyboard dimensions for normalization
        self.width = 360  # Fixed width for English keyboard
        self.height = 215  # Fixed height for English keyboard
        
        # Load dataset directly from JSONL
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get data item
        item = self.data[idx]
        
        # Extract coordinates
        x_coords = np.array(item['curve']['x'], dtype=np.float32)
        y_coords = np.array(item['curve']['y'], dtype=np.float32)
        t_coords = np.array(item['curve']['t'], dtype=np.float32)
        
        # Normalize coordinates (0-1 range)
        x_norm = x_coords / self.width
        y_norm = y_coords / self.height
        
        # Compute velocities
        if len(x_coords) > 1:
            dt = np.diff(t_coords)
            dt[dt == 0] = 1  # Avoid division by zero
            
            vx = np.diff(x_norm) / dt * 1000  # Convert to per second
            vy = np.diff(y_norm) / dt * 1000
            
            # Clip velocities to prevent extreme values
            vx = np.clip(vx, -10, 10)
            vy = np.clip(vy, -10, 10)
            
            # Pad velocities to match position length
            vx = np.concatenate([vx, [vx[-1]]])
            vy = np.concatenate([vy, [vy[-1]]])
            
            # Compute accelerations
            dvx = np.diff(vx)
            dvy = np.diff(vy)
            
            # Clip accelerations
            dvx = np.clip(dvx, -10, 10)
            dvy = np.clip(dvy, -10, 10)
            
            ax = np.concatenate([dvx, [dvx[-1] if len(dvx) > 0 else 0]])
            ay = np.concatenate([dvy, [dvy[-1] if len(dvy) > 0 else 0]])
        else:
            # Single point - no velocity or acceleration
            vx = np.zeros_like(x_norm)
            vy = np.zeros_like(y_norm)
            ax = np.zeros_like(x_norm)
            ay = np.zeros_like(y_norm)
        
        # Stack features: [x, y, vx, vy, ax, ay]
        features = np.stack([x_norm, y_norm, vx, vy, ax, ay], axis=1)
        
        # Check for NaN and replace with zeros
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, 0.0)
        
        # Pad or truncate to max_seq_len
        seq_len = features.shape[0]
        if seq_len > self.max_seq_len:
            features = features[:self.max_seq_len]
        elif seq_len < self.max_seq_len:
            padding = np.zeros((self.max_seq_len - seq_len, features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float()
        
        # Get target word index
        target_word = item.get('word', '').lower()
        target_idx = self.vocab_manager.encode(target_word) if target_word else 0
        
        return features_tensor, target_idx


def collate_fn(batch):
    """Custom collate function for batching."""
    features, targets = zip(*batch)
    features = torch.stack(features)
    targets = torch.tensor(targets, dtype=torch.long)
    return features, targets


class MobileSwipeTrainer(pl.LightningModule):
    """PyTorch Lightning module for training mobile swipe typing model."""
    
    def __init__(self,
                 vocab_size: int = 10000,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Create mobile model
        self.model = create_mobile_model(vocab_size=vocab_size)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=1)  # Ignore <pad> token
        
        # Metrics tracking
        self.train_accuracy = []
        self.val_accuracy = []
        
        print(f"Model initialized with {self._count_parameters():,} parameters")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward(self, x, mask=None):
        return self.model(x, mask)
    
    def training_step(self, batch, batch_idx):
        features, targets = batch
        
        # Forward pass
        logits = self(features)
        loss = self.criterion(logits, targets)
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, targets = batch
        
        # Forward pass (no teacher forcing - direct prediction)
        logits = self(features)
        loss = self.criterion(logits, targets)
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        # Calculate top-5 accuracy
        top5_preds = logits.topk(5, dim=-1)[1]
        top5_accuracy = (top5_preds == targets.unsqueeze(1)).any(dim=1).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_top5_accuracy', top5_accuracy, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        features, targets = batch
        
        # Forward pass
        logits = self(features)
        
        # Calculate metrics
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        # Top-5 accuracy
        top5_preds = logits.topk(5, dim=-1)[1]
        top5_accuracy = (top5_preds == targets.unsqueeze(1)).any(dim=1).float().mean()
        
        self.log('test_accuracy', accuracy)
        self.log('test_top5_accuracy', top5_accuracy)
    
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


def export_to_onnx(model: nn.Module, save_path: str, vocab_size: int = 10000):
    """Export model to ONNX format."""
    model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_len = 150
    dummy_input = torch.randn(batch_size, seq_len, 6)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['trajectory'],
        output_names=['word_probabilities'],
        dynamic_axes={
            'trajectory': {0: 'batch_size'},
            'word_probabilities': {0: 'batch_size'}
        }
    )
    print(f"Model exported to ONNX: {save_path}")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # Print model size
    model_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"ONNX model size: {model_size:.2f} MB")


def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Initialize vocabulary manager
    vocab_manager = VocabularyManager(max_size=10000)
    
    # Create datasets
    train_dataset = MobileSwipeDataset(
        "data/combined_dataset/combined_english_swipes_train.jsonl",
        vocab_manager
    )
    
    val_dataset = MobileSwipeDataset(
        "data/combined_dataset/combined_english_swipes_val.jsonl",
        vocab_manager
    )
    
    test_dataset = MobileSwipeDataset(
        "data/combined_dataset/combined_english_swipes_test.jsonl",
        vocab_manager
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    model = MobileSwipeTrainer(
        vocab_size=vocab_manager.size,
        learning_rate=1e-4,  # Reduced learning rate for stability
        weight_decay=1e-5
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='checkpoints/mobile',
        filename='mobile-swipe-{epoch:02d}-{val_accuracy:.3f}',
        save_top_k=3,
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max'
    )
    
    # Logger
    logger = TensorBoardLogger('logs', name='mobile_swipe')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size of 256
        precision=16  # Mixed precision training
    )
    
    # Train
    print("\nStarting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocabulary size: {vocab_manager.size}")
    
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, test_loader)
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        # Load best model
        best_model = MobileSwipeTrainer.load_from_checkpoint(
            best_model_path,
            vocab_size=vocab_manager.size
        )
        
        # Export
        export_to_onnx(
            best_model.model,
            "models/mobile_swipe_model.onnx",
            vocab_size=vocab_manager.size
        )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()