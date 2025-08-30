#!/usr/bin/env python3
"""
Training script for ExecuTorch-compatible mobile-optimized neural swipe typing model.
Optimized for RTX 4090M 16GB VRAM with maximum accuracy focus.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from mobile_optimized_model_executorch import MobileSwipeTypingModelExecutorch, create_mobile_model_executorch
from collate_functions import pad_collate_fn
import os
from tqdm import tqdm

class SwipeDatasetExecutorch(Dataset):
    """Dataset for ExecuTorch-compatible mobile swipe typing with efficient feature extraction"""
    
    def __init__(self, jsonl_path: str, max_seq_len: int = 150, max_word_len: int = 20):
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        self.data = []
        
        # Character tokenizer
        self.char_to_idx = {'<pad>': 0, '<eos>': 1, '<unk>': 2, '<sos>': 3}
        self.char_to_idx.update({chr(ord('a') + i): i + 4 for i in range(26)})
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        print(f"Loading dataset from {jsonl_path}")
        self._load_data(jsonl_path)
        
    def _load_data(self, jsonl_path: str):
        """Load and preprocess data"""
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Loading data"):
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                
                # Extract trajectory and features
                curve = entry['curve']
                word = entry['word'].lower()
                
                # Skip words that are too long
                if len(word) > self.max_word_len - 2:  # Reserve space for SOS/EOS
                    continue
                
                # Extract features
                features = self._extract_features(
                    curve['x'], curve['y'], curve['t']
                )
                
                if features is None:
                    continue
                
                # Tokenize target word
                target = self._tokenize_word(word)
                if target is None:
                    continue
                
                self.data.append({
                    'features': features,
                    'target': target,
                    'word': word
                })
        
        print(f"Loaded {len(self.data)} valid samples")
    
    def _extract_features(self, x, y, t) -> Optional[torch.Tensor]:
        """Extract 6D features: x, y, vx, vy, ax, ay"""
        if len(x) < 2 or len(y) < 2 or len(t) < 2:
            return None
        
        # Convert to numpy arrays
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        t = np.array(t, dtype=np.float32)
        
        # Normalize coordinates
        x_norm = x / 360.0  # Normalize to [0, 1]
        y_norm = y / 215.0  # Normalize to [0, 1]
        
        # Calculate velocities
        dt = np.diff(t)
        dt = np.maximum(dt, 1.0)  # Avoid division by zero
        
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        # Pad velocities to match coordinate length
        vx = np.concatenate([[0.0], vx])
        vy = np.concatenate([[0.0], vy])
        
        # Calculate accelerations
        if len(vx) > 1:
            dvx = np.diff(vx)
            dvy = np.diff(vy)
            if len(dvx) > 0 and len(dt) > 1:
                dt_acc = dt[:-1] if len(dt) > len(dvx) else dt[:len(dvx)]
                dt_acc = np.maximum(dt_acc, 1.0)  # Avoid division by zero
                ax = dvx / dt_acc
                ay = dvy / dt_acc
            else:
                ax = np.array([0.0])
                ay = np.array([0.0])
        else:
            ax = np.array([0.0])
            ay = np.array([0.0])
        
        # Pad accelerations to match coordinate length
        while len(ax) < len(x_norm):
            ax = np.concatenate([[0.0], ax])
            ay = np.concatenate([[0.0], ay])
        
        # Truncate if too long
        ax = ax[:len(x_norm)]
        ay = ay[:len(x_norm)]
        
        # Clip velocities and accelerations
        vx = np.clip(vx, -1000, 1000) / 1000.0  # Normalize
        vy = np.clip(vy, -1000, 1000) / 1000.0
        ax = np.clip(ax, -500, 500) / 500.0
        ay = np.clip(ay, -500, 500) / 500.0
        
        # Stack features
        features = np.stack([x_norm, y_norm, vx, vy, ax, ay], axis=1)
        
        # Truncate to max sequence length
        if len(features) > self.max_seq_len:
            features = features[:self.max_seq_len]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _tokenize_word(self, word: str) -> Optional[torch.Tensor]:
        """Convert word to character indices with SOS/EOS tokens"""
        if not word or len(word) > self.max_word_len - 2:
            return None
        
        # Add SOS token, characters, and EOS token
        indices = [self.char_to_idx['<sos>']]
        
        for char in word:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx['<unk>'])
        
        indices.append(self.char_to_idx['<eos>'])
        
        # Pad to max_word_len
        while len(indices) < self.max_word_len:
            indices.append(self.char_to_idx['<pad>'])
        
        return torch.tensor(indices[:self.max_word_len], dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['features'], item['target']

class MobileSwipeTrainerExecutorch(pl.LightningModule):
    """PyTorch Lightning module for ExecuTorch-compatible mobile swipe typing"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config
        
        # Create model
        model_params = config['training_config']['model_params']
        self.model = MobileSwipeTypingModelExecutorch(
            d_model=model_params['d_model'],
            vocab_size=model_params['vocab_size'],
            max_seq_len=model_params['max_seq_len'],
            max_word_len=model_params['max_word_len'],
            nhead=model_params['nhead'],
            num_encoder_layers=model_params['num_encoder_layers'],
            num_decoder_layers=model_params['num_decoder_layers'],
            dropout=model_params['dropout']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Metrics
        self.train_acc = []
        self.val_acc = []
    
    def forward(self, features, target_chars=None):
        return self.model(features, target_chars)
    
    def training_step(self, batch, batch_idx):
        features, targets = batch
        
        # Teacher forcing: use target[:-1] as input, target[1:] as output
        input_chars = targets[:, :-1]
        output_chars = targets[:, 1:]
        
        # Forward pass
        logits = self.model(features, input_chars)
        
        # Calculate loss
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), output_chars.reshape(-1))
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            # Only count non-padding tokens
            mask = (output_chars != 0)
            if mask.sum() > 0:
                accuracy = (predictions == output_chars)[mask].float().mean()
                self.train_acc.append(accuracy.item())
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, targets = batch
        
        input_chars = targets[:, :-1]
        output_chars = targets[:, 1:]
        
        logits = self.model(features, input_chars)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), output_chars.reshape(-1))
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        mask = (output_chars != 0)
        if mask.sum() > 0:
            accuracy = (predictions == output_chars)[mask].float().mean()
            self.val_acc.append(accuracy.item())
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy if mask.sum() > 0 else 0.0, prog_bar=True)
        
        return loss
    
    def on_training_epoch_end(self):
        if self.train_acc:
            avg_acc = np.mean(self.train_acc)
            self.log('avg_train_acc', avg_acc)
            self.train_acc.clear()
    
    def on_validation_epoch_end(self):
        if self.val_acc:
            avg_acc = np.mean(self.val_acc)
            self.log('avg_val_acc', avg_acc)
            self.val_acc.clear()
    
    def configure_optimizers(self):
        training_params = self.config['training_config']['training_params']
        optimizer = optim.AdamW(
            self.parameters(),
            lr=training_params['learning_rate'],
            weight_decay=training_params['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=training_params['learning_rate'],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

def main():
    """Main training function"""
    print("=== ExecuTorch-Compatible Mobile Model Training ===")
    
    # Load configuration
    config_path = 'configs/config_mobile_optimized.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create datasets
    data_params = config['training_config']['data_params']
    
    print("Creating datasets...")
    train_dataset = SwipeDatasetExecutorch(data_params['train_data'])
    val_dataset = SwipeDatasetExecutorch(data_params['val_data'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    training_params = config['training_config']['training_params']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params['batch_size'],
        shuffle=True,
        num_workers=data_params['num_workers'],
        pin_memory=data_params['pin_memory'],
        collate_fn=pad_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=data_params['num_workers'],
        pin_memory=data_params['pin_memory'],
        collate_fn=pad_collate_fn
    )
    
    # Create model
    model = MobileSwipeTrainerExecutorch(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/mobile_model_executorch',
        filename='mobile-swipe-executorch-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=training_params['early_stopping_patience'],
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=training_params['max_epochs'],
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=TensorBoardLogger('lightning_logs', name='mobile_swipe_model_executorch'),
        gradient_clip_val=training_params['gradient_clip_norm'],
        precision='32',  # Full precision for stability
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        val_check_interval=0.25,  # Validate 4 times per epoch
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("=== Training Completed ===")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    main()