#!/usr/bin/env python3
"""
Comprehensive training script for mobile-optimized neural swipe typing model.
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
from mobile_optimized_model import MobileSwipeTypingModel, create_mobile_model
from collate_functions import pad_collate_fn
import os
from tqdm import tqdm

class SwipeDataset(Dataset):
    """Dataset for mobile swipe typing with efficient feature extraction"""
    
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
                
                # Skip if word too long or contains non-alphabetic characters
                if len(word) > self.max_word_len - 2 or not word.isalpha():
                    continue
                
                # Extract features
                features = self._extract_features(curve['x'], curve['y'], curve['t'])
                if features is None:
                    continue
                
                # Tokenize word
                char_sequence = self._tokenize_word(word)
                if char_sequence is None:
                    continue
                
                self.data.append({
                    'features': features,
                    'target': char_sequence,
                    'word': word
                })
        
        print(f"Loaded {len(self.data)} valid samples")
    
    def _extract_features(self, x: List[float], y: List[float], t: List[int]) -> Optional[torch.Tensor]:
        """Extract 6-dimensional features: x, y, vx, vy, ax, ay"""
        if len(x) < 3 or len(x) != len(y) or len(x) != len(t):
            return None
        
        # Limit sequence length for mobile efficiency
        if len(x) > self.max_seq_len:
            x = x[:self.max_seq_len]
            y = y[:self.max_seq_len]
            t = t[:self.max_seq_len]
        
        # Convert to numpy arrays
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        t = np.array(t, dtype=np.float32)
        
        # Normalize coordinates to [0, 1]
        x = np.clip(x / 360.0, 0, 1)
        y = np.clip(y / 215.0, 0, 1)
        
        # Calculate velocities
        dt = np.diff(t, prepend=t[0])
        dt = np.maximum(dt, 1.0)  # Avoid division by zero
        
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        
        vx = dx / dt * 1000  # pixels per second
        vy = dy / dt * 1000
        
        # Calculate accelerations
        dvx = np.diff(vx, prepend=vx[0])
        dvy = np.diff(vy, prepend=vy[0])
        
        ax = dvx / dt * 1000  # pixels per second squared
        ay = dvy / dt * 1000
        
        # Clip extreme values for stability
        vx = np.clip(vx, -1000, 1000)
        vy = np.clip(vy, -1000, 1000)
        ax = np.clip(ax, -500, 500)
        ay = np.clip(ay, -500, 500)
        
        # Stack features: [x, y, vx, vy, ax, ay]
        features = np.stack([x, y, vx, vy, ax, ay], axis=-1)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _tokenize_word(self, word: str) -> Optional[torch.Tensor]:
        """Convert word to character sequence with SOS/EOS tokens"""
        if len(word) > self.max_word_len - 2:
            return None
        
        # Add SOS token
        char_ids = [self.char_to_idx['<sos>']]
        
        # Add character tokens
        for char in word:
            char_ids.append(self.char_to_idx.get(char, self.char_to_idx['<unk>']))
        
        # Add EOS token
        char_ids.append(self.char_to_idx['<eos>'])
        
        # Pad to max length
        while len(char_ids) < self.max_word_len:
            char_ids.append(self.char_to_idx['<pad>'])
        
        return torch.tensor(char_ids, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        return item['features'], item['target']

class MobileSwipeTrainer(pl.LightningModule):
    """PyTorch Lightning module for training mobile swipe model"""
    
    def __init__(self, model: MobileSwipeTypingModel, learning_rate: float = 0.001,
                 weight_decay: float = 0.0001, warmup_steps: int = 1000):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Metrics tracking
        self.train_correct = 0
        self.train_total = 0
        self.val_correct = 0
        self.val_total = 0
        
        self.save_hyperparameters()
    
    def forward(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None):
        return self.model(features, targets)
    
    def training_step(self, batch, batch_idx):
        features, targets = batch
        
        # Teacher forcing: use target sequence for training
        input_targets = targets[:, :-1]  # Remove last token
        output_targets = targets[:, 1:]  # Remove first token (SOS)
        
        # Forward pass
        logits = self.model(features, input_targets)
        
        # Calculate loss
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), 
                             output_targets.reshape(-1))
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1)
        mask = output_targets != 0  # Don't count padding in accuracy
        acc = (preds == output_targets)[mask].float().mean() if mask.sum() > 0 else torch.tensor(0.0)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, targets = batch
        
        # Teacher forcing for validation
        input_targets = targets[:, :-1]
        output_targets = targets[:, 1:]
        
        # Forward pass
        logits = self.model(features, input_targets)
        
        # Calculate loss
        loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                             output_targets.reshape(-1))
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1)
        mask = output_targets != 0
        acc = (preds == output_targets)[mask].float().mean() if mask.sum() > 0 else torch.tensor(0.0)
        
        # Logging
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # AdamW optimizer with weight decay
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

def create_data_loaders(config: Dict, training_params: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    print("Creating data loaders...")
    
    # Create datasets
    train_dataset = SwipeDataset(config['train_data'], 
                                max_seq_len=config['max_sequence_length'])
    val_dataset = SwipeDataset(config['val_data'],
                              max_seq_len=config['max_sequence_length'])
    test_dataset = SwipeDataset(config['test_data'],
                               max_seq_len=config['max_sequence_length'])
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_params['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True),
        collate_fn=pad_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True),
        collate_fn=pad_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_params['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True),
        collate_fn=pad_collate_fn
    )
    
    return train_loader, val_loader, test_loader

def train_mobile_model(config_path: str = 'configs/config_mobile_optimized.json'):
    """Main training function"""
    print("=== Mobile Swipe Typing Model Training ===")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    train_config = config['training_config']
    model_params = train_config['model_params']
    training_params = train_config['training_params']
    data_params = train_config['data_params']
    
    # Set device and precision
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Set tensor core precision for better performance
        torch.set_float32_matmul_precision('medium')
    
    # Create model
    print("Creating mobile-optimized model...")
    model = MobileSwipeTypingModel(**model_params)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data_params, training_params)
    
    # Create trainer module
    trainer_module = MobileSwipeTrainer(
        model,
        learning_rate=training_params['learning_rate'],
        weight_decay=training_params['weight_decay'],
        warmup_steps=training_params['warmup_steps']
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/mobile_model',
        filename='mobile-swipe-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=training_params['early_stopping_patience'],
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    logger = TensorBoardLogger('lightning_logs', name='mobile_swipe_model')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_params['max_epochs'],
        devices=1 if device == 'cuda' else 'auto',
        accelerator='gpu' if device == 'cuda' else 'cpu',
        precision=32,  # Use FP32 to avoid mixed precision issues
        gradient_clip_val=training_params['gradient_clip_norm'],
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.25,  # Validate 4 times per epoch
        num_sanity_val_steps=0,  # Skip sanity validation to avoid potential issues
    )
    
    # Train model
    print("Starting training...")
    trainer.fit(trainer_module, train_loader, val_loader)
    
    # Test best model
    print("Testing best model...")
    trainer.test(trainer_module, test_loader, ckpt_path='best')
    
    # Save final model for deployment
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    
    # Load best model and save for deployment
    best_model = MobileSwipeTrainer.load_from_checkpoint(best_model_path).model
    
    # Save deployment-ready model
    deployment_path = 'mobile_swipe_model_final.pth'
    torch.save(best_model.state_dict(), deployment_path)
    print(f"Deployment model saved: {deployment_path}")
    
    print("Training completed successfully!")
    return best_model_path, deployment_path

def test_training_functionality():
    """Quick test to ensure training pipeline works"""
    print("=== Training Functionality Test ===")
    
    # Create small test dataset
    test_data = [
        {
            "curve": {
                "x": list(np.random.uniform(0, 360, 50)),
                "y": list(np.random.uniform(0, 215, 50)),
                "t": list(range(0, 50 * 20, 20)),
                "grid_name": "qwerty_english"
            },
            "word": "test"
        }
        for _ in range(100)
    ]
    
    # Save test data
    with open('test_data.jsonl', 'w') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')
    
    # Create small config
    test_config = {
        'training_config': {
            'model_params': {
                'd_model': 32, 'vocab_size': 28, 'max_seq_len': 50,
                'max_word_len': 10, 'nhead': 2, 'num_encoder_layers': 1,
                'num_decoder_layers': 1, 'dropout': 0.1
            },
            'training_params': {
                'batch_size': 16, 'learning_rate': 0.001, 'weight_decay': 0.0001,
                'max_epochs': 2, 'early_stopping_patience': 5,
                'gradient_clip_norm': 1.0, 'warmup_steps': 10
            },
            'data_params': {
                'train_data': 'test_data.jsonl', 'val_data': 'test_data.jsonl',
                'test_data': 'test_data.jsonl', 'max_sequence_length': 50,
                'batch_size': 16, 'num_workers': 0, 'pin_memory': False
            }
        }
    }
    
    with open('test_config.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    
    try:
        # Test dataset loading
        dataset = SwipeDataset('test_data.jsonl', max_seq_len=50, max_word_len=10)
        print(f"Test dataset loaded: {len(dataset)} samples")
        
        # Test single forward pass
        model = MobileSwipeTypingModel(**test_config['training_config']['model_params'])
        features, target = dataset[0]
        
        with torch.no_grad():
            output = model(features.unsqueeze(0), target.unsqueeze(0))
            print(f"Model forward pass successful: {output.shape}")
        
        print("Training functionality test PASSED!")
        
        # Cleanup
        os.remove('test_data.jsonl')
        os.remove('test_config.json')
        
        return True
        
    except Exception as e:
        print(f"Training functionality test FAILED: {e}")
        return False

if __name__ == "__main__":
    # Test functionality first
    if test_training_functionality():
        # Run full training
        train_mobile_model()
    else:
        print("Fix training issues before running full training")