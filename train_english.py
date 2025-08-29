#!/usr/bin/env python3
"""
Training script for English neural swipe typing model.
Adapted for the new English dataset with converted coordinates.
"""

import os
import sys
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizerv1
from dataset import CurveDataset, CollateFnV2
from feature_extraction.feature_extractors import (
    weights_function_v1, get_transforms
)
from metrics import get_word_level_accuracy
from model import MODEL_GETTERS_DICT

# Lightning imports
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
import torchmetrics


@dataclass
class TrainingConfig:
    """Centralized configuration for all training parameters"""
    # Model configuration
    model_name: str = "v3_nearest_and_traj_transformer_bigger"
    grid_name: str = "qwerty_english"
    transform_name: str = "traj_feats_and_nearest_key"
    
    # Feature configuration
    use_time: bool = False
    use_velocity: bool = True
    use_acceleration: bool = True
    uniform_noise_range: int = 0  # Data augmentation
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    label_smoothing: float = 0.1
    train_batch_size: int = 64
    val_batch_size: int = 128
    max_epochs: int = 150
    random_seed: int = 42
    
    # Optimizer configuration
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    gradient_clip_val: float = 1.0
    
    # Early stopping and checkpointing
    early_stopping_patience: int = 10
    early_stopping_monitor: str = 'val_loss'
    early_stopping_mode: str = 'min'
    checkpoint_save_top_k: int = 3
    checkpoint_monitor: str = 'val_loss'
    checkpoint_mode: str = 'min'
    
    # Training configuration
    num_workers: int = 2
    persistent_workers: bool = True
    batch_first: bool = False
    log_every_n_steps: int = 25
    val_check_interval: int = 50
    enable_progress_bar: bool = True
    
    # Data paths
    data_root: str = "data"
    train_file: str = "raw_converted_english_swipes_train.jsonl"
    val_file: str = "raw_converted_english_swipes_val.jsonl" 
    vocab_file: str = "data_preprocessed/english_vocab.txt"
    grid_file: str = "data_preprocessed/gridname_to_grid.json"
    
    # Dataset sizes (for progress tracking)
    train_total: int = 37688
    val_total: int = 8076
    
    # Logging configuration
    log_dir: str = 'lightning_logs/'
    log_name: str = 'english_swipe_training'
    checkpoint_dir: str = 'checkpoints_english/'
    checkpoint_filename: str = 'english-{epoch:02d}-{val_loss:.3f}-{val_word_acc:.3f}'
    
    # Model architecture constants
    vocab_size: int = 30  # Character-level tokenization
    criterion_ignore_index: int = -100
    
    @property
    def n_coord_feats(self) -> int:
        """Calculate number of coordinate features based on enabled options"""
        return 2 * (1 + self.use_velocity + self.use_acceleration) + self.use_time
    
    @property
    def train_path(self) -> str:
        return os.path.join(self.data_root, self.train_file)
    
    @property
    def val_path(self) -> str:
        return os.path.join(self.data_root, self.val_file)
    
    @property
    def vocab_path(self) -> str:
        return os.path.join(self.data_root, self.vocab_file)
    
    @property
    def grid_path(self) -> str:
        return os.path.join(self.data_root, self.grid_file)


# Predefined configurations
CONFIGS = {
    'default': TrainingConfig(),
    
    'test': TrainingConfig(
        max_epochs=3,
        train_batch_size=16,
        val_batch_size=32,
        early_stopping_patience=2,
        val_check_interval=10,
        log_every_n_steps=5,
        log_name='test_training'
    ),
    
    'aggressive': TrainingConfig(
        learning_rate=2e-4,
        weight_decay=1e-4,
        label_smoothing=0.2,
        train_batch_size=96,
        val_batch_size=192,
        max_epochs=300,
        lr_scheduler_patience=12,
        early_stopping_patience=30,
        gradient_clip_val=0.5,
        uniform_noise_range=3,  # Data augmentation
        log_name='aggressive_training'
    ),
    
    'conservative': TrainingConfig(
        learning_rate=5e-5,
        weight_decay=5e-6,
        label_smoothing=0.05,
        train_batch_size=32,
        val_batch_size=64,
        max_epochs=200,
        lr_scheduler_patience=3,
        early_stopping_patience=8,
        gradient_clip_val=2.0,
        uniform_noise_range=0,
        log_name='conservative_training'
    ),
    
    'fast': TrainingConfig(
        learning_rate=3e-4,
        train_batch_size=128,
        val_batch_size=256,
        max_epochs=50,
        lr_scheduler_patience=3,
        early_stopping_patience=5,
        val_check_interval=25,
        log_every_n_steps=10,
        log_name='fast_training'
    ),
}


class LitNeuroswipeModel(LightningModule):
    def __init__(self, config: TrainingConfig, num_classes: int):
        super().__init__()
        
        self.save_hyperparameters(asdict(config))
        self.config = config
        
        self.model = MODEL_GETTERS_DICT[config.model_name](n_coord_feats=config.n_coord_feats)
        
        # Initialize weights to prevent NaN
        self._init_weights()
        
        # Metrics with correct number of classes
        self.train_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, 
            ignore_index=config.criterion_ignore_index)
        self.val_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, 
            ignore_index=config.criterion_ignore_index)

    def forward(self, encoder_in, y, encoder_in_pad_mask, y_pad_mask):
        return self.model.forward(encoder_in, y, encoder_in_pad_mask, y_pad_mask)

    def cross_entropy_with_reshape(self, pred, target):
        pred_flat = pred.view(-1, pred.shape[-1])
        target_flat = target.reshape(-1)
        return F.cross_entropy(
            pred_flat, target_flat, 
            ignore_index=self.config.criterion_ignore_index,
            label_smoothing=self.config.label_smoothing)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=self.config.lr_scheduler_patience, 
            factor=self.config.lr_scheduler_factor)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.config.early_stopping_monitor
        }

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_size = batch_y.shape[-1]
        
        pred = self.forward(*batch_x)
        loss = self.cross_entropy_with_reshape(pred, batch_y)
        
        # Word-level accuracy
        argmax_pred = torch.argmax(pred, dim=2)
        _, _, _, dec_seq_pad_mask = batch_x
        wl_accuracy = get_word_level_accuracy(
            argmax_pred.T, batch_y.T, 
            pad_token=self.config.criterion_ignore_index, 
            mask=dec_seq_pad_mask)
        
        # Token-level accuracy
        flat_y = batch_y.reshape(-1)
        n_classes = pred.shape[-1]
        flat_preds = pred.reshape(-1, n_classes)
        self.train_token_acc(flat_preds, flat_y)
        
        self.log('train_token_acc', self.train_token_acc, on_step=True, on_epoch=False)
        self.log("train_word_acc", wl_accuracy, on_step=True, on_epoch=True,
                prog_bar=True, batch_size=batch_size)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                prog_bar=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_size = batch_y.shape[-1]
        
        pred = self.forward(*batch_x)
        loss = self.cross_entropy_with_reshape(pred, batch_y)
        
        # Word-level accuracy
        argmax_pred = torch.argmax(pred, dim=2)
        _, _, _, dec_seq_pad_mask = batch_x
        wl_accuracy = get_word_level_accuracy(
            argmax_pred.T, batch_y.T,
            pad_token=self.config.criterion_ignore_index,
            mask=dec_seq_pad_mask)
        
        # Token-level accuracy
        flat_y = batch_y.reshape(-1)
        n_classes = pred.shape[-1]
        flat_preds = pred.reshape(-1, n_classes)
        self.val_token_acc(flat_preds, flat_y)
        
        self.log('val_token_acc', self.val_token_acc, on_step=False, on_epoch=True)
        self.log("val_word_acc", wl_accuracy, on_step=False, on_epoch=True,
                prog_bar=True, batch_size=batch_size)
        self.log("val_loss", loss, on_step=False, on_epoch=True, 
                prog_bar=True, batch_size=batch_size)
        
        return loss
    
    def _init_weights(self):
        """Initialize model weights to prevent NaN gradients"""
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                torch.nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight.data, 0.0, 0.1)
            elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
                torch.nn.init.ones_(module.weight.data)
                torch.nn.init.zeros_(module.bias.data)


def print_config(config: TrainingConfig):
    """Print training configuration in a nice format"""
    print("🔧 Training Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Grid: {config.grid_name}")
    print(f"  Transform: {config.transform_name}")
    print(f"  Features: {config.n_coord_feats} (time={config.use_time}, vel={config.use_velocity}, acc={config.use_acceleration})")
    print(f"  Batch sizes: train={config.train_batch_size}, val={config.val_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Label smoothing: {config.label_smoothing}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Augmentation noise: {config.uniform_noise_range}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def train_model(config: TrainingConfig):
    """Main training function"""
    print_config(config)
    
    # Set random seed
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    
    # Initialize tokenizers
    char_tokenizer = CharLevelTokenizerv2(config.vocab_path)
    word_pad_idx = char_tokenizer.char_to_idx['<pad>']
    num_classes = len(char_tokenizer.idx_to_char)
    
    print(f"  Vocabulary size: {num_classes}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        gridname_to_grid_path=config.grid_path,
        grid_names=[config.grid_name],
        transform_name=config.transform_name,
        char_tokenizer=char_tokenizer,
        uniform_noise_range=config.uniform_noise_range,
        include_time=config.use_time,
        include_velocities=config.use_velocity,
        include_accelerations=config.use_acceleration,
        dist_weights_func=weights_function_v1,
        ds_paths_list=[config.train_path, config.val_path],
        totals=(config.train_total, config.val_total)
    )
    
    # Create datasets
    print("\n📚 Loading datasets...")
    train_dataset = CurveDataset(
        data_path=config.train_path,
        store_gnames=False,
        get_item_transform=train_transform,
        total=config.train_total
    )
    
    val_dataset = CurveDataset(
        data_path=config.val_path,
        store_gnames=False,
        get_item_transform=val_transform,
        total=config.val_total
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    collate_fn = CollateFnV2(word_pad_idx=word_pad_idx, batch_first=config.batch_first)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True,
        num_workers=config.num_workers, 
        persistent_workers=config.persistent_workers, 
        collate_fn=collate_fn)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.val_batch_size, 
        shuffle=False,
        num_workers=config.num_workers, 
        persistent_workers=config.persistent_workers, 
        collate_fn=collate_fn)
    
    # Create model
    print("\n🤖 Initializing model...")
    model = LitNeuroswipeModel(config=config, num_classes=num_classes)
    
    # Setup callbacks
    early_stopping_cb = EarlyStopping(
        monitor=config.early_stopping_monitor, 
        mode=config.early_stopping_mode, 
        patience=config.early_stopping_patience, 
        verbose=True)
    
    model_checkpoint_cb = ModelCheckpoint(
        monitor=config.checkpoint_monitor, 
        mode=config.checkpoint_mode, 
        save_top_k=config.checkpoint_save_top_k,
        dirpath=config.checkpoint_dir,
        filename=config.checkpoint_filename)
    
    # Setup trainer
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[early_stopping_cb, model_checkpoint_cb],
        logger=pl_loggers.TensorBoardLogger(
            save_dir=config.log_dir, 
            name=config.log_name
        ),
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        enable_progress_bar=config.enable_progress_bar,
        gradient_clip_val=config.gradient_clip_val
    )
    
    print("\n🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\n✅ Training completed!")
    print(f"📁 Best model saved at: {model_checkpoint_cb.best_model_path}")
    
    return trainer, model


def main():
    parser = argparse.ArgumentParser(description='Train neural swipe typing model')
    parser.add_argument('--config', type=str, default='default',
                       choices=list(CONFIGS.keys()),
                       help='Predefined configuration to use')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations and exit')
    
    # Allow overriding specific parameters
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, help='Override train batch size')
    parser.add_argument('--max-epochs', type=int, help='Override max epochs')
    parser.add_argument('--model-name', type=str, help='Override model name')
    
    args = parser.parse_args()
    
    if args.list_configs:
        print("Available configurations:")
        for name, config in CONFIGS.items():
            print(f"  {name}: lr={config.learning_rate}, batch={config.train_batch_size}, epochs={config.max_epochs}")
        return
    
    # Get base configuration
    config = CONFIGS[args.config]
    
    # Apply overrides
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.train_batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.model_name is not None:
        config.model_name = args.model_name
    
    print(f"🎯 Using configuration: '{args.config}'")
    train_model(config)


if __name__ == "__main__":
    main()