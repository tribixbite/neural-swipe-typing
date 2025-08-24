#!/usr/bin/env python3
"""Test training script for small English subset."""

import json
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import CurveDataset, custom_collate_fn
from model import MODEL_GETTERS_DICT
from feature_extraction.feature_extractors import get_transform
from pl_module import LitNeuroswipeModel


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_english_subset.json')
    parser.add_argument('--gpus', type=int, default=0)
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    pred_config = config['prediction_config']
    train_config = config['training_config']
    
    print("Loading configuration...")
    print(f"Model: {pred_config['model_params'][0][1]}")
    print(f"Transform: {pred_config['transform_name']}")
    print(f"Batch sizes: train={train_config['batch_size_train']}, val={train_config['batch_size_val']}")
    
    # Load vocabulary
    with open(pred_config['voc_path'], 'r') as f:
        voc = f.read().strip().split('\n')
    n_classes = len(voc)
    print(f"Vocabulary size: {n_classes}")
    
    # Load keyboard grid
    with open(pred_config['grid_name_to_grid__path'], 'r') as f:
        grid_name_to_grid = json.load(f)
    
    # Get transform
    transform = get_transform(
        transform_name=pred_config['transform_name'],
        voc=voc,
        grid_name_to_grid=grid_name_to_grid,
        include_time=pred_config.get('include_time', False),
        include_velocities=pred_config.get('include_velocities', True),
        include_accelerations=pred_config.get('include_accelerations', True)
    )
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CurveDataset(
        path=pred_config['data_split__to__path']['train'],
        grid_name_to_grid=grid_name_to_grid,
        transform=transform,
        max_n_points=200
    )
    
    val_dataset = CurveDataset(
        path=pred_config['data_split__to__path']['val'],
        grid_name_to_grid=grid_name_to_grid,
        transform=transform,
        max_n_points=200
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Test loading a single sample
    print("\nTesting data loading...")
    sample = train_dataset[0]
    if isinstance(sample, tuple) and len(sample) == 2:
        model_input, target = sample
        print(f"Sample loaded successfully")
        print(f"Model input type: {type(model_input)}")
        if isinstance(model_input, tuple):
            print(f"  Encoder input shape: {model_input[0].shape if hasattr(model_input[0], 'shape') else type(model_input[0])}")
            print(f"  Decoder input shape: {model_input[1].shape if hasattr(model_input[1], 'shape') else type(model_input[1])}")
        print(f"Target shape: {target.shape if hasattr(target, 'shape') else type(target)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size_train'],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=train_config['num_workers'],
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size_val'],
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=train_config['num_workers'],
        persistent_workers=False
    )
    
    # Test batch loading
    print("\nTesting batch loading...")
    batch = next(iter(train_loader))
    packed_model_in, dec_out = batch
    encoder_in, decoder_in, swipe_pad_mask, word_pad_mask = packed_model_in
    print(f"Batch loaded successfully")
    print(f"Encoder input shape: {encoder_in.shape if hasattr(encoder_in, 'shape') else [x.shape for x in encoder_in] if isinstance(encoder_in, tuple) else type(encoder_in)}")
    print(f"Decoder input shape: {decoder_in.shape}")
    print(f"Decoder output shape: {dec_out.shape}")
    print(f"Swipe pad mask shape: {swipe_pad_mask.shape}")
    print(f"Word pad mask shape: {word_pad_mask.shape}")
    
    # Create criterion
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=train_config['label_smoothing']
    )
    
    # Create Lightning model  
    lit_model = LitNeuroswipeModel(
        model_name=pred_config['model_params'][0][1],
        criterion=criterion,
        num_classes=n_classes,
        train_batch_size=train_config['batch_size_train'],
        optim_kwargs={'lr': train_config['learning_rate'], 'weight_decay': 0},
        label_smoothing=train_config['label_smoothing']
    )
    
    print(f"\nModel created: {type(lit_model.model).__name__}")
    print(f"Model parameters: {sum(p.numel() for p in lit_model.model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    lit_model.eval()
    with torch.no_grad():
        try:
            output = lit_model.model(*packed_model_in)
            print(f"Forward pass successful!")
            print(f"Output shape: {output.shape}")
            print(f"Expected shape: (batch_size={train_config['batch_size_train']}, seq_len, n_classes={n_classes})")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./results/models_for_debug/english_subset',
        filename='english_subset-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=train_config['early_stopping_patience'],
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger('./results/logs', name='english_subset')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_config['num_epochs'],
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        gradient_clip_val=train_config['gradient_clip_val'],
        accumulate_grad_batches=train_config['accumulate_grad_batches'],
        val_check_interval=1.0,
        log_every_n_steps=10
    )
    
    print("\n" + "="*50)
    print("Starting training on small subset...")
    print("="*50)
    
    # Train
    trainer.fit(lit_model, train_loader, val_loader)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print("="*50)


if __name__ == "__main__":
    main()