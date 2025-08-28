#!/usr/bin/env python3
"""
Training script for English neural swipe typing model.
Adapted for the new English dataset with converted coordinates.
"""

import os
import sys
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


class LitNeuroswipeModel(LightningModule):
    def __init__(self, model_name: str, n_coord_feats: int, num_classes: int,
                 criterion_ignore_index: int = -100, 
                 lr: float = 1e-4, weight_decay: float = 0,
                 label_smoothing: float = 0.045):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = MODEL_GETTERS_DICT[model_name](n_coord_feats=n_coord_feats)
        self.criterion_ignore_index = criterion_ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        
        # Metrics
        self.train_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)
        self.val_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)

    def forward(self, encoder_in, y, encoder_in_pad_mask, y_pad_mask):
        return self.model.forward(encoder_in, y, encoder_in_pad_mask, y_pad_mask)

    def cross_entropy_with_reshape(self, pred, target):
        pred_flat = pred.view(-1, pred.shape[-1])
        target_flat = target.reshape(-1)
        return F.cross_entropy(pred_flat, target_flat, 
                             ignore_index=self.criterion_ignore_index,
                             label_smoothing=self.label_smoothing)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
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
            pad_token=self.criterion_ignore_index, 
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
            pad_token=self.criterion_ignore_index,
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


def main():
    # Configuration
    GRID_NAME = "qwerty_english"
    TRAIN_BATCH_SIZE = 64  # Smaller batch size for 16GB VRAM
    VAL_BATCH_SIZE = 128
    MODEL_NAME = "v3_nearest_and_traj_transformer_bigger"
    TRANSFORM_NAME = "traj_feats_and_nearest_key"
    RANDOM_SEED = 42
    MAX_EPOCHS = 50
    
    # Feature configuration
    USE_TIME = False
    USE_VELOCITY = True
    USE_ACCELERATION = True
    N_COORD_FEATS = 2 * (1 + USE_VELOCITY + USE_ACCELERATION) + USE_TIME  # 6 features
    
    # Paths - using converted datasets
    DATA_ROOT = "data"
    train_path = os.path.join(DATA_ROOT, "converted_english_swipes_train.jsonl")
    val_path = os.path.join(DATA_ROOT, "converted_english_swipes_val.jsonl")
    gridname_to_grid_path = os.path.join(DATA_ROOT, "data_preprocessed", "gridname_to_grid.json")
    voc_path = os.path.join(DATA_ROOT, "data_preprocessed", "english_vocab.txt")
    
    print(f"Training configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Grid: {GRID_NAME}")
    print(f"  Batch size: {TRAIN_BATCH_SIZE}")
    print(f"  Coordinate features: {N_COORD_FEATS}")
    print(f"  Max epochs: {MAX_EPOCHS}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    
    # Initialize tokenizers
    char_tokenizer = CharLevelTokenizerv2(voc_path)
    WORD_PAD_IDX = char_tokenizer.char_to_idx['<pad>']
    NUM_CLASSES = 28  # 26 letters + <sos> + <eos>
    
    print(f"  Vocabulary size: {len(char_tokenizer.idx_to_char)}")
    print(f"  Number of classes: {NUM_CLASSES}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        gridname_to_grid_path=gridname_to_grid_path,
        grid_names=[GRID_NAME],
        transform_name=TRANSFORM_NAME,
        char_tokenizer=char_tokenizer,
        uniform_noise_range=0,  # No augmentation for initial test
        include_time=USE_TIME,
        include_velocities=USE_VELOCITY,
        include_accelerations=USE_ACCELERATION,
        dist_weights_func=weights_function_v1,
        ds_paths_list=[train_path, val_path],
        totals=(15177, 759)  # From english_dataset_stats.json
    )
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CurveDataset(
        data_path=train_path,
        store_gnames=False,
        get_item_transform=train_transform,
        total=15177
    )
    
    val_dataset = CurveDataset(
        data_path=val_path,
        store_gnames=False,
        get_item_transform=val_transform,
        total=759
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    collate_fn = CollateFnV2(word_pad_idx=WORD_PAD_IDX, batch_first=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=2, persistent_workers=True, collate_fn=collate_fn)
    
    val_loader = DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=2, persistent_workers=True, collate_fn=collate_fn)
    
    # Create model
    print("\nInitializing model...")
    model = LitNeuroswipeModel(
        model_name=MODEL_NAME,
        n_coord_feats=N_COORD_FEATS,
        num_classes=NUM_CLASSES,
        criterion_ignore_index=WORD_PAD_IDX,
        lr=1e-4,
        weight_decay=0,
        label_smoothing=0.045
    )
    
    # Setup callbacks
    early_stopping_cb = EarlyStopping(
        monitor='val_loss', mode='min', patience=10, verbose=True)
    
    model_checkpoint_cb = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=3,
        dirpath='checkpoints_english/',
        filename='english-{epoch:02d}-{val_loss:.3f}-{val_word_acc:.3f}')
    
    # Setup trainer
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[early_stopping_cb, model_checkpoint_cb],
        logger=pl_loggers.TensorBoardLogger(
            save_dir='lightning_logs/', 
            name='english_swipe_training'
        ),
        log_every_n_steps=50,
        val_check_interval=200,
        enable_progress_bar=True
    )
    
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\nTraining completed!")
    print(f"Best model saved at: {model_checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()