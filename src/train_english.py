#!/usr/bin/env python3
"""
Training script for English swipe typing transformer.
Adapted from train.ipynb for command-line execution.
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
import torchmetrics

from ns_tokenizers import CharLevelTokenizerv2
from create_english_keyboard_tokenizer import KeyboardTokenizerEnglish
from dataset import CurveDataset, CollateFnV2
from feature_extraction.feature_extractors import (
    weights_function_v1_softmax, 
    weights_function_v1, 
    weights_function_sigmoid_normalized_v1,
    get_transforms
)
from metrics import get_word_level_accuracy
from model import MODEL_GETTERS_DICT


def init_random_seed(value):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)


def cross_entropy_with_reshape(pred, target, ignore_index=-100, label_smoothing=0.0):
    """
    pred - BatchSize x TargetLen x VocabSize
    target - BatchSize x TargetLen
    """
    pred_flat = pred.view(-1, pred.shape[-1])
    target_flat = target.reshape(-1)
    return F.cross_entropy(pred_flat, target_flat,
                           ignore_index=ignore_index,
                           label_smoothing=label_smoothing)


def get_lr_scheduler(optimizer, patience=20, factor=0.5):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, factor=factor, verbose=True)


class LitNeuroswipeModel(LightningModule):
    def __init__(self, model_name: str, n_coord_feats: int, criterion,
                 num_classes: int, train_batch_size: int = None,
                 criterion_ignore_index: int = -100, optim_kwargs=None,
                 optimizer_ctor=None, lr_scheduler_ctor=None, label_smoothing=0.0):
        super().__init__()
        
        self.save_hyperparameters(ignore=["criterion", 'lr_scheduler_ctor', 'optimizer_ctor'])
        
        self.optim_kwargs = optim_kwargs or dict(lr=1e-4, weight_decay=0)
        self.model_name = model_name
        self.train_batch_size = train_batch_size
        self.label_smoothing = label_smoothing
        self.criterion_ignore_index = criterion_ignore_index
        
        self.optimizer_ctor = optimizer_ctor
        self.lr_scheduler_ctor = lr_scheduler_ctor
        
        self.model = MODEL_GETTERS_DICT[model_name](n_coord_feats=n_coord_feats)
        self.criterion = criterion
        
        self.train_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)
        self.val_token_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)
        self.train_token_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)
        self.val_token_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes, ignore_index=criterion_ignore_index)
    
    def forward(self, encoder_in, y, encoder_in_pad_mask, y_pad_mask):
        return self.model.forward(encoder_in, y, encoder_in_pad_mask, y_pad_mask)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_ctor(self.parameters(), **self.optim_kwargs)
        optimizers_configuration = {'optimizer': optimizer}
        
        if self.lr_scheduler_ctor:
            lr_scheduler = self.lr_scheduler_ctor(optimizer)
            optimizers_configuration['lr_scheduler'] = lr_scheduler
            optimizers_configuration['monitor'] = 'val_loss'
        
        return optimizers_configuration
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_size = batch_y.shape[-1]
        
        encoder_in, decoder_in, swipe_pad_mask, dec_seq_pad_mask = batch_x
        pred = self.forward(*batch_x)
        
        loss = self.criterion(pred, batch_y, ignore_index=self.criterion_ignore_index,
                              label_smoothing=self.label_smoothing)
        
        argmax_pred = torch.argmax(pred, dim=2)
        wl_accuracy = get_word_level_accuracy(
            argmax_pred.T, batch_y.T, pad_token=self.criterion_ignore_index, mask=dec_seq_pad_mask)
        
        flat_y = batch_y.reshape(-1)
        n_classes = pred.shape[-1]
        flat_preds = pred.reshape(-1, n_classes)
        
        self.train_token_acc(flat_preds, flat_y)
        self.log('train_token_level_accuracy', self.train_token_acc, on_step=True, on_epoch=False)
        
        self.train_token_f1(flat_preds, flat_y)
        self.log('train_token_level_f1', self.train_token_f1, on_step=True, on_epoch=False)
        
        self.log("train_word_level_accuracy", wl_accuracy, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_size = batch_y.shape[-1]
        
        encoder_in, decoder_in, swipe_pad_mask, dec_seq_pad_mask = batch_x
        pred = self.forward(*batch_x)
        
        loss = self.criterion(pred, batch_y, ignore_index=self.criterion_ignore_index,
                              label_smoothing=self.label_smoothing)
        
        argmax_pred = torch.argmax(pred, dim=2)
        wl_accuracy = get_word_level_accuracy(
            argmax_pred.T, batch_y.T, pad_token=self.criterion_ignore_index, mask=dec_seq_pad_mask)
        
        flat_y = batch_y.reshape(-1)
        n_classes = pred.shape[-1]
        flat_preds = pred.reshape(-1, n_classes)
        
        self.val_token_acc(flat_preds, flat_y)
        self.log('val_token_level_accuracy', self.val_token_acc, on_step=False, on_epoch=True)
        
        self.val_token_f1(flat_preds, flat_y)
        self.log('val_token_level_f1', self.val_token_f1, on_step=False, on_epoch=True)
        
        self.log("val_word_level_accuracy", wl_accuracy, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size)
        
        return loss


def main():
    parser = argparse.ArgumentParser(description="Train English swipe typing model")
    parser.add_argument("--config", type=str, default="configs/config_english.json",
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    training_config = config['training_config']
    
    # Set random seed
    init_random_seed(training_config['random_seed'])
    
    # Create tokenizers
    char_tokenizer = CharLevelTokenizerv2(training_config['voc_path'])
    kb_tokenizer = KeyboardTokenizerEnglish()
    
    # Get transforms
    DIST_WEIGHTS_FUNCS = {
        'weights_function_v1_softmax': weights_function_v1_softmax,
        'weights_function_v1': weights_function_v1,
        'weights_function_sigmoid_normalized_v1': weights_function_sigmoid_normalized_v1,
    }
    
    dist_weights_func = DIST_WEIGHTS_FUNCS[training_config['dist_weights_func_name']]
    
    train_transform, val_transform = get_transforms(
        gridname_to_grid_path=training_config['grid_path'],
        grid_names=[training_config['grid_name']],
        transform_name=training_config['transform_name'],
        char_tokenizer=kb_tokenizer,  # Pass keyboard tokenizer (confusing parameter name)
        uniform_noise_range=training_config['noise_range'],
        include_time=training_config['use_time'],
        include_velocities=training_config['use_velocity'],
        include_accelerations=training_config['use_acceleration'],
        dist_weights_func=dist_weights_func,
        ds_paths_list=[training_config['data_paths']['train'], 
                      training_config['data_paths']['val']],
        totals=(training_config['dataset_stats']['train_samples'],
                training_config['dataset_stats']['val_samples'])
    )
    
    # Create datasets
    train_dataset = CurveDataset(
        data_path=training_config['data_paths']['train'],
        store_gnames=False,
        init_transform=None,
        get_item_transform=train_transform,
        total=training_config['dataset_stats']['train_samples']
    )
    
    val_dataset = CurveDataset(
        data_path=training_config['data_paths']['val'],
        store_gnames=False,
        init_transform=None,
        get_item_transform=val_transform,
        total=training_config['dataset_stats']['val_samples']
    )
    
    # Create data loaders
    collate_fn = CollateFnV2(
        word_pad_idx=char_tokenizer.char_to_idx['<pad>'],
        batch_first=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size_train'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size_val'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    # Calculate number of coordinate features
    n_coord_feats = 2 * (training_config['use_coords'] + 
                        training_config['use_velocity'] + 
                        training_config['use_acceleration']) + training_config['use_time']
    
    # Create model
    pl_model = LitNeuroswipeModel(
        model_name=training_config['model_name'],
        criterion=cross_entropy_with_reshape,
        n_coord_feats=n_coord_feats,
        num_classes=len(char_tokenizer.idx_to_char),
        train_batch_size=training_config['batch_size_train'],
        criterion_ignore_index=char_tokenizer.char_to_idx['<pad>'],
        optim_kwargs=dict(lr=training_config['learning_rate'],
                         weight_decay=training_config['weight_decay']),
        optimizer_ctor=torch.optim.Adam,
        lr_scheduler_ctor=lambda opt: get_lr_scheduler(
            opt, 
            patience=training_config['lr_scheduler_patience'],
            factor=training_config['lr_scheduler_factor']
        ),
        label_smoothing=training_config['label_smoothing']
    )
    
    # Create callbacks
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=training_config['log_dir'],
        name=f"english_{training_config['model_name']}"
    )
    
    early_stopping_cb = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=training_config['early_stopping_patience']
    )
    
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=5,
        dirpath=training_config['checkpoint_dir'],
        filename='english-{epoch:02d}-{val_loss:.3f}-{val_word_level_accuracy:.3f}'
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=[early_stopping_cb, checkpoint_cb],
        logger=tb_logger,
        val_check_interval=training_config['val_check_interval'],
        log_every_n_steps=100,
        num_sanity_val_steps=0
    )
    
    # Train
    trainer.fit(pl_model, train_loader, val_loader, ckpt_path=args.checkpoint)
    
    print(f"Training complete! Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()