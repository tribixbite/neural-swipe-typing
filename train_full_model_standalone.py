#!/usr/bin/env python3
"""
Train character-level swipe typing model on full dataset to achieve 70% accuracy.
This uses the complete combined dataset for maximum performance.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from pathlib import Path
from tqdm import tqdm
import random


class KeyboardGrid:
    """Standalone QWERTY grid using normalized [0,1] coordinates."""

    def __init__(self):
        # Normalized canvas like your image
        self.width = 1.0
        self.height = 1.0

        row_h = 1.0 / 3.0
        # Row layouts
        top = list("qwertyuiop")        # 10 keys
        mid = list("asdfghjkl")         # 9 keys, offset by ~0.5 of a top key
        bot = list("zxcvbnm")           # 7 keys, offset by ~1.5 of a top key

        top_w = 1.0 / len(top)          # 0.1
        mid_w = 1.0 / len(mid)          # ≈0.111...
        bot_w = 1.0 / len(bot)          # ≈0.142857...

        mid_x0 = 0.5 * top_w            # 0.05
        bot_x0 = 1.5 * top_w            # 0.15

        self.qwerty = {"width": self.width, "height": self.height, "keys": []}
        self.key_positions = {}

        def add_row(keys, y0, x0, kw):
            for i, k in enumerate(keys):
                x = x0 + i * kw
                y = y0
                w = kw
                h = row_h
                cx, cy = x + w / 2.0, y + h / 2.0
                self.key_positions[k] = (cx, cy)
                self.qwerty["keys"].append({"label": k, "hitbox": {"x": x, "y": y, "w": w, "h": h}})

        # Build rows (y increases downward)
        add_row(top, 0.0 * row_h, 0.0,     top_w)  # qwertyuiop
        add_row(mid, 1.0 * row_h, mid_x0,  mid_w)  # asdfghjkl
        add_row(bot, 2.0 * row_h, bot_x0,  bot_w)  # zxcvbnm

        # Special tokens (center-ish and origin for pad)
        self.key_positions["<unk>"] = (0.5, 0.5)
        self.key_positions["<pad>"] = (0.0, 0.0)

    def get_nearest_key(self, x: float, y: float) -> str:
        nearest, dmin = "<unk>", float("inf")
        for label, (kx, ky) in self.key_positions.items():
            if label in ("<unk>", "<pad>"): 
                continue
            d = ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5
            if d < dmin:
                dmin, nearest = d, label
        return nearest


class CharTokenizer:
    """Character-level tokenizer matching the original."""

    def __init__(self):
        # Basic alphabet + special tokens
        chars = list("abcdefghijklmnopqrstuvwxyz")
        special = ["<pad>", "<unk>", "<sos>", "<eos>"]

        self.vocab = special + chars
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        self.pad_idx = self.char_to_idx["<pad>"]
        self.unk_idx = self.char_to_idx["<unk>"]
        self.sos_idx = self.char_to_idx["<sos>"]
        self.eos_idx = self.char_to_idx["<eos>"]

        self.vocab_size = len(self.vocab)

    def encode_word(self, word: str) -> List[int]:
        """Encode a word to token indices."""
        indices = [self.sos_idx]
        for char in word.lower():
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.unk_idx)
        indices.append(self.eos_idx)
        return indices

    def decode(self, indices: List[int]) -> str:
        """Decode token indices to word."""
        chars = []
        for idx in indices:
            if idx == self.sos_idx or idx == self.eos_idx:
                continue
            if idx == self.pad_idx:
                break
            chars.append(self.idx_to_char.get(idx, "?"))
        return "".join(chars)


class SwipeDataset(Dataset):
    """Dataset for swipe trajectories with character-level targets."""

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 250,
        max_word_len: int = 25,
        max_samples: int = None,
    ):
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len

        # Load keyboard grid
        self.keyboard = KeyboardGrid()
        self.tokenizer = CharTokenizer()

        # Load data
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                # Handle combined dataset format with curve field
                if "curve" in item and "word" in item:
                    curve = item["curve"]
                    if "x" in curve and "y" in curve and "t" in curve:
                        processed_item = {
                            "x": curve["x"],
                            "y": curve["y"],
                            "t": curve["t"],
                            "word": item["word"],
                            "grid_name": "qwerty_english",
                        }
                        self.data.append(processed_item)
                # Handle synthetic trace format
                elif "word_seq" in item:
                    word_seq = item["word_seq"]
                    if "x" in word_seq and "y" in word_seq and "time" in word_seq:
                        processed_item = {
                            "x": word_seq["x"],
                            "y": word_seq["y"],
                            "t": word_seq["time"],
                            "word": item.get("word", "unknown"),
                            "grid_name": "qwerty_english",
                        }
                        self.data.append(processed_item)
                elif "points" in item and isinstance(item["points"], list):
                    pts = item["points"]
                    xs = [p["x"] for p in pts]
                    ys = [p["y"] for p in pts]
                    ts = [p["t"] for p in pts]
                    processed_item = {
                        "x": xs,
                        "y": ys,
                        "t": ts,
                        "word": item.get("word", "unknown"),
                        "grid_name": "qwerty_english",
                    }
                    self.data.append(processed_item)
                elif "grid_name" in item and item["grid_name"] == "qwerty_english":
                    self.data.append(item)

                # Limit samples if specified (for faster iteration during development)
                if max_samples and len(self.data) >= max_samples:
                    break

        print(f"Loaded {len(self.data)} swipe examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract trajectory
        xs = np.array(item["x"], dtype=np.float32)
        ys = np.array(item["y"], dtype=np.float32)
        ts = np.array(item["t"], dtype=np.float32)

        # Normalize coordinates
        xs = xs / self.keyboard.width
        ys = ys / self.keyboard.height

        # Compute velocities and accelerations
        dt = np.diff(ts, prepend=ts[0])
        dt = np.maximum(dt, 1e-6)  # Avoid division by zero

        vx = np.zeros_like(xs)
        vy = np.zeros_like(ys)
        vx[1:] = np.diff(xs) / dt[1:]
        vy[1:] = np.diff(ys) / dt[1:]

        ax = np.zeros_like(xs)
        ay = np.zeros_like(ys)
        ax[1:] = np.diff(vx) / dt[1:]
        ay[1:] = np.diff(vy) / dt[1:]

        # Clip for stability
        vx = np.clip(vx, -10, 10)
        vy = np.clip(vy, -10, 10)
        ax = np.clip(ax, -10, 10)
        ay = np.clip(ay, -10, 10)

        # Get nearest keys for each point
        nearest_keys = []
        for x, y in zip(item["x"], item["y"]):
            key = self.keyboard.get_nearest_key(x, y)
            nearest_keys.append(
                self.tokenizer.char_to_idx.get(key, self.tokenizer.unk_idx)
            )

        # Stack trajectory features
        traj_features = np.stack([xs, ys, vx, vy, ax, ay], axis=1)

        # Pad or truncate to max_seq_len
        seq_len = len(xs)
        if seq_len > self.max_seq_len:
            traj_features = traj_features[: self.max_seq_len]
            nearest_keys = nearest_keys[: self.max_seq_len]
            seq_len = self.max_seq_len
        elif seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            traj_features = np.pad(
                traj_features, ((0, pad_len), (0, 0)), mode="constant"
            )
            nearest_keys = nearest_keys + [self.tokenizer.pad_idx] * pad_len

        # Encode target word
        word = item["word"]
        target_indices = self.tokenizer.encode_word(word)

        # Pad target to max_word_len
        if len(target_indices) > self.max_word_len:
            target_indices = target_indices[: self.max_word_len - 1] + [
                self.tokenizer.eos_idx
            ]
        else:
            pad_len = self.max_word_len - len(target_indices)
            target_indices = target_indices + [self.tokenizer.pad_idx] * pad_len

        return {
            "traj_features": torch.tensor(traj_features, dtype=torch.float32),
            "nearest_keys": torch.tensor(nearest_keys, dtype=torch.long),
            "target": torch.tensor(target_indices, dtype=torch.long),
            "seq_len": seq_len,
            "word": word,
        }


class CharacterLevelSwipeModel(nn.Module):
    """Character-level model that generates words like the original."""

    def __init__(
        self,
        traj_dim: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        kb_vocab_size: int = 30,
        char_vocab_size: int = 30,
        max_seq_len: int = 250,
    ):
        super().__init__()

        self.d_model = d_model

        # Encoder: Process trajectory
        self.traj_proj = nn.Linear(traj_dim, d_model // 2)
        self.kb_embedding = nn.Embedding(kb_vocab_size, d_model // 2)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder: Generate characters
        self.char_embedding = nn.Embedding(char_vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, char_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_trajectory(self, traj_features, nearest_keys, src_mask=None):
        """Encode the swipe trajectory."""
        batch_size, seq_len, _ = traj_features.shape

        # Project features
        traj_enc = self.traj_proj(traj_features)
        kb_enc = self.kb_embedding(nearest_keys)

        # Combine
        combined = torch.cat([traj_enc, kb_enc], dim=-1)
        combined = self.encoder_norm(combined)

        # Add positional encoding
        combined = combined + self.pe[:, :seq_len, :]

        # Encode
        memory = self.encoder(combined, src_key_padding_mask=src_mask)

        return memory

    def forward(
        self, traj_features, nearest_keys, targets, src_mask=None, tgt_mask=None
    ):
        """Forward pass with teacher forcing."""
        # Encode trajectory
        memory = self.encode_trajectory(traj_features, nearest_keys, src_mask)

        # Prepare target input (shift right, add <sos>)
        batch_size, tgt_len = targets.shape
        tgt_input = targets[:, :-1]  # Remove last token

        # Embed targets
        tgt_emb = self.char_embedding(tgt_input) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pe[:, : tgt_len - 1, :]

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len - 1).to(
            tgt_emb.device
        )

        # Decode
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(output)

        return logits

    @torch.no_grad()
    def generate_beam(
        self,
        traj_features,
        nearest_keys,
        tokenizer,
        src_mask=None,
        beam_size=5,
        max_len=20,
    ):
        """Generate word using beam search."""
        self.eval()

        # Encode trajectory
        memory = self.encode_trajectory(traj_features, nearest_keys, src_mask)
        batch_size = memory.shape[0]

        # Initialize beams
        beams = [[(0.0, [tokenizer.sos_idx])] for _ in range(batch_size)]

        for step in range(max_len):
            new_beams = [[] for _ in range(batch_size)]

            for b in range(batch_size):
                for score, seq in beams[b]:
                    # Skip finished sequences
                    if seq[-1] == tokenizer.eos_idx:
                        new_beams[b].append((score, seq))
                        continue

                    # Prepare input
                    tgt_input = torch.tensor([seq], device=memory.device)
                    tgt_emb = self.char_embedding(tgt_input) * math.sqrt(self.d_model)
                    tgt_emb = tgt_emb + self.pe[:, : len(seq), :]

                    # Decode
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(
                        len(seq)
                    ).to(memory.device)
                    output = self.decoder(
                        tgt_emb, memory[b : b + 1], tgt_mask=causal_mask
                    )

                    # Get next token probabilities
                    logits = self.output_proj(output[:, -1, :])
                    probs = F.log_softmax(logits, dim=-1).squeeze(0)

                    # Get top k tokens
                    top_probs, top_indices = torch.topk(
                        probs, min(beam_size, probs.shape[-1])
                    )

                    for prob, idx in zip(top_probs, top_indices):
                        new_score = score - prob.item()
                        new_seq = seq + [idx.item()]
                        new_beams[b].append((new_score, new_seq))

                # Keep top beam_size sequences
                new_beams[b].sort(key=lambda x: x[0])
                beams[b] = new_beams[b][:beam_size]

        # Get best sequences
        results = []
        for b in range(batch_size):
            best_seq = beams[b][0][1]
            word = tokenizer.decode(best_seq)
            results.append(word)

        return results


def train_full_model():
    """Train on full dataset to achieve target 70% accuracy."""

    # Configuration for full training
    batch_size = 128  # Larger batch for better gradient estimates
    learning_rate = 4e-5  # Slightly higher LR for faster convergence
    num_epochs = 500  # More epochs to reach target
    patience = 40  # Early stopping patience
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Training Full Character-Level Swipe Model")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Target: 70% word accuracy (matching original model)")
    print("-" * 60)

    # Use full combined dataset
    train_data_path = "data/train_hwsfuto.jsonl"
    val_data_path = "data/val_hwsfuto.jsonl"
    test_data_path = "data/test_hwsfuto.jsonl"

    # Check dataset exists
    if not os.path.exists(train_data_path):
        print(f"Error: Dataset not found at {train_data_path}")
        return

    print(f"Loading datasets...")

    # Load full datasets - no max_samples limit
    train_dataset = SwipeDataset(train_data_path)  # Full 68k samples
    val_dataset = SwipeDataset(val_data_path)  # Full validation set
    test_dataset = SwipeDataset(test_data_path)  # Test set for final eval

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print("-" * 60)

    # Create dataloaders with num_workers for faster loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model with optimal architecture
    tokenizer = CharTokenizer()
    model = CharacterLevelSwipeModel(
        traj_dim=6,
        d_model=256,  # Larger model for better capacity
        nhead=8,
        num_encoder_layers=6,  # Deeper encoder
        num_decoder_layers=4,  # Deeper decoder
        dim_feedforward=1024,  # Larger feedforward
        dropout=0.1,
        char_vocab_size=tokenizer.vocab_size,
        kb_vocab_size=tokenizer.vocab_size,
    ).to(device)

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    print(f"Model size (FP32): {param_count * 4 / 1024 / 1024:.2f} MB")
    print("-" * 60)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Learning rate scheduler - cosine annealing with warmup
    warmup_epochs = 2
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs / num_epochs,
        anneal_strategy="cos",
    )

    # --- START CHECKPOINT RESUME LOGIC ---

    start_epoch = 0
    best_val_acc = 0
    patience_counter = 0
    checkpoint_dir = Path("checkpoints/full_character_model_standalone_hwsfuto3")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Find the best (most accurate) checkpoint to resume from
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if checkpoint_files:
        try:
            # Sort files by accuracy (the last part of the filename)
            checkpoint_files.sort(
                key=lambda p: float(p.stem.split('-')[-1]), 
                reverse=True
            )
            latest_ckpt_path = checkpoint_files[0]

            print(f"Resuming from checkpoint: {latest_ckpt_path}")
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            start_epoch = checkpoint["epoch"] + 1
            best_val_acc = checkpoint.get("val_word_acc", 0) # Use .get for safety
            
            print(f"Resumed from Epoch {start_epoch}, Best Acc: {best_val_acc:.2%}")
        
        except Exception as e:
            print(f"WARNING: Could not load checkpoint. Starting from scratch. Error: {e}")
            start_epoch = 0
            best_val_acc = 0
    
    else:
        print("No checkpoint found. Starting from scratch.")

    # --- END CHECKPOINT RESUME LOGIC ---


    print("Starting training...")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            traj_features = batch["traj_features"].to(device)
            nearest_keys = batch["nearest_keys"].to(device)
            targets = batch["target"].to(device)

            # Create masks
            seq_lens = batch["seq_len"]
            src_mask = torch.zeros(
                traj_features.shape[0],
                traj_features.shape[1],
                dtype=torch.bool,
                device=device,
            )
            for i, seq_len in enumerate(seq_lens):
                src_mask[i, seq_len:] = True

            tgt_mask = targets[:, :-1] == tokenizer.pad_idx

            # Forward pass
            logits = model(traj_features, nearest_keys, targets, src_mask, tgt_mask)

            # Compute loss
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]), targets[:, 1:].reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track metrics
            train_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            mask = targets[:, 1:] != tokenizer.pad_idx
            train_correct += ((predictions == targets[:, 1:]) & mask).sum().item()
            train_total += mask.sum().item()

            # Update progress
            if batch_idx % 10 == 0:
                acc = train_correct / max(train_total, 1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{acc:.2%}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_correct_words = 0
        val_total_words = 0
        val_top5_correct = 0

        limit_val_batches = int(len(val_loader) * 1)
        if limit_val_batches == 0:
            limit_val_batches = 1 # Ensure at least one batch runs

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", total=limit_val_batches)
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= limit_val_batches:
                    break
                traj_features = batch["traj_features"].to(device)
                nearest_keys = batch["nearest_keys"].to(device)
                words = batch["word"]

                # Create masks
                seq_lens = batch["seq_len"]
                src_mask = torch.zeros(
                    traj_features.shape[0],
                    traj_features.shape[1],
                    dtype=torch.bool,
                    device=device,
                )
                for i, seq_len in enumerate(seq_lens):
                    src_mask[i, seq_len:] = True

                # Generate with beam search
                generated_words = model.generate_beam(
                    traj_features, nearest_keys, tokenizer, src_mask, beam_size=5
                )

                # Compute accuracy
                for gen_word, true_word in zip(generated_words, words):
                    val_total_words += 1
                    if gen_word == true_word:
                        val_correct_words += 1

                # Update progress
                word_acc = val_correct_words / max(val_total_words, 1)
                pbar.set_postfix({"word_acc": f"{word_acc:.2%}"})

        val_word_acc = val_correct_words / val_total_words

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Char Acc: {train_acc:.2%}")
        print(f"  Val   - Word Acc: {val_word_acc:.2%}")

        # Save checkpoint if improved
        if val_word_acc > best_val_acc:
            best_val_acc = val_word_acc
            patience_counter = 0

            checkpoint_path = (
                checkpoint_dir / f"full-model-{epoch + 1:02d}-{val_word_acc:.3f}.ckpt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_word_acc": val_word_acc,
                    "train_acc": train_acc,
                },
                checkpoint_path,
            )
            print(f"  ✓ New best model saved: {checkpoint_path}")

            # Check if target reached
            if val_word_acc >= 0.99:
                print("\n" + "=" * 60)
                print(f"🎉 TARGET ACHIEVED! {val_word_acc:.1%} word accuracy!")
                print("Successfully matched original model performance!")
                print("=" * 60)

                # Run test set evaluation
                print("\nEvaluating on test set...")
                test_correct = 0
                test_total = 0

                with torch.no_grad():
                    for batch in tqdm(test_loader, desc="Test"):
                        traj_features = batch["traj_features"].to(device)
                        nearest_keys = batch["nearest_keys"].to(device)
                        words = batch["word"]

                        seq_lens = batch["seq_len"]
                        src_mask = torch.zeros(
                            traj_features.shape[0],
                            traj_features.shape[1],
                            dtype=torch.bool,
                            device=device,
                        )
                        for i, seq_len in enumerate(seq_lens):
                            src_mask[i, seq_len:] = True

                        generated_words = model.generate_beam(
                            traj_features,
                            nearest_keys,
                            tokenizer,
                            src_mask,
                            beam_size=5,
                        )

                        for gen_word, true_word in zip(generated_words, words):
                            test_total += 1
                            if gen_word == true_word:
                                test_correct += 1

                test_acc = test_correct / test_total
                print(f"Test Set Accuracy: {test_acc:.2%}")
                break
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping - no improvement for {patience} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2%}")
                break

        print("-" * 60)

    if best_val_acc < 0.99:
        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2%}")
        print("Consider:")
        print("- Training for more epochs")
        print("- Adjusting hyperparameters")
        print("- Using data augmentation")


if __name__ == "__main__":
    train_full_model()
