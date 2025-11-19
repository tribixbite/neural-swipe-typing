#!/usr/bin/env python3
"""
Test script for evaluating the accuracy of the exported ONNX models in the 'alpha' directory.
It uses a sample of 100 words from the validation set, including specific required words.
"""

import json
import math
import random
from pathlib import Path
from typing import List, Dict
import os

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# --- Start of re-used components from training/export scripts ---

class KeyboardGrid:
    """Standalone QWERTY grid using normalized [0,1] coordinates."""

    def __init__(self):
        self.width = 1.0
        self.height = 1.0
        row_h = 1.0 / 3.0
        key_w = 1.0 / 10.0
        top = list("qwertyuiop")
        mid = list("asdfghjkl")
        bot = list("zxcvbnm")
        top_x0, mid_x0, bot_x0 = 0.0, 0.05, 0.15
        self.key_positions = {}

        def add_row(keys, y0, x0):
            for i, k in enumerate(keys):
                cx, cy = x0 + i * key_w + key_w / 2.0, y0 + row_h / 2.0
                self.key_positions[k] = (cx, cy)

        add_row(top, 0.0 * row_h, top_x0)
        add_row(mid, 1.0 * row_h, mid_x0)
        add_row(bot, 2.0 * row_h, bot_x0)
        self.key_positions["<unk>"] = (0.5, 0.5)
        self.key_positions["<pad>"] = (0.0, 0.0)
        self._labels = [k for k in self.key_positions.keys() if k not in ("<unk>", "<pad>")]
        self._pos = np.array([self.key_positions[k] for k in self._labels], dtype=np.float32)

    def get_nearest_key_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> List[str]:
        if xs.size == 0: return []
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        diff = coords[:, None, :] - self._pos[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(d2, axis=1)
        return [self._labels[i] for i in idx]

class CharTokenizer:
    """Character-level tokenizer matching the training script."""
    def __init__(self):
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

    def decode(self, indices: List[int]) -> str:
        chars = []
        for idx in indices:
            if idx in (self.sos_idx, self.eos_idx): continue
            if idx == self.pad_idx: break
            chars.append(self.idx_to_char.get(idx, "?"))
        return "".join(chars)

class SwipeDataset(Dataset):
    """Dataset for swipe trajectories, adapted from train_full_model_standalone.py"""

    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 250,
        max_word_len: int = 25,
        max_samples: int = None,
        samples: List[Dict] = None, # Allow passing in pre-filtered samples
    ):
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        self.keyboard = KeyboardGrid()
        self.tokenizer = CharTokenizer()

        if samples:
            self.data = samples
        else:
            self.data = []
            with open(data_path, "r") as f:
                for line in f:
                    self.data.append(json.loads(line))
                    if max_samples and len(self.data) >= max_samples:
                        break
        
        # This internal list will hold the raw data dictionaries
        self.raw_data = self.data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        raw_item = self.raw_data[idx]
        
        # Logic to handle multiple data formats
        if "curve" in raw_item and "x" in raw_item["curve"]:
            curve_data = raw_item["curve"]
            xs = np.array(curve_data["x"], dtype=np.float32) / self.keyboard.width
            ys = np.array(curve_data["y"], dtype=np.float32) / self.keyboard.height
            ts = np.array(curve_data["t"], dtype=np.float32)
        elif "word_seq" in raw_item:
            word_seq = raw_item["word_seq"]
            xs, ys, ts = word_seq["x"], word_seq["y"], word_seq["time"]
        elif "points" in raw_item and isinstance(raw_item["points"], list):
            pts = raw_item["points"]
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            ts = [p["t"] for p in pts]
        else:
            xs, ys, ts = raw_item["x"], raw_item["y"], raw_item["t"]

        xs = np.array(xs, dtype=np.float32) / self.keyboard.width
        ys = np.array(ys, dtype=np.float32) / self.keyboard.height
        ts = np.array(ts, dtype=np.float32)

        dt = np.diff(ts, prepend=ts[0])
        dt = np.maximum(dt, 1e-6)

        vx = np.zeros_like(xs); vy = np.zeros_like(ys)
        vx[1:] = np.diff(xs) / dt[1:]
        vy[1:] = np.diff(ys) / dt[1:]

        ax = np.zeros_like(xs); ay = np.zeros_like(ys)
        ax[1:] = np.diff(vx) / dt[1:]
        ay[1:] = np.diff(vy) / dt[1:]

        vx, vy = np.clip(vx, -10, 10), np.clip(vy, -10, 10)
        ax, ay = np.clip(ax, -10, 10), np.clip(ay, -10, 10)

        near_labels = self.keyboard.get_nearest_key_vectorized(xs, ys)
        nearest_keys = [self.tokenizer.char_to_idx.get(k, self.tokenizer.unk_idx) for k in near_labels]

        traj_features = np.stack([xs, ys, vx, vy, ax, ay], axis=1)

        seq_len = len(xs)
        if seq_len > self.max_seq_len:
            traj_features = traj_features[:self.max_seq_len]
            nearest_keys = nearest_keys[:self.max_seq_len]
            seq_len = self.max_seq_len
        elif seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            traj_features = np.pad(traj_features, ((0, pad_len), (0, 0)), mode="constant")
            nearest_keys.extend([self.tokenizer.pad_idx] * pad_len)

        return {
            "trajectory_features": traj_features.astype(np.float32),
            "nearest_keys": np.array(nearest_keys, dtype=np.int32),
            "actual_length": np.array([seq_len], dtype=np.int32),
            "word": raw_item["word"],
        }

# --- End of re-used components ---

def generate_beam_onnx(encoder_session, decoder_session, preprocessed_input, tokenizer, beam_size=5, max_len=20):
    """Generates a word using beam search with ONNX models."""
    
    encoder_inputs = {
        "trajectory_features": np.expand_dims(preprocessed_input["trajectory_features"], 0),
        "nearest_keys": np.expand_dims(preprocessed_input["nearest_keys"], 0),
        "actual_length": preprocessed_input["actual_length"],
    }
    memory = encoder_session.run(None, encoder_inputs)[0]

    beams = [(0.0, [tokenizer.sos_idx])]

    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == tokenizer.eos_idx:
                new_beams.append((score, seq))
                continue

            # --- WORKAROUND for hardcoded sequence length in ONNX model ---
            # Pad the sequence to the fixed length (max_len) the model was exported with.
            padded_seq = seq + [tokenizer.pad_idx] * (max_len - len(seq))

            decoder_inputs = {
                "memory": memory,
                "target_tokens": np.array([padded_seq], dtype=np.int32),
                "actual_src_length": preprocessed_input["actual_length"],
            }
            log_probs = decoder_session.run(None, decoder_inputs)[0]
            
            # Get log probs for the *actual* last token, not the padded one.
            last_log_probs = log_probs[0, len(seq) - 1, :]
            
            top_k_log_probs, top_k_indices = torch.topk(torch.from_numpy(last_log_probs), beam_size)

            for i in range(beam_size):
                new_seq = seq + [top_k_indices[i].item()]
                new_score = score + top_k_log_probs[i].item()
                new_beams.append((new_score, new_seq))
        
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

    _, best_seq = beams[0]
    return tokenizer.decode(best_seq)


def main():
    """Main function to run the accuracy test."""
    model_dir = Path("alpha")
    data_file = Path("data/val_hwsfuto.jsonl")
    num_samples = 100
    required_words = {'only', 'way', 'zen', 'expand'}

    if not model_dir.exists() or not data_file.exists():
        print(f"Error: Model dir '{model_dir}' or data file '{data_file}' not found.")
        return

    print("Loading ONNX models...")
    encoder_session = ort.InferenceSession(str(model_dir / "swipe_encoder_android.onnx"))
    decoder_session = ort.InferenceSession(str(model_dir / "swipe_decoder_android.onnx"))

    with open(model_dir / "model_config.json", "r") as f:
        model_config = json.load(f)
    max_seq_len = model_config["limits"]["max_seq_len"]

    print(f"Loading and sampling data from {data_file}...")
    all_data = [json.loads(line) for line in open(data_file)]
    
    selected_samples = []
    remaining_data = []
    
    # Grab required words
    found_words = set()
    for item in all_data:
        if item['word'] in required_words and item['word'] not in found_words:
            selected_samples.append(item)
            found_words.add(item['word'])
        else:
            remaining_data.append(item)

    # Sample the rest
    num_to_sample = num_samples - len(selected_samples)
    if num_to_sample > 0 and remaining_data:
        selected_samples.extend(random.sample(remaining_data, min(num_to_sample, len(remaining_data))))

    print(f"Testing on {len(selected_samples)} samples.")

    # Use the full dataset class for preprocessing
    eval_dataset = SwipeDataset(data_path=str(data_file), max_seq_len=max_seq_len, samples=selected_samples)

    correct_predictions = 0
    pbar = tqdm(range(len(eval_dataset)), desc="Evaluating")
    
    for i in pbar:
        preprocessed = eval_dataset[i]
        
        predicted_word = generate_beam_onnx(
            encoder_session, decoder_session, preprocessed, eval_dataset.tokenizer
        )
        
        true_word = preprocessed["word"]
        
        if predicted_word == true_word:
            correct_predictions += 1
            
        accuracy = correct_predictions / (i + 1)
        pbar.set_postfix({"accuracy": f"{accuracy:.2%}"})

    final_accuracy = correct_predictions / len(eval_dataset)
    print(f"\nFinal Word Accuracy: {final_accuracy:.2%}")
    print(f"Correct: {correct_predictions} / Total: {len(eval_dataset)}")

if __name__ == "__main__":
    main()