#!/usr/bin/env python3
"""
Standalone export and quantization script for swipe typing models.
Creates optimized ONNX models for both web and Android deployment.

Usage:
    uv run export_and_quantize_standalone.py path/to/checkpoint target_directory
"""
import inspect
import os
import sys
import json
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

# ONNX related imports
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization import quantize_static, CalibrationDataReader


# Model classes (copied from train_full_model_standalone.py)
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


def load_checkpoint(checkpoint_path: Path) -> Tuple[CharacterLevelSwipeModel, str, Dict]:
    """Load model from checkpoint and return model, accuracy, and config."""
    print(f"Loading checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Initialize tokenizer and get config
    tokenizer = CharTokenizer()

    # Model architecture read from checkpoint config if present
    ck_config = checkpoint.get('config', {})
    d_model = int(ck_config.get('d_model', 256))
    nhead = int(ck_config.get('nhead', 8))
    num_encoder_layers = int(ck_config.get('num_encoder_layers', 6))
    num_decoder_layers = int(ck_config.get('num_decoder_layers', 4))
    dim_feedforward = int(ck_config.get('dim_feedforward', 1024))
    max_seq_len = int(ck_config.get('max_seq_len', 250))
    traj_dim = int(ck_config.get('traj_dim', 6))

    model = CharacterLevelSwipeModel(
        traj_dim=traj_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,  # No dropout for inference
        char_vocab_size=tokenizer.vocab_size,
        kb_vocab_size=tokenizer.vocab_size,
        max_seq_len=max_seq_len
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    accuracy = checkpoint.get('val_word_acc', 0.0)
    print(f"Model loaded: {accuracy:.1%} word accuracy")

    config = {
        'accuracy': accuracy,
        'd_model': d_model,
        'vocab_size': tokenizer.vocab_size,
        'max_seq_len': max_seq_len,
        'max_word_len': int(ck_config.get('max_word_len', 20))
    }

    return model, float(accuracy), config



def export_encoder_onnx(model: CharacterLevelSwipeModel, output_path: Path, opset: int = 17) -> Dict:
    """Export encoder to ONNX format."""
    print(f"Exporting encoder to: {output_path}")

    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, traj_features, nearest_keys, src_mask):
            return self.model.encode_trajectory(traj_features, nearest_keys, src_mask)

    wrapper = EncoderWrapper(model)
    wrapper.eval()

    # Create sample inputs for 250 sequence length
    batch_size = 1
    seq_len = 250
    traj_features = torch.randn(batch_size, seq_len, 6)
    nearest_keys = torch.randint(0, 30, (batch_size, seq_len))
    src_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Dynamic axes for variable length sequences (encoder)
    dynamic_axes = {
        'trajectory_features': {0: 'batch', 1: 'sequence'},
        'nearest_keys': {0: 'batch', 1: 'sequence'},
        'src_mask': {0: 'batch', 1: 'sequence'},
        'encoder_output': {0: 'batch', 1: 'sequence'}
    }



    # Export
    torch.onnx.export(
        wrapper,
        (traj_features, nearest_keys, src_mask),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['trajectory_features', 'nearest_keys', 'src_mask'],
        output_names=['encoder_output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Encoder exported: {size_mb:.2f} MB")

    # Validate using onnxruntime
    try:
        so = ort.SessionOptions()
        sess = ort.InferenceSession(str(output_path), so, providers=['CPUExecutionProvider'])
        _ = sess.run(None, {
            'trajectory_features': traj_features.numpy(),
            'nearest_keys': nearest_keys.numpy(),
            'src_mask': src_mask.numpy()
        })
    except Exception as e:
        print(f"⚠ Encoder ONNX runtime check failed: {e}")
        raise


    return {'path': str(output_path), 'size_mb': size_mb}


def export_decoder_onnx(model: CharacterLevelSwipeModel, output_path: Path, opset: int = 17) -> Dict:
    """Export decoder to ONNX format."""
    print(f"Exporting decoder to: {output_path}")

    class DecoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.d_model = model.d_model

        def forward(self, memory, tgt_tokens, src_mask, target_padding_mask, target_causal_mask):
            """
            Inputs:
            - memory: (B, enc_seq, d_model) float32
            - tgt_tokens: (B, tgt_len) int64
            - src_mask: (B, enc_seq) boolean/int (True=PAD)
            - target_padding_mask: (B, tgt_len) boolean/int (True=PAD)
            - target_causal_mask: (tgt_len, tgt_len) float32 (0.0 allowed, -1e9 blocked)
            """

            # Ensure src_mask is boolean (coerce numeric masks)
            src_mask = src_mask.bool()

            # Embed target tokens as before
            batch_size, tgt_len = tgt_tokens.shape
            tgt_emb = self.model.char_embedding(tgt_tokens) * math.sqrt(self.d_model)
            tgt_emb = tgt_emb + self.model.pe[:, :tgt_len, :]

            # Coerce masks directly (avoid tracer branching)
            tgt_key_padding_mask = target_padding_mask.bool() if target_padding_mask is not None else None
            causal_mask = target_causal_mask.to(dtype=tgt_emb.dtype, device=tgt_emb.device) if target_causal_mask is not None else None

            # Call the decoder
            output = self.model.decoder(
                tgt_emb,
                memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            logits = self.model.output_proj(output)
            return logits

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    # Sample inputs
    batch_size = 1
    seq_len = 250
    tgt_len = 20
    memory = torch.randn(batch_size, seq_len, model.d_model)
    tgt_tokens = torch.randint(0, 30, (batch_size, tgt_len), dtype=torch.long)
    src_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    target_padding_mask = torch.zeros(batch_size, tgt_len, dtype=torch.bool)
    target_causal_mask = torch.zeros(tgt_len, tgt_len, dtype=torch.float32)

    dynamic_axes = {
        'memory': {0: 'batch', 1: 'enc_sequence'},
        'target_tokens': {0: 'batch', 1: 'dec_sequence'},
        'src_mask': {0: 'batch', 1: 'enc_sequence'},
        'target_padding_mask': {0: 'batch', 1: 'dec_sequence'},
        'target_causal_mask': {0: 'dec_sequence', 1: 'dec_sequence'},
        'logits': {0: 'batch', 1: 'dec_sequence'}
    }


    torch.onnx.export(
        wrapper,
        (memory, tgt_tokens, src_mask, target_padding_mask, target_causal_mask),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['memory', 'target_tokens', 'src_mask', 'target_padding_mask', 'target_causal_mask'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Decoder exported: {size_mb:.2f} MB")

    # Validate with onnxruntime
    try:
        so = ort.SessionOptions()
        sess = ort.InferenceSession(str(output_path), so, providers=['CPUExecutionProvider'])
        _ = sess.run(None, {
            'memory': memory.numpy(),
            'target_tokens': tgt_tokens.numpy(),
            'src_mask': src_mask.numpy(),
            'target_padding_mask': target_padding_mask.numpy(),
            'target_causal_mask': target_causal_mask.numpy()
        })
    except Exception as e:
        print(f"⚠ Decoder ONNX runtime check failed: {e}")
        raise


    return {'path': str(output_path), 'size_mb': size_mb}


def try_simplify_onnx(input_path: Path, output_path: Path, strict: bool = True) -> float:
    """Simplify ONNX graph using onnxsim. Returns size in MB of simplified model.
    If onnxsim is not available, attempts to install, otherwise copies input to output.
    """
    print(f"Simplifying {input_path.name}")
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    try:
        try:
            from onnxsim import simplify
        except ImportError:
            import subprocess, sys as _sys
            print("  onnxsim not found; installing...")
            subprocess.check_call([_sys.executable, '-m', 'pip', 'install', 'onnxsim'])
            from onnxsim import simplify
        model = onnx.load(str(input_path))
        model_simp, check = simplify(model)
        if not check:
            raise RuntimeError("onnxsim check failed")
        onnx.save(model_simp, str(output_path))
    except Exception as e:
        if strict:
            raise SystemExit(
                f"Simplification failed: {e}. Rerun with --no-simplify to skip graph simplification."
            )
        else:
            print(f"  ⚠ Simplify failed: {e}. Using original model.")
            import shutil
            shutil.copyfile(str(input_path), str(output_path))
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Simplified: {size_mb:.2f} MB ({(1 - size_mb/original_size)*100:.1f}% vs raw)")
    return size_mb


def preprocess_onnx_model(input_path: Path, output_path: Path) -> float:
    """Apply ONNX preprocessing before quantization."""
    print(f"Preprocessing {input_path.name}")

    original_size = os.path.getsize(input_path) / (1024 * 1024)
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(
        str(input_path),
        str(output_path),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
        auto_merge=True,
        int_max=2**31-1,
        guess_output_rank=False,
        verbose=0,
        save_as_external_data=False,
        all_tensors_to_one_file=True,
        external_data_location="",
        external_data_size_threshold=1024,
        )
    except Exception:
        import onnx
        from onnx import shape_inference
        model = onnx.load(str(input_path))
        model = shape_inference.infer_shapes(model)
        onnx.save(model, str(output_path))

    

    processed_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - processed_size/original_size) * 100
    print(f"  Preprocessed: {processed_size:.2f} MB ({reduction:+.1f}%)")

    return processed_size


def quantize_onnx_model(input_path: Path, output_path: Path, target: str = "android",
                        use_static: bool = True,
                        calibration_data: Optional[str] = None) -> float:
    """Apply dynamic quantization to ONNX model."""
    print(f"Quantizing {input_path.name} for {target}")

    original_size = os.path.getsize(input_path) / (1024 * 1024)

    # If target is android and static requested, require calibration data
    did_static = False
    if target == "android" and use_static:
        if calibration_data is None:
            raise SystemExit(
                "Static quantization is enabled by default for Android but no calibration data was provided. "
                "Provide --calibration-data path/to/train.jsonl or rerun with --dynamic-only to use dynamic quantization."
            )
        try:
            print("  Using static quantization with calibration data")

            class DummyDataReader(CalibrationDataReader):
                def __init__(self, model_path: str, calib_path: str, max_samples: int = 50):
                    self.model_path = model_path
                    self.calib_path = calib_path
                    self.max_samples = max_samples
                    self._iter = None

                def _build_inputs(self):
                    """
                    Build a list of input dicts for calibration. This supports both encoder
                    and decoder ONNX graphs. If self.model_path indicates a decoder ONNX,
                    we will attempt to run the corresponding encoder ONNX to produce
                    'memory' inputs for decoder calibration.
                    """
                    inputs = []
                    is_decoder = "decoder" in Path(self.model_path).name.lower()

                    # Locate potential encoder ONNX to produce memory (for decoder calibration)
                    encoder_onnx_for_decoder = None
                    if is_decoder:
                        # Try to find an encoder sibling file by replacing 'decoder' with 'encoder'
                        try:
                            dec_name = Path(self.model_path).name
                            enc_name = dec_name.replace("decoder", "encoder")
                            enc_path_candidate = Path(self.model_path).parent / enc_name
                            if enc_path_candidate.exists():
                                encoder_onnx_for_decoder = str(enc_path_candidate)
                        except Exception:
                            encoder_onnx_for_decoder = None

                        # As a fallback, try generic name 'swipe_encoder_*_base.onnx' in same dir
                        if encoder_onnx_for_decoder is None:
                            for p in Path(self.model_path).parent.glob("swipe_encoder*base.onnx"):
                                encoder_onnx_for_decoder = str(p)
                                break

                    # Initialize encoder session if we will use it
                    encoder_sess = None
                    if encoder_onnx_for_decoder is not None:
                        try:
                            encoder_sess = ort.InferenceSession(encoder_onnx_for_decoder, providers=['CPUExecutionProvider'])
                        except Exception:
                            encoder_sess = None

                    try:
                        with open(self.calib_path, 'r') as f:
                            for i, line in enumerate(f):
                                if i >= self.max_samples:
                                    break
                                try:
                                    item = json.loads(line)
                                except Exception:
                                    continue

                                # Extract curve/points/word_seq as before
                                if 'curve' in item and 'x' in item['curve']:
                                    x = item['curve']['x']; y = item['curve']['y']; t = item['curve']['t']
                                elif 'points' in item:
                                    pts = item['points']; x=[p['x'] for p in pts]; y=[p['y'] for p in pts]; t=[p['t'] for p in pts]
                                elif 'word_seq' in item:
                                    x=item['word_seq']['x']; y=item['word_seq']['y']
                                    t=item['word_seq'].get('time', list(range(len(x))))
                                else:
                                    continue

                                L = min(len(x), 250)
                                # Build encoder inputs (B=1)
                                traj = np.zeros((1, 250, 6), dtype=np.float32)
                                nk = np.zeros((1, 250), dtype=np.int64)
                                sm = np.zeros((1, 250), dtype=np.bool_)  # boolean mask (True=PAD)
                                xs = np.array(x[:L], dtype=np.float32)
                                ys = np.array(y[:L], dtype=np.float32)
                                ts = np.array(t[:L], dtype=np.float32)
                                # simple normalization
                                xs = xs / max(xs.max(), 1.0)
                                ys = ys / max(ys.max(), 1.0)
                                dt = np.diff(ts, prepend=ts[0]); dt = np.maximum(dt, 1e-6)
                                vx = np.zeros_like(xs); vy = np.zeros_like(ys)
                                vx[1:] = np.diff(xs)/dt[1:]; vy[1:] = np.diff(ys)/dt[1:]
                                ax = np.zeros_like(xs); ay = np.zeros_like(ys)
                                ax[1:] = np.diff(vx)/dt[1:]; ay[1:] = np.diff(vy)/dt[1:]
                                traj[0,:L,:] = np.stack([xs, ys, vx, vy, ax, ay], axis=1)
                                sm[0,L:] = True

                                if not is_decoder:
                                    # Encoder calibration sample
                                    inputs.append({'trajectory_features': traj, 'nearest_keys': nk, 'src_mask': sm})
                                else:
                                    # Decoder calibration: need to produce 'memory' (encoder output)
                                    memory = None
                                    if encoder_sess is not None:
                                        try:
                                            # Run encoder ONNX to get memory
                                            out = encoder_sess.run(None, {
                                                'trajectory_features': traj,
                                                'nearest_keys': nk,
                                                'src_mask': sm
                                            })
                                            # encoder_output is the first output by design
                                            memory = np.asarray(out[0]).astype(np.float32)
                                        except Exception:
                                            memory = None

                                    if memory is None:
                                        # Fallback: random memory of appropriate shape (B, enc_seq, d_model)
                                        # We try to infer d_model from the graph name or default to 256
                                        d_model_val = 256
                                        # Attempt to read d_model from model filename if present
                                        nm = Path(self.model_path).name
                                        # default shapes
                                        memory = np.random.randn(1, 250, d_model_val).astype(np.float32)

                                    # Build decoder inputs: target_tokens, target_padding_mask, target_causal_mask
                                    # Use a safe vocab size (30, matching training tokenizer)
                                    vocab_size = 30
                                    max_tgt = 20
                                    # Simple synthetic target: SOS at pos0 (index 2), rest random tokens with some pads
                                    tgt_tokens = np.zeros((1, max_tgt), dtype=np.int64)
                                    tgt_tokens[0,0] = 2  # <sos> index in your tokenizer
                                    # fill a few subsequent tokens randomly
                                    seq_len_t = min(5, max_tgt-1)
                                    tgt_tokens[0,1:1+seq_len_t] = np.random.randint(3, vocab_size, size=(seq_len_t,))
                                    # padding mask True where PAD; assume pad index 0 is pad, so mark zeros after seq
                                    target_padding_mask = np.zeros((1, max_tgt), dtype=np.bool_)
                                    if seq_len_t + 1 < max_tgt:
                                        target_padding_mask[0, seq_len_t+1:] = True
                                    # causal mask: (T,T) float32, 0 on allowed, -1e9 for masked positions
                                    T = max_tgt
                                    causal = np.triu(np.full((T,T), -1e9, dtype=np.float32), 1)
                                    inputs.append({
                                        'memory': memory,
                                        'target_tokens': tgt_tokens,
                                        'src_mask': sm,
                                        'target_padding_mask': target_padding_mask,
                                        'target_causal_mask': causal
                                    })
                    except Exception:
                        pass

                    # if nothing, fallback to a couple of random samples
                    if not inputs:
                        for _ in range(10):
                            traj = np.random.randn(1,250,6).astype(np.float32)
                            nk = np.random.randint(0,30,size=(1,250)).astype(np.int64)
                            sm = np.zeros((1,250), dtype=np.bool_)
                            if not is_decoder:
                                inputs.append({'trajectory_features': traj, 'nearest_keys': nk, 'src_mask': sm})
                            else:
                                # random memory + dummy decoder inputs
                                memory = np.random.randn(1,250,256).astype(np.float32)
                                tgt_tokens = np.random.randint(1,30,size=(1,20)).astype(np.int64)
                                target_padding_mask = (tgt_tokens == 0)
                                T = 20
                                causal = np.triu(np.full((T,T), -1e9, dtype=np.float32), 1)
                                inputs.append({'memory': memory, 'target_tokens': tgt_tokens, 'src_mask': sm, 'target_padding_mask': target_padding_mask, 'target_causal_mask': causal})

                    return inputs

                def get_next(self):
                    # lazy build
                    if self._iter is None:
                        inputs = self._build_inputs()
                        self._iter = iter(inputs)
                    try:
                        return next(self._iter)
                    except StopIteration:
                        return None

                def rewind(self):
                    # ORT may call rewind() between calibration passes
                    self._iter = None

            dr = DummyDataReader(str(input_path), calibration_data)

            # Call quantize_static in a backwards/forwards-compatible way:
            # some ort versions accept optimize_model, some don't — inspect the signature and only pass supported kwargs.
            _try_msg = None
            try:
                sig = inspect.signature(quantize_static)
                supported = set(sig.parameters.keys())
                call_kwargs = {}
                # desired kwargs (keep minimal and essential)
                desired_kwargs = {'weight_type': QuantType.QInt8, 'per_channel': False}
                for k, v in desired_kwargs.items():
                    if k in supported:
                        call_kwargs[k] = v
                # prefer explicit optimize_model=False when available
                if 'optimize_model' in supported:
                    call_kwargs['optimize_model'] = False

                quantize_static(str(input_path), str(output_path), dr, **call_kwargs)
                did_static = True
            except Exception as e:
                _try_msg = str(e)
                # Final fallback: try calling with only required args (some ORT builds accept this)
                try:
                    quantize_static(str(input_path), str(output_path), dr)
                    did_static = True
                except Exception as e2:
                    raise SystemExit(
                        f"Static quantization failed: {e}. Fallback attempt also failed: {e2}. "
                        "Rerun with --dynamic-only to use dynamic quantization instead."
                    )

        except Exception as e:
            raise SystemExit(
                f"Static quantization failed: {e}. Rerun with --dynamic-only to use dynamic quantization instead."
            )

    if not did_static:
        if target == "web":
            # Web-optimized dynamic quantization
            quantize_dynamic(
                str(input_path),
                str(output_path),
                weight_type=QuantType.QUInt8,
                per_channel=False,
                reduce_range=False,
            )
        else:  # android
            # Android-optimized dynamic quantization
            quantize_dynamic(
                str(input_path),
                str(output_path),
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=True,
            )

    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size/original_size) * 100
    print(f"  Quantized: {quantized_size:.2f} MB ({reduction:.1f}% smaller)")

    return quantized_size


def create_config_files(output_dir: Path, accuracy: str, config: Dict):
    """Create tokenizer and model configuration files."""

    # Tokenizer config
    tokenizer = CharTokenizer()
    tokenizer_config = {
        'vocab_size': tokenizer.vocab_size,
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
        'special_tokens': {
            'pad_token': '<pad>',
            'pad_idx': tokenizer.pad_idx,
            'eos_token': '<eos>',
            'eos_idx': tokenizer.eos_idx,
            'unk_token': '<unk>',
            'unk_idx': tokenizer.unk_idx,
            'sos_token': '<sos>',
            'sos_idx': tokenizer.sos_idx
        }
    }

    tokenizer_path = output_dir / 'tokenizer_config.json'
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"✓ Tokenizer config: {tokenizer_path}")

    # Model config
    model_config = {
        'model_type': 'character_level_transformer',
        'architecture': {
            'trajectory_dim': 6,
            'd_model': config['d_model'],
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 4,
            'dim_feedforward': 1024,
            'vocab_size': config['vocab_size'],
            'max_seq_length': config['max_seq_len'],
            'max_word_length': config['max_word_len']
        },
        'feature_extraction': {
            'input_features': ['x', 'y', 'vx', 'vy', 'ax', 'ay'],
            'normalization': {
                'keyboard_width': 1.0,
                'keyboard_height': 1.0,
                'velocity_clipping': [-10, 10],
                'acceleration_clipping': [-10, 10]
            }
        },
        'inference': {
            'inputs': {
                'memory': {'shape': ['B', 'enc_seq', 'd_model'], 'dtype': 'float32', 'meaning': 'Encoder output memory'},
                'target_tokens': {'shape': ['B', 'dec_seq'], 'dtype': 'int64', 'meaning': 'Decoder input token ids (including <sos>)'},
                'src_mask': {'shape': ['B', 'enc_seq'], 'dtype': 'bool', 'meaning': 'Encoder padding mask (True = PAD)'},
                'target_padding_mask': {'shape': ['B', 'dec_seq'], 'dtype': 'bool', 'meaning': 'Decoder padding mask (True = PAD)'},
                'target_causal_mask': {'shape': ['dec_seq', 'dec_seq'], 'dtype': 'float32', 'meaning': 'Decoder causal mask (0.0 allowed, large negative/-1e9 blocked)'}
            },
            'beam_size': 5,
            'max_length': 20,
            'length_penalty': 1.0,
            'temperature': 1.0
        },

        'performance': {
            'word_accuracy': config['accuracy'],
            'deployment_targets': ['web', 'android']
        }
    }

    model_path = output_dir / 'model_config.json'
    with open(model_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✓ Model config: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Export and quantize swipe typing model')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint file')
    parser.add_argument('output_dir', type=str, help='Output directory for ONNX files')
    parser.add_argument('--with-preprocessing', action='store_true', help='Enable ONNX preprocessing (may fail)')
    parser.add_argument('--targets', nargs='+', default=['web', 'android'],
                       choices=['web', 'android'], help='Target platforms')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version (default 17)')
    parser.add_argument('--clean-base', action='store_true', help='Remove raw and simplified ONNX files after creating optimized models')
    parser.add_argument('--calibration-data', type=str, default=None, help='Path to JSONL calibration data for static quant (Android)')
    parser.add_argument('--dynamic-only', action='store_true', help='Use dynamic quantization instead of static (Android)')
    parser.add_argument('--no-simplify', action='store_true', help='Skip ONNX graph simplification step')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Swipe Model Export and Quantization")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Targets: {', '.join(args.targets)}")
    print("-" * 60)

    # Load model
    model, accuracy, config = load_checkpoint(checkpoint_path)

    # Export base ONNX models
    print("\n=== ONNX Export ===")
    encoder_path = output_dir / 'swipe_encoder.onnx'
    decoder_path = output_dir / 'swipe_decoder.onnx'

    encoder_info = export_encoder_onnx(model, encoder_path, opset=args.opset)
    decoder_info = export_decoder_onnx(model, decoder_path, opset=args.opset)

    total_original_mb = encoder_info['size_mb'] + decoder_info['size_mb']
    print(f"Total original size: {total_original_mb:.2f} MB")

    # Preprocessing and quantization
    results = {}

    for target in args.targets:
        print(f"\n=== {target.upper()} Optimization ===")

        # File paths
        enc_prep = output_dir / f'swipe_encoder_{target}_preprocessed.onnx'
        dec_prep = output_dir / f'swipe_decoder_{target}_preprocessed.onnx'
        # Simplified bases
        enc_simp = output_dir / f'swipe_encoder_{target}_base.onnx'
        dec_simp = output_dir / f'swipe_decoder_{target}_base.onnx'
        enc_final = output_dir / f'swipe_encoder_{target}.onnx'
        dec_final = output_dir / f'swipe_decoder_{target}.onnx'

        # Step 1: Preprocessing (addresses the warning)
        if args.with_preprocessing:
            try:
                enc_prep_size = preprocess_onnx_model(encoder_path, enc_prep)
                dec_prep_size = preprocess_onnx_model(decoder_path, dec_prep)
                prep_input_enc, prep_input_dec = enc_prep, dec_prep
            except Exception as e:
                print(f"⚠ Preprocessing failed: {e}")
                print("Continuing without preprocessing...")
                prep_input_enc, prep_input_dec = encoder_path, decoder_path
                enc_prep_size = encoder_info['size_mb']
                dec_prep_size = decoder_info['size_mb']
        else:
            print("Skipping preprocessing (use --with-preprocessing to enable)")
            prep_input_enc, prep_input_dec = encoder_path, decoder_path
            enc_prep_size = encoder_info['size_mb']
            dec_prep_size = decoder_info['size_mb']

        # Step 2: Simplify graphs (default enabled; fail fast if issues)
        if args.no_simplify:
            import shutil
            shutil.copyfile(str(prep_input_enc), str(enc_simp))
            shutil.copyfile(str(prep_input_dec), str(dec_simp))
        else:
            enc_simp_size = try_simplify_onnx(prep_input_enc, enc_simp, strict=True)
            dec_simp_size = try_simplify_onnx(prep_input_dec, dec_simp, strict=True)

        # Step 3: Quantization
        use_static = (target == 'android' and not args.dynamic_only)
        enc_quant_size = quantize_onnx_model(enc_simp, enc_final, target, use_static=use_static, calibration_data=args.calibration_data)
        dec_quant_size = quantize_onnx_model(dec_simp, dec_final, target, use_static=use_static, calibration_data=args.calibration_data)

        total_final_mb = enc_quant_size + dec_quant_size
        total_reduction = (1 - total_final_mb / total_original_mb) * 100

        results[target] = {
            'encoder_mb': enc_quant_size,
            'decoder_mb': dec_quant_size,
            'total_mb': total_final_mb,
            'reduction_pct': total_reduction
        }

        print(f"  Final {target} size: {total_final_mb:.2f} MB ({total_reduction:.1f}% reduction)")

        # Clean up intermediate files
        if args.with_preprocessing:
            enc_prep.unlink(missing_ok=True)
            dec_prep.unlink(missing_ok=True)
        if args.clean_base:
            # remove per-target simplified intermediates
            enc_simp.unlink(missing_ok=True)
            dec_simp.unlink(missing_ok=True)

    # Create config files
    print("\n=== Configuration Files ===")
    create_config_files(output_dir, accuracy, config)

    # Keep raw ONNX by default; if requested, remove after all targets processed
    if args.clean_base:
        encoder_path.unlink(missing_ok=True)
        decoder_path.unlink(missing_ok=True)

    # Summary
    print("\n" + "=" * 60)
    print("✅ Export Complete!")
    print("=" * 60)
    print(f"📦 Output directory: {output_dir}")
    print(f"🎯 Model accuracy: {float(accuracy)*100:.1f}%")
    print("\nOptimized models:")

    for target, info in results.items():
        print(f"  {target.upper()}:")
        print(f"    - swipe_encoder_{target}.onnx ({info['encoder_mb']:.2f} MB)")
        print(f"    - swipe_decoder_{target}.onnx ({info['decoder_mb']:.2f} MB)")
        print(f"    - Total: {info['total_mb']:.2f} MB ({info['reduction_pct']:.1f}% smaller)")

    print("\nConfiguration files:")
    print("  - tokenizer_config.json")
    print("  - model_config.json")

    print(f"\n📱 Ready for deployment!")


if __name__ == "__main__":
    main()
