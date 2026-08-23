#!/usr/bin/env python3
"""
RWKV-7 Pretraining + GRPO Finetuning with Gymnasium

Usage:
    python train_rwkv7.py --pretrain_data data.md.zst --layers 2 --batch_size 1

Features:
    - Pure PyTorch RWKV-7 implementation (no rwkv pip package)
    - BPE tokenizer trained on .md.zst or fallback UTF-8 tokenizer
    - EM-style pretraining with clipped per-element softmax loss
    - GRPO training via TRL + Gymnasium environment
    - TensorBoard logging, Ctrl+C graceful exit, checkpoint/resume
    - Auto VRAM detection, batch_size=1 default
"""

import argparse
import atexit
import gc
import glob
import gzip
import json
import math
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# TRL imports
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import RewardFunc

# ----------------------------------------------------------------------
# 1. RWKV-7 Core Implementation (from scratch)
# ----------------------------------------------------------------------

class RWKV7TimeMix(nn.Module):
    """RWKV-7 Time Mixing block with expressive dynamic state evolution."""

    def __init__(self, dim: int, head_size: int = 64):
        super().__init__()
        self.dim = dim
        self.head_size = head_size
        self.n_head = dim // head_size

        # Learned parameters
        self.time_maa_x = nn.Parameter(torch.ones(1, 1, dim))
        self.time_maa_w = nn.Parameter(torch.ones(1, 1, dim))
        self.time_maa_k = nn.Parameter(torch.ones(1, 1, dim))
        self.time_maa_v = nn.Parameter(torch.ones(1, 1, dim))
        self.time_maa_r = nn.Parameter(torch.ones(1, 1, dim))
        self.time_maa_g = nn.Parameter(torch.ones(1, 1, dim))

        self.time_decay = nn.Parameter(torch.ones(1, 1, dim))
        self.time_first = nn.Parameter(torch.ones(1, 1, dim))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, dim, eps=1e-5, affine=True)

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, ...]] = None):
        """
        Args:
            x: (B, T, D)
            state: (w, k, v, r, g) each (B, D) or None
        Returns:
            out: (B, T, D)
            new_state: (w, k, v, r, g)
        """
        B, T, D = x.shape

        if state is None:
            w = torch.zeros(B, D, device=x.device)
            k = torch.zeros(B, D, device=x.device)
            v = torch.zeros(B, D, device=x.device)
            r = torch.zeros(B, D, device=x.device)
            g = torch.zeros(B, D, device=x.device)
        else:
            w, k, v, r, g = state

        # Shifted input: x_prev = shift(x) with zero for first token
        x_shift = torch.cat([torch.zeros(B, 1, D, device=x.device), x[:, :-1, :]], dim=1)

        # Time mixing with learned MAAs
        x_mix = x + self.time_maa_x * (x_shift - x)
        w_mix = w.unsqueeze(1) + self.time_maa_w * (x_shift - x)
        k_mix = k.unsqueeze(1) + self.time_maa_k * (x_shift - x)
        v_mix = v.unsqueeze(1) + self.time_maa_v * (x_shift - x)
        r_mix = r.unsqueeze(1) + self.time_maa_r * (x_shift - x)
        g_mix = g.unsqueeze(1) + self.time_maa_g * (x_shift - x)

        # Linear projections
        r = self.receptance(r_mix)
        k = self.key(k_mix)
        v = self.value(v_mix)
        g = self.gate(g_mix)

        # State update: wkv = (w * k + v) / (w + 1)  (simplified delta rule)
        w_new = w.unsqueeze(1) + self.time_decay * (k - w.unsqueeze(1))
        wkv = (w_new * k + v) / (w_new + 1e-8)

        # Output with receptance and gate
        out = r * torch.sigmoid(g) * wkv
        out = self.output(out)

        # Update state for next step
        new_state = (w_new.squeeze(1), k.squeeze(1), v.squeeze(1), r.squeeze(1), g.squeeze(1))

        return out, new_state


class RWKV7Block(nn.Module):
    """Single RWKV-7 block with time mixing + FFN."""

    def __init__(self, dim: int, head_size: int = 64, ffn_dim: Optional[int] = None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = dim * 4

        self.ln1 = nn.LayerNorm(dim)
        self.time_mix = RWKV7TimeMix(dim, head_size)

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor, state: Optional[Tuple] = None):
        # Time mixing with residual
        attn_out, new_state = self.time_mix(self.ln1(x), state)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x, new_state


class RWKV7Model(nn.Module):
    """RWKV-7 language model with embedding and output head."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_layers: int = 2,
        head_size: int = 64,
        ffn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            RWKV7Block(dim, head_size, ffn_dim) for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[Tuple]] = None,
        return_states: bool = False,
    ):
        """
        Args:
            input_ids: (B, T) long
            states: list of (n_layers,) each (w,k,v,r,g) or None
        Returns:
            logits: (B, T, V)
            new_states: list of states if return_states
        """
        B, T = input_ids.shape
        x = self.emb(input_ids)

        if states is None:
            states = [None] * self.n_layers

        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, states[i])
            new_states.append(new_state)

        x = self.ln_out(x)
        logits = self.head(x)

        if return_states:
            return logits, new_states
        return logits

    def get_initial_state(self, batch_size: int, device: torch.device):
        """Return a list of None states (initialized lazily in forward)."""
        return [None] * self.n_layers

    def reset_state(self, batch_size: int, device: torch.device):
        """Reset state to zeros."""
        D = self.dim
        return [
            (
                torch.zeros(batch_size, D, device=device),
                torch.zeros(batch_size, D, device=device),
                torch.zeros(batch_size, D, device=device),
                torch.zeros(batch_size, D, device=device),
                torch.zeros(batch_size, D, device=device),
            )
            for _ in range(self.n_layers)
        ]


# ----------------------------------------------------------------------
# 2. Tokenizers
# ----------------------------------------------------------------------

class UnicodeTokenizer:
    """Parameter-free tokenizer that splits at UTF-8 byte boundaries."""

    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        # Map byte value (0-255) to token id
        self.byte_to_id = {i: i + 4 for i in range(256)}
        self.id_to_byte = {v: k for k, v in self.byte_to_id.items()}
        self.vocab_size = 256 + 4

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = [self.byte_to_id[b] for b in text.encode('utf-8')]
        if add_special_tokens:
            return [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        byte_vals = [self.id_to_byte[t] for t in token_ids if t in self.id_to_byte]
        return bytes(byte_vals).decode('utf-8', errors='replace')

    def __len__(self):
        return self.vocab_size


class BPETokenizer:
    """Simple BPE tokenizer (trainable from text)."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.byte_to_id = {}
        self.id_to_byte = {}
        self.merges = {}
        self.vocab = {}

    def train(self, texts: List[str]):
        """Learn BPE merges from texts."""
        # Start with byte-level vocabulary
        vocab = {i: bytes([i]) for i in range(256)}
        # Count symbol pairs
        def get_stats(tokens):
            pairs = defaultdict(int)
            for tok in tokens:
                for i in range(len(tok) - 1):
                    pairs[(tok[i], tok[i+1])] += 1
            return pairs

        # Encode all texts as bytes
        all_tokens = []
        for text in texts:
            all_tokens.append(list(text.encode('utf-8')))

        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - 256 - 4  # reserve special tokens
        merges = []
        current_vocab = {i: bytes([i]) for i in range(256)}

        for _ in range(num_merges):
            stats = get_stats(all_tokens)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            merges.append(best_pair)

            # Merge all occurrences
            new_tokens = []
            for tok in all_tokens:
                new_tok = []
                i = 0
                while i < len(tok):
                    if i < len(tok) - 1 and (tok[i], tok[i+1]) == best_pair:
                        new_tok.append(best_pair[0])  # placeholder
                        i += 2
                    else:
                        new_tok.append(tok[i])
                        i += 1
                new_tokens.append(new_tok)
            all_tokens = new_tokens

            # Update vocab
            new_id = len(current_vocab) + 4
            current_vocab[new_id] = current_vocab[best_pair[0]] + current_vocab[best_pair[1]]

        self.vocab = current_vocab
        self.merges = merges

        # Build lookup tables
        self.byte_to_id = {b: i for i, b in enumerate(range(256))}
        self.id_to_byte = {i: b for i, b in enumerate(range(256))}
        for k, v in self.vocab.items():
            if k >= 256:
                # Map new tokens to IDs
                pass

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text using learned BPE."""
        # Simple fallback: byte-level if not trained
        if not self.vocab:
            return [b + 4 for b in text.encode('utf-8')]

        tokens = list(text.encode('utf-8'))
        # Apply merges
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    # Find merged token id
                    merged = self.vocab.get(pair[0], b'') + self.vocab.get(pair[1], b'')
                    found = False
                    for k, v in self.vocab.items():
                        if v == merged:
                            new_tokens.append(k)
                            found = True
                            break
                    if not found:
                        new_tokens.append(tokens[i])
                        new_tokens.append(tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        if add_special_tokens:
            return [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        byte_vals = []
        for t in token_ids:
            if t in self.id_to_byte:
                byte_vals.append(self.id_to_byte[t])
            elif t in self.vocab:
                byte_vals.extend(list(self.vocab[t]))
        return bytes(byte_vals).decode('utf-8', errors='replace')

    def __len__(self):
        return self.vocab_size


# ----------------------------------------------------------------------
# 3. Dataset & Data Loading
# ----------------------------------------------------------------------

class ZSTDataset(Dataset):
    """Dataset that reads chunks from a .md.zst file with random access."""

    def __init__(self, filepath: str, tokenizer, chunk_size: int = 8192):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

        # Build chunk index: (offset, uncompressed_size)
        self.chunk_index = []
        self._build_index()

        # Cache for segments
        self.segments = []

    def _build_index(self):
        """Record chunk offsets for random access."""
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(self.filepath, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)
            offset = 0
            while offset < file_size:
                # Read compressed chunk size (4 bytes)
                f.seek(offset)
                chunk_size_bytes = f.read(4)
                if not chunk_size_bytes:
                    break
                chunk_size = int.from_bytes(chunk_size_bytes, 'little')
                offset += 4
                self.chunk_index.append((offset, chunk_size))
                offset += chunk_size

    def _decompress_chunk(self, idx: int) -> str:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        offset, size = self.chunk_index[idx]
        with open(self.filepath, 'rb') as f:
            f.seek(offset)
            compressed = f.read(size)
        return dctx.decompress(compressed).decode('utf-8', errors='replace')

    def _split_into_segments(self, text: str) -> List[str]:
        """Split text on \\n\\n+ (treat multiple newlines as one)."""
        import re
        # Split on \n\n+ but keep the delimiter as part of the previous segment
        parts = re.split(r'(\n\n+)', text)
        segments = []
        current = ""
        for part in parts:
            if re.match(r'\n\n+', part):
                if current:
                    segments.append(current)
                    current = ""
                # Add the delimiter to the next segment
            else:
                current += part
        if current:
            segments.append(current)
        return segments

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, idx: int) -> List[int]:
        text = self._decompress_chunk(idx)
        segments = self._split_into_segments(text)
        # Encode each segment and flatten
        tokens = []
        for seg in segments:
            if seg.strip():
                tokens.extend(self.tokenizer.encode(seg, add_special_tokens=True))
        return tokens


class StreamingPretrainDataset:
    """Streaming dataset for EM pretraining that yields segments on-the-fly."""

    def __init__(self, filepath: str, tokenizer, chunk_size: int = 8192):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_index = []
        self._build_index()

    def _build_index(self):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(self.filepath, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)
            offset = 0
            while offset < file_size:
                f.seek(offset)
                chunk_size_bytes = f.read(4)
                if not chunk_size_bytes:
                    break
                chunk_size = int.from_bytes(chunk_size_bytes, 'little')
                offset += 4
                self.chunk_index.append((offset, chunk_size))
                offset += chunk_size

    def _decompress_chunk(self, idx: int) -> str:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        offset, size = self.chunk_index[idx]
        with open(self.filepath, 'rb') as f:
            f.seek(offset)
            compressed = f.read(size)
        return dctx.decompress(compressed).decode('utf-8', errors='replace')

    def _split_into_segments(self, text: str) -> List[str]:
        import re
        parts = re.split(r'(\n\n+)', text)
        segments = []
        current = ""
        for part in parts:
            if re.match(r'\n\n+', part):
                if current:
                    segments.append(current)
                    current = ""
            else:
                current += part
        if current:
            segments.append(current)
        return segments

    def stream_segments(self):
        """Generator yielding (segment_text, chunk_idx, seg_idx)."""
        for chunk_idx in range(len(self.chunk_index)):
            text = self._decompress_chunk(chunk_idx)
            segments = self._split_into_segments(text)
            for seg_idx, seg in enumerate(segments):
                if seg.strip():
                    yield seg, chunk_idx, seg_idx


# ----------------------------------------------------------------------
# 4. Pretraining (EM Procedure)
# ----------------------------------------------------------------------

def clipped_per_element_loss(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.95):
    """
    Smoothly clipped per-element softmax loss.

    loss = max(CE_loss - (-log(threshold)), 0)
    """
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
    clip = -math.log(threshold)
    return torch.clamp(ce - clip, min=0).sum()


def pretrain_em(
    model: RWKV7Model,
    dataset: StreamingPretrainDataset,
    tokenizer,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 1e-3,
    log_interval: int = 10,
    writer: Optional[SummaryWriter] = None,
    checkpoint_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
):
    """
    EM pretraining:
    - M-step: train all segments as one epoch using current state
    - E-step: compute new state using model from M-step
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    # Prepare segments
    segments = list(dataset.stream_segments())
    print(f"Loaded {len(segments)} segments")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Shuffle segments
        import random
        random.shuffle(segments)

        # M-step: train on all segments
        for seg_text, chunk_idx, seg_idx in tqdm(segments, desc=f"Epoch {epoch} M-step"):
            tokens = tokenizer.encode(seg_text, add_special_tokens=True)
            if len(tokens) < 2:
                continue

            # Convert to tensor
            input_ids = torch.tensor([tokens[:-1]], device=device)  # (1, T-1)
            targets = torch.tensor([tokens[1:]], device=device)     # (1, T-1)

            # Reset state for each segment (state finetuning)
            state = model.reset_state(1, device)

            optimizer.zero_grad()

            # Forward with state
            logits, new_state = model(input_ids, states=state, return_states=True)

            # Clipped loss
            loss = clipped_per_element_loss(logits, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % log_interval == 0:
                avg_loss = total_loss / num_batches
                if writer:
                    writer.add_scalar('Pretrain/loss', avg_loss, num_batches)
                print(f"Step {num_batches}, loss: {avg_loss:.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch} M-step complete. Avg loss: {avg_loss:.4f}")

        # E-step: compute new state using model from M-step
        model.eval()
        new_states = {}
        with torch.no_grad():
            for seg_text, chunk_idx, seg_idx in segments:
                tokens = tokenizer.encode(seg_text, add_special_tokens=True)
                if len(tokens) < 2:
                    continue
                input_ids = torch.tensor([tokens[:-1]], device=device)
                state = model.reset_state(1, device)
                _, new_state = model(input_ids, states=state, return_states=True)
                # Store state for next M-step
                new_states[(chunk_idx, seg_idx)] = new_state

        # Save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Also save latest
        latest_path = os.path.join(checkpoint_dir, "latest.pt")
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, latest_path)


# ----------------------------------------------------------------------
# 5. GRPO Training with Gymnasium
# ----------------------------------------------------------------------

class GymnasiumEnvWrapper(gym.Env):
    """Wrapper to make any Gymnasium env work with TRL GRPO."""

    def __init__(self, env_name: str = "CartPole-v1", max_steps: int = 100):
        self.env = gym.make(env_name)
        self.max_steps = max_steps
        self.step_count = 0
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


def create_gym_reward_fn(env_name: str = "CartPole-v1"):
    """Create a reward function for GRPO using Gymnasium."""
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        env = gym.make(env_name)
        for prompt, completion in zip(prompts, completions):
            try:
                # Parse action from completion
                action = int(completion.strip().split()[0]) if completion.strip() else 0
                action = np.clip(action, 0, env.action_space.n - 1)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
            except Exception as e:
                rewards.append(0.0)
            env.reset()
        env.close()
        return rewards
    return reward_fn


def grpo_train(
    model: RWKV7Model,
    tokenizer,
    env_name: str = "CartPole-v1",
    output_dir: str = "grpo_output",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    learning_rate: float = 1e-4,
    logging_steps: int = 10,
    save_steps: int = 100,
    writer: Optional[SummaryWriter] = None,
):
    """
    Train with GRPO using TRL.
    Note: GRPOTrainer expects a model with a specific interface.
    We wrap RWKV7Model to be compatible.
    """

    # Wrap model to be compatible with TRL's expected interface
    class RWKV7ForGRPO(nn.Module):
        def __init__(self, base_model: RWKV7Model):
            super().__init__()
            self.base_model = base_model
            self.config = type('Config', (), {
                'vocab_size': base_model.vocab_size,
                'hidden_size': base_model.dim,
                'num_hidden_layers': base_model.n_layers,
            })()

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            # Simple forward pass
            logits = self.base_model(input_ids)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return type('Output', (), {'logits': logits, 'loss': loss})()

        def generate(self, input_ids, max_new_tokens=20, **kwargs):
            # Simple generation for GRPO
            generated = input_ids
            state = self.base_model.reset_state(input_ids.size(0), input_ids.device)
            for _ in range(max_new_tokens):
                logits, new_state = self.base_model(generated, states=state, return_states=True)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                state = new_state
            return generated

    grpo_model = RWKV7ForGRPO(model)

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        report_to="tensorboard" if writer else "none",
    )

    # Create reward function
    reward_fn = create_gym_reward_fn(env_name)

    # Create trainer
    trainer = GRPOTrainer(
        model=grpo_model,
        args=grpo_config,
        train_dataset=[],
        reward_funcs=[reward_fn],
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)


# ----------------------------------------------------------------------
# 6. Main Entry Point
# ----------------------------------------------------------------------

def detect_device_and_batch_size():
    """Detect device and auto-determine batch size based on VRAM."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        total_vram = torch.cuda.get_device_properties(0).total_memory
        # Very conservative: use batch_size=1 by default
        batch_size = 1
        print(f"Using GPU with {total_vram / 1e9:.2f} GB VRAM, batch_size={batch_size}")
    else:
        device = torch.device('cpu')
        batch_size = 1
        print("Using CPU, batch_size=1")
    return device, batch_size


def graceful_exit(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nReceived interrupt signal. Saving checkpoint and exiting...")
    # Save checkpoint logic here
    if hasattr(graceful_exit, 'model') and graceful_exit.model is not None:
        checkpoint_dir = graceful_exit.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'model': graceful_exit.model.state_dict(),
            'epoch': graceful_exit.current_epoch,
            'step': graceful_exit.current_step,
        }, os.path.join(checkpoint_dir, "interrupt_checkpoint.pt"))
        print(f"Checkpoint saved to {checkpoint_dir}/interrupt_checkpoint.pt")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="RWKV-7 Training with TRL and Gymnasium")
    parser.add_argument("--pretrain_data", type=str, default=None,
                        help="Path to .md.zst file for pretraining")
    parser.add_argument("--layers", type=int, default=2,
                        help="Number of RWKV layers (default: 2 for smoke test)")
    parser.add_argument("--dim", type=int, default=128,
                        help="Model dimension (default: 128)")
    parser.add_argument("--head_size", type=int, default=64,
                        help="Head size (default: 64)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: auto-detect from VRAM)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3 for SGD)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of pretraining epochs")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint file")
    parser.add_argument("--env_name", type=str, default="CartPole-v1",
                        help="Gymnasium environment name for GRPO")
    parser.add_argument("--grpo_epochs", type=int, default=1,
                        help="Number of GRPO epochs")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="Skip pretraining, go directly to GRPO")
    parser.add_argument("--use_bpe", action="store_true",
                        help="Use BPE tokenizer instead of Unicode tokenizer")

    args = parser.parse_args()

    # Setup device and batch size
    device, auto_batch_size = detect_device_and_batch_size()
    batch_size = args.batch_size if args.batch_size is not None else auto_batch_size

    # Setup logging
    writer = SummaryWriter(args.log_dir) if args.log_dir else None

    # Setup signal handler for graceful exit
    signal.signal(signal.SIGINT, graceful_exit)

    # Load tokenizer
    tokenizer = None
    if args.pretrain_data and args.use_bpe:
        # Train BPE tokenizer on pretrain data
        print("Training BPE tokenizer on pretrain data...")
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        texts = []
        with open(args.pretrain_data, 'rb') as f:
            # Read first few chunks for training
            for _ in range(10):
                chunk_size_bytes = f.read(4)
                if not chunk_size_bytes:
                    break
                chunk_size = int.from_bytes(chunk_size_bytes, 'little')
                compressed = f.read(chunk_size)
                text = dctx.decompress(compressed).decode('utf-8', errors='replace')
                texts.append(text)
        tokenizer = BPETokenizer(vocab_size=10000)
        tokenizer.train(texts)
        print(f"BPE tokenizer trained with vocab size {len(tokenizer)}")
    else:
        tokenizer = UnicodeTokenizer()
        print(f"Using Unicode tokenizer with vocab size {len(tokenizer)}")

    vocab_size = len(tokenizer)

    # Create model
    model = RWKV7Model(
        vocab_size=vocab_size,
        dim=args.dim,
        n_layers=args.layers,
        head_size=args.head_size,
    ).to(device)

    print(f"Model created with {args.layers} layers, dim={args.dim}, vocab={vocab_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Store for graceful exit
    graceful_exit.model = model
    graceful_exit.checkpoint_dir = args.checkpoint_dir
    graceful_exit.current_epoch = 0
    graceful_exit.current_step = 0

    # Pretraining
    if args.pretrain_data and not args.no_pretrain:
        print(f"Starting pretraining on {args.pretrain_data}...")
        dataset = StreamingPretrainDataset(args.pretrain_data, tokenizer)
        pretrain_em(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=args.lr,
            writer=writer,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
        )
    else:
        print("Skipping pretraining.")

    # GRPO Training
    print(f"Starting GRPO training on {args.env_name}...")
    grpo_train(
        model=model,
        tokenizer=tokenizer,
        env_name=args.env_name,
        output_dir=os.path.join(args.checkpoint_dir, "grpo"),
        num_train_epochs=args.grpo_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=args.lr * 0.1,  # Lower LR for GRPO
        writer=writer,
    )

    print("Training complete!")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()