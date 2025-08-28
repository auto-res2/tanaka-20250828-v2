#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing and dataset/dataloader creation for RASST experiments.
- Supports WikiText-103 if datasets/transformers are available.
- Provides synthetic data generators for quick tests and offline runs.
"""
from typing import Optional, Tuple
import math
import numpy as np
import torch
from torch.utils.data import Dataset

# Optional external libs for real datasets
HAS_DATASETS = True
HAS_TRANSFORMERS = True
try:
    from datasets import load_dataset  # type: ignore
except Exception:
    HAS_DATASETS = False
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    HAS_TRANSFORMERS = False

from .train import TrainConfig  # for type hints only


class LMTextDataset(Dataset):
    """Language modeling dataset for WikiText-103.
    Returns (input_ids[:-1], input_ids[1:]) pairs.
    """
    def __init__(self, tokenizer, split: str = "train", seq_len: int = 1024, max_samples: Optional[int] = None):
        if not HAS_DATASETS:
            raise RuntimeError("datasets library not available. Use synthetic_* data_pattern.")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        text = "\n\n".join(ds["text"])  # naive concat
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        self.seq_len = seq_len
        chunks = [ids[i:i+seq_len+1] for i in range(0, len(ids)-seq_len-1, seq_len)]
        if max_samples is not None:
            chunks = chunks[:max_samples]
        self.samples = chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x = torch.tensor(x, dtype=torch.long)
        return x[:-1], x[1:]


class SyntheticLMTextDataset(Dataset):
    """Synthetic LM dataset for quick functional tests and robustness.
    Patterns:
    - 'synthetic_uniform': tokens sampled uniformly from [0, vocab_size).
    - 'synthetic_zipf': tokens sampled with Zipf-like distribution.
    - 'synthetic_repeating': repeating motif plus noise.
    """
    def __init__(self, vocab_size: int, seq_len: int, num_sequences: int, pattern: str = 'synthetic_uniform'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.pattern = pattern
        self.data = self._generate()

    def _generate(self):
        rng = np.random.default_rng(42)
        L = self.seq_len + 1
        data = []
        if self.pattern == 'synthetic_uniform':
            for _ in range(self.num_sequences):
                seq = rng.integers(0, self.vocab_size, size=L, endpoint=False, dtype=np.int64)
                data.append(torch.from_numpy(seq))
        elif self.pattern == 'synthetic_zipf':
            a = 1.3
            for _ in range(self.num_sequences):
                raw = rng.zipf(a, size=L)
                seq = (raw % self.vocab_size).astype(np.int64)
                data.append(torch.from_numpy(seq))
        elif self.pattern == 'synthetic_repeating':
            motif_len = max(4, self.seq_len // 8)
            motif = rng.integers(0, self.vocab_size, size=motif_len, endpoint=False, dtype=np.int64)
            for _ in range(self.num_sequences):
                rep = np.tile(motif, int(math.ceil(L / motif_len)))[:L]
                # add noise
                noise_idx = rng.choice(L, size=max(1, L // 10), replace=False)
                rep[noise_idx] = rng.integers(0, self.vocab_size, size=len(noise_idx), endpoint=False, dtype=np.int64)
                data.append(torch.from_numpy(rep))
        else:
            raise ValueError(f"Unknown synthetic pattern: {self.pattern}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1].long(), x[1:].long()


def get_tokenizer(name: str):
    if not HAS_TRANSFORMERS:
        return None
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def make_datasets(cfg: TrainConfig):
    tokenizer = None
    vocab_size = 50257

    if cfg.data_pattern.startswith('synthetic_'):
        # Synthetic data
        train_ds = SyntheticLMTextDataset(vocab_size=vocab_size, seq_len=cfg.seq_len,
                                          num_sequences=cfg.max_train_samples or 4096,
                                          pattern=cfg.data_pattern)
        val_ds = SyntheticLMTextDataset(vocab_size=vocab_size, seq_len=cfg.seq_len,
                                        num_sequences=cfg.max_val_samples or 512,
                                        pattern=cfg.data_pattern)
    else:
        if not (HAS_DATASETS and HAS_TRANSFORMERS):
            raise RuntimeError("datasets/transformers not available; use synthetic_* data_pattern.")
        tokenizer = get_tokenizer(cfg.vocab)
        vocab_size = tokenizer.vocab_size
        train_ds = LMTextDataset(tokenizer, split="train", seq_len=cfg.seq_len, max_samples=cfg.max_train_samples)
        val_ds = LMTextDataset(tokenizer, split="validation", seq_len=cfg.seq_len, max_samples=cfg.max_val_samples)

    return train_ds, val_ds, tokenizer, vocab_size
