# -*- coding: utf-8 -*-
"""
Preprocessing utilities for HEAT experiments
- Prepare a tiny text corpus (from datasets if available, else synthetic)
- Prompt noise injection
- Reproducibility helpers (shared with training)
"""
from typing import List

import os
import random
import numpy as np
import torch

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except Exception:
    DATASETS_AVAILABLE = False


# -------------------------
# Reproducibility & Device
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------
# Data utilities and noise
# -------------------------

def prepare_text_corpus(tokenizer, n_samples: int = 64, max_len: int = 128) -> List[str]:
    """Prepare a tiny corpus from datasets (if available) or synthetic text."""
    texts: List[str] = []
    if DATASETS_AVAILABLE:
        try:
            ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]')
            for rec in ds:
                t = rec['text']
                if t and not t.isspace():
                    texts.append(t)
                    if len(texts) >= n_samples:
                        break
        except Exception:
            pass
    # Fallback: synthetic patterns
    while len(texts) < n_samples:
        texts.append("This is a simple synthetic sentence used for quick HEAT testing. The model should learn patterns.")
    # Ensure tokenization length constraint for PPL eval
    truncated: List[str] = []
    for t in texts:
        ids = tokenizer(t, return_tensors='pt', truncation=True, max_length=max_len).input_ids[0].tolist()
        truncated.append(tokenizer.decode(ids))
    return truncated


def add_noise_to_prompt_ids(input_ids: torch.Tensor, vocab_size: int, noise_ratio: float = 0.1) -> torch.Tensor:
    ids = input_ids.clone()
    T = ids.size(1)
    n = max(1, int(T * noise_ratio))
    pos = torch.randperm(T)[:n]
    rand_tokens = torch.randint(0, vocab_size, (n,), device=ids.device)
    ids[:, pos] = rand_tokens
    return ids
