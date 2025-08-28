import os
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import psutil
except Exception:
    psutil = None


# ---------------------
# Reproducibility & device
# ---------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_info() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    return torch.device("cpu"), "cpu"


def peak_mem_gb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        return peak / (1024 ** 3)
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    return 0.0


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------
# Synthetic datasets
# ---------------------

class SyntheticSeqClassDataset(Dataset):
    """Sequence classification dataset with simple rules.
    Patterns:
      - 'sum_mod_k': label = sum(tokens) % n_classes
      - 'threshold': label = 1 if any(token >= vocab_size//2) else 0
      - 'pattern_mix': mixture of rules
    """
    def __init__(self, n: int, seq_len: int, vocab_size: int, n_classes: int = 3, pattern: str = 'sum_mod_k', seed: int = 0):
        super().__init__()
        set_seed(seed)
        self.n = n
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.pattern = pattern
        self.x = torch.randint(0, vocab_size, (n, seq_len), dtype=torch.long)
        self.y = self._make_labels(self.x)

    def _make_labels(self, x: torch.Tensor) -> torch.Tensor:
        if self.pattern == 'sum_mod_k':
            return (x.sum(dim=1) % self.n_classes).long()
        elif self.pattern == 'threshold':
            y = (x.ge(self.vocab_size // 2).any(dim=1)).long()
            return y.clamp(0, 1)
        elif self.pattern == 'pattern_mix':
            half = x.size(0) // 2
            y1 = (x[:half].sum(dim=1) % self.n_classes).long()
            y2 = (x[half:].ge(self.vocab_size // 2).any(dim=1)).long()
            y2 = y2.clamp(0, min(self.n_classes - 1, 1))
            return torch.cat([y1, y2], dim=0)
        else:
            return (x.sum(dim=1) % self.n_classes).long()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"input_ids": self.x[idx], "labels": self.y[idx]}


class SyntheticLMDataset(Dataset):
    """Simple language modeling dataset: sequences of integers; predict next token."""
    def __init__(self, n: int, seq_len: int, vocab_size: int, seed: int = 0):
        super().__init__()
        set_seed(seed)
        self.n = n
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.x = torch.randint(0, vocab_size, (n, seq_len), dtype=torch.long)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        inp = self.x[idx]
        labels = inp.clone()
        return {"input_ids": inp, "labels": labels}
