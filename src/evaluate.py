#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation utilities and plotting helpers for RASST experiments.
- estimate_ppl, peak_memory_report
- plotting helpers that save high-quality PDFs to .research/iteration1/images
"""
import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# High-quality PDF settings
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

IMAGES_DIR = os.path.join('.research', 'iteration1', 'images')

def _ensure_images_dir() -> str:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    return IMAGES_DIR


@torch.no_grad()
def estimate_ppl(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), yb.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += yb.numel()
    model.train()
    if total_tokens == 0:
        return float('nan')
    return math.exp(total_loss / total_tokens)


def peak_memory_report() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {'alloc_mb': 0.0, 'reserve_mb': 0.0}
    torch.cuda.synchronize()
    return {
        'alloc_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'reserve_mb': torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


# ------------------------------
# Plotting helpers (save as PDF)
# ------------------------------

def plot_training_loss(log: Dict, condition: str):
    _ensure_images_dir()
    plt.figure(figsize=(6, 4))
    plt.plot(log['steps'], log['loss'], label='train_loss')
    plt.xlabel('Step'); plt.ylabel('Loss'); plt.title(f'Training Loss ({condition})')
    plt.grid(True); plt.legend()
    fname = os.path.join(IMAGES_DIR, f"training_loss_{condition}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


def plot_perplexity(log: Dict, condition: str):
    _ensure_images_dir()
    steps = [s for s,p in zip(log['steps'], log['ppl']) if not np.isnan(p)]
    ppls = [p for p in log['ppl'] if not np.isnan(p)]
    if len(steps) == 0:
        print("No PPL points to plot.")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(steps, ppls, marker='o', label='val_ppl')
    plt.xlabel('Step'); plt.ylabel('Perplexity'); plt.title(f'Validation PPL ({condition})')
    plt.grid(True); plt.legend()
    fname = os.path.join(IMAGES_DIR, f"perplexity_{condition}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


def plot_bar(values_dict: Dict[str, float], topic: str, condition: str):
    _ensure_images_dir()
    names = list(values_dict.keys())
    vals = [values_dict[k] for k in names]
    plt.figure(figsize=(7, 4))
    sns.barplot(x=names, y=vals, color='C0')
    plt.xticks(rotation=30, ha='right')
    plt.title(f'{topic.replace("_", " ").title()} ({condition})')
    plt.ylabel(topic.replace('_',' ').title())
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    fname = os.path.join(IMAGES_DIR, f"{topic}_{condition}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


def plot_line(xs: List[float], ys: List[float], xlabel: str, ylabel: str, topic: str, condition: str):
    _ensure_images_dir()
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs {xlabel} ({condition})')
    plt.grid(True)
    fname = os.path.join(IMAGES_DIR, f"{topic}_{condition}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


def plot_confusion(cm: np.ndarray, labels: List[str], condition: str):
    _ensure_images_dir()
    plt.figure(figsize=(4.8,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix ({condition})')
    fname = os.path.join(IMAGES_DIR, f"confusion_matrix_{condition}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")
