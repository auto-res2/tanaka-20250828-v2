#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training and model definitions for RASST.
- Implements RASST Transformer with reversible blocks, shared INT8 base weights, and low-rank adapters with ReLoRA.
- Provides run_training() used by src.main.

Notes:
- All plots are handled in src.evaluate to avoid circular imports.
- Data loading is handled in src.preprocess.
"""
import os
import math
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional: bitsandbytes for 8-bit Adam
try:
    import bitsandbytes as bnb  # type: ignore
    HAS_BNB = True
except Exception:
    HAS_BNB = False

# Local relative imports
from .preprocess import make_datasets
from .evaluate import estimate_ppl, peak_memory_report

SEED = 42

def set_seed(seed: int = SEED):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------
# In-place friendly layers
# ------------------------------
class InplaceLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class ReGELU2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU-gated GELU2-style approximation
        return F.relu(x) * torch.tanh(0.79788456 * (x + 0.044715 * (x ** 3)))


# ------------------------------
# Shared INT8 base weights (SURM proxy) + Adapters
# ------------------------------
class SharedInt8Base:
    """Row-wise quantized shared base matrix used across modules/layers.
    This is a practical proxy for the proposed structured unrestricted-rank matrix (SURM).
    """
    def __init__(self, in_dim: int, out_dim: int, name: str, device: str):
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.qweight: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None  # [out_dim, 1]

    def init_(self, init_std: float = 0.02, structured: bool = True):
        with torch.no_grad():
            w = torch.empty(self.out_dim, self.in_dim).normal_(0, init_std)
            if structured:
                # Simple structure proxy: local smoothing (Toeplitz-ish)
                w = (w + torch.roll(w, shifts=1, dims=0) + torch.roll(w, shifts=1, dims=1)) / 3.0
            maxv = w.abs().amax(dim=1, keepdim=True) + 1e-8
            self.scale = (maxv / 127.0).to(torch.float32)
            self.qweight = (w / self.scale).round().clamp_(-127, 127).to(torch.int8)
            self.qweight = self.qweight.to(self.device)
            self.scale = self.scale.to(self.device)

    def to(self, device: str):
        self.device = device
        if self.qweight is not None:
            self.qweight = self.qweight.to(device)
        if self.scale is not None:
            self.scale = self.scale.to(device)
        return self

    def dequant(self) -> torch.Tensor:
        return self.qweight.to(torch.float32) * self.scale

    def merge_dense_(self, delta: torch.Tensor):
        # Merge dense delta (from adapters) and requantize row-wise (ReLoRA)
        with torch.no_grad():
            w = self.dequant() + delta.to(self.qweight.device, dtype=torch.float32)
            maxv = w.abs().amax(dim=1, keepdim=True) + 1e-8
            self.scale = (maxv / 127.0).to(torch.float32)
            self.qweight = (w / self.scale).round().clamp_(-127, 127).to(torch.int8)


class LowRankAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float = 1.0, init_scale: float = 1e-3):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        if rank > 0:
            self.A = nn.Parameter(init_scale * torch.randn(out_dim, rank))
            self.B = nn.Parameter(init_scale * torch.randn(in_dim, rank))
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)

    def weight(self) -> Optional[torch.Tensor]:
        if self.rank <= 0:
            return None
        return (self.A @ self.B.t()) * self.alpha

    def reset(self, init_scale: float = 1e-3):
        if self.rank > 0:
            with torch.no_grad():
                self.A.copy_(init_scale * torch.randn_like(self.A))
                self.B.copy_(init_scale * torch.randn_like(self.B))


class AdapterLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, shared_base: SharedInt8Base, rank: int = 32, bias: bool = True):
        super().__init__()
        self.shared_base = shared_base
        self.adapter = LowRankAdapter(in_dim, out_dim, rank)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.shared_base.dequant()  # [out, in] fp32
        y = x @ base.t()
        if self.adapter.rank > 0:
            y = y + x @ self.adapter.weight().t()
        if self.bias is not None:
            y = y + self.bias
        return y

    def merge_and_reset(self):
        if self.adapter.rank > 0:
            self.shared_base.merge_dense_(self.adapter.weight().detach())
            self.adapter.reset()


# ------------------------------
# Attention and FFN
# ------------------------------
class RASSTSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, shared_q: SharedInt8Base, shared_k: SharedInt8Base,
                 shared_v: SharedInt8Base, shared_o: SharedInt8Base, rank: int = 32, attn_p: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = AdapterLinear(d_model, d_model, shared_q, rank)
        self.k = AdapterLinear(d_model, d_model, shared_k, rank)
        self.v = AdapterLinear(d_model, d_model, shared_v, rank)
        self.o = AdapterLinear(d_model, d_model, shared_o, rank)
        self.drop = nn.Dropout(attn_p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, Dh]
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o(y)
        return y

    def merge_and_reset(self):
        for mod in [self.q, self.k, self.v, self.o]:
            mod.merge_and_reset()


class RASSTFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, shared_in: SharedInt8Base, shared_out: SharedInt8Base,
                 rank: int = 32, pdrop: float = 0.0):
        super().__init__()
        self.fc1 = AdapterLinear(d_model, d_ff, shared_in, rank)
        self.act = ReGELU2()
        self.fc2 = AdapterLinear(d_ff, d_model, shared_out, rank)
        self.drop = nn.Dropout(pdrop)

    def forward(self, x: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        B, T, C = x.shape
        if T <= chunk_size:
            return self.fc2(self.drop(self.act(self.fc1(x))))
        outs = []
        for s in range(0, T, chunk_size):
            xe = x[:, s:s+chunk_size]
            outs.append(self.fc2(self.drop(self.act(self.fc1(xe)))))
        return torch.cat(outs, dim=1)

    def merge_and_reset(self):
        self.fc1.merge_and_reset(); self.fc2.merge_and_reset()


# ------------------------------
# Reversible block
# ------------------------------
class RevBlock(nn.Module):
    def __init__(self, F: nn.Module, G: nn.Module, ln1: nn.Module, ln2: nn.Module):
        super().__init__()
        self.F = F
        self.G = G
        self.ln1 = ln1
        self.ln2 = ln2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise additive coupling. Our F and G expect full d_model inputs.
        # We "lift" half-vectors to full dimension by zero-padding, apply module, then
        # project back by slicing the first half. This preserves the reversible structure
        # while keeping modules unchanged.
        B, T, C = x.shape
        C2 = C // 2
        x1, x2 = x.split(C2, dim=-1)

        # Lift x2 to full dim and apply F
        x2_full = torch.cat([x2, torch.zeros_like(x1)], dim=-1)
        f_out_half = self.F(self.ln1(x2_full))[..., :C2]
        y1 = x1 + f_out_half

        # Lift y1 to full dim and apply G
        y1_full = torch.cat([y1, torch.zeros_like(x2)], dim=-1)
        g_out_half = self.G(self.ln2(y1_full))[..., :C2]
        y2 = x2 + g_out_half

        return torch.cat([y1, y2], dim=-1)


# ------------------------------
# RASST Transformer
# ------------------------------
class RASSTTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, n_layers: int = 12, n_heads: int = 12, d_ff: int = 3072,
                 reversible: bool = True, inplace_ops: bool = True, shared_structured: bool = True,
                 adapter_rank: int = 32, relora_steps: int = 500, device: str = 'cuda:0'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.relora_steps = relora_steps
        self.reversible = reversible
        self.adapter_rank = adapter_rank
        self.device_name = device

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 8192, d_model))  # supports long context

        def make_shared(name, out_d, in_d):
            s = SharedInt8Base(in_d, out_d, name, device)
            s.init_(structured=shared_structured)
            s.to(device)
            return s

        self.shared = {
            'q': make_shared('q', d_model, d_model),
            'k': make_shared('k', d_model, d_model),
            'v': make_shared('v', d_model, d_model),
            'o': make_shared('o', d_model, d_model),
            'f1': make_shared('f1', d_ff, d_model),
            'f2': make_shared('f2', d_model, d_ff),
        }

        blocks = []
        for _ in range(n_layers):
            ln1 = InplaceLayerNorm(d_model) if inplace_ops else nn.LayerNorm(d_model)
            ln2 = InplaceLayerNorm(d_model) if inplace_ops else nn.LayerNorm(d_model)
            attn = RASSTSelfAttention(d_model, n_heads, self.shared['q'], self.shared['k'], self.shared['v'], self.shared['o'], rank=adapter_rank)
            ffn = RASSTFFN(d_model, d_ff, self.shared['f1'], self.shared['f2'], rank=adapter_rank)
            if reversible:
                blocks.append(RevBlock(attn, ffn, ln1, ln2))
            else:
                blocks.append(nn.ModuleList([ln1, attn, ln2, ffn]))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = InplaceLayerNorm(d_model) if inplace_ops else nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie weights

        self.to(device)
        self._since_merge = 0

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        if self.reversible:
            assert self.d_model % 2 == 0, "d_model must be even for reversible blocks"
            for blk in self.blocks:
                x = blk(x)
        else:
            # Non-reversible baseline with residual stacking (+ optional checkpointing by caller if desired)
            for blk in self.blocks:
                ln1, attn, ln2, ffn = blk
                x = x + attn(ln1(x))
                x = x + ffn(ln2(x))
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def merge_adapters_if_needed(self):
        if self.adapter_rank <= 0 or self.relora_steps <= 0:
            return
        self._since_merge += 1
        if self._since_merge >= self.relora_steps:
            for blk in self.blocks:
                if isinstance(blk, RevBlock):
                    blk.F.merge_and_reset(); blk.G.merge_and_reset()
                else:
                    _, attn, _, ffn = blk
                    attn.merge_and_reset(); ffn.merge_and_reset()
            self._since_merge = 0


# ------------------------------
# Config, builders, training loop
# ------------------------------
@dataclass
class TrainConfig:
    vocab: str = "gpt2"
    seq_len: int = 1024
    batch_size: int = 8
    steps: int = 200
    lr: float = 3e-4
    warmup: int = 100
    eval_every: int = 50
    eval_tokens: int = 200000
    model_size: str = "125M"  # "125M", "355M", "TINY"
    reversible: bool = True
    inplace_ops: bool = True
    shared_structured: bool = True
    adapter_rank: int = 32
    relora_steps: int = 200
    baseline: str = "RASST"  # "RASST", "vanilla", "reformer", "lora", "meft", "tempo"
    device: str = "cuda:0"
    tokens_per_update: int = 0  # if >0 overrides accum steps calculation
    accum_steps: int = 1
    data_pattern: str = "wikitext"  # or synthetic_*
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = 2000


def compute_accum_steps(cfg: TrainConfig) -> int:
    tpu = cfg.tokens_per_update if cfg.tokens_per_update > 0 else cfg.batch_size * cfg.seq_len
    micro = cfg.batch_size * cfg.seq_len
    return max(1, tpu // max(1, micro))


def build_model(cfg: TrainConfig, vocab_size: int) -> RASSTTransformer:
    if cfg.model_size == "125M":
        d_model, n_layers, n_heads, d_ff = 768, 12, 12, 3072
    elif cfg.model_size == "355M":
        d_model, n_layers, n_heads, d_ff = 1024, 24, 16, 4096
    elif cfg.model_size == "TINY":
        d_model, n_layers, n_heads, d_ff = 128, 2, 4, 512
    else:
        raise ValueError("Unknown model size")

    reversible = cfg.reversible
    inplace_ops = cfg.inplace_ops
    shared_structured = cfg.shared_structured
    adapter_rank = cfg.adapter_rank
    relora_steps = cfg.relora_steps

    # Configure baselines
    if cfg.baseline == "vanilla":
        reversible = False; inplace_ops = False; shared_structured = False; adapter_rank = 0
    elif cfg.baseline == "reformer":
        reversible = True; inplace_ops = False; shared_structured = False; adapter_rank = 0
    elif cfg.baseline == "lora":
        reversible = False; inplace_ops = False; shared_structured = False; adapter_rank = 128; relora_steps = 0
    elif cfg.baseline == "meft":
        reversible = True; inplace_ops = False; shared_structured = False; adapter_rank = 32; relora_steps = 0
    elif cfg.baseline == "tempo":
        reversible = False; inplace_ops = True; shared_structured = False; adapter_rank = 0

    model = RASSTTransformer(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                             n_heads=n_heads, d_ff=d_ff, reversible=reversible,
                             inplace_ops=inplace_ops, shared_structured=shared_structured,
                             adapter_rank=adapter_rank, relora_steps=relora_steps,
                             device=cfg.device)
    return model


def param_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.is_floating_point():
            b = torch.finfo(p.dtype).bits // 8
        else:
            b = torch.iinfo(p.dtype).bits // 8
        total += p.numel() * b
    return total


@torch.no_grad()
def _safe_median(xs: List[float]) -> float:
    if len(xs) == 0:
        return float('nan')
    import numpy as np
    return float(np.median(xs))


def run_training(cfg: TrainConfig):
    set_seed()

    tokenizer = None
    vocab_size = 50257

    # Build datasets and loaders
    train_ds, val_ds, tokenizer, vocab_size = make_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

    # Build model
    model = build_model(cfg, vocab_size=vocab_size)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if HAS_BNB:
        opt = bnb.optim.Adam8bit(params, lr=cfg.lr)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)

    warmup = max(1, cfg.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s+1)/warmup))

    scaler = torch.amp.GradScaler('cuda', enabled=('cuda' in cfg.device))

    cfg.accum_steps = compute_accum_steps(cfg)
    tokens_per_micro = cfg.batch_size * cfg.seq_len

    log = {
        'cfg': asdict(cfg), 'steps': [], 'loss': [], 'ppl': [],
        'tps': [], 'peak_alloc_mb': [], 'peak_reserve_mb': [], 'lr': []
    }

    if torch.cuda.is_available() and 'cuda' in cfg.device:
        torch.cuda.reset_peak_memory_stats()

    model.train()
    itr = iter(train_loader)

    start_time = time.time()
    for step in range(cfg.steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for acc in range(cfg.accum_steps):
            try:
                xb, yb = next(itr)
            except StopIteration:
                itr = iter(train_loader)
                xb, yb = next(itr)
            xb, yb = xb.to(cfg.device, non_blocking=True), yb.to(cfg.device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=('cuda' in cfg.device)):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1)) / cfg.accum_steps
            scaler.scale(loss).backward()
            total_loss += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        model.merge_adapters_if_needed()

        dt = time.time() - t0
        tokens_this_step = cfg.accum_steps * tokens_per_micro
        tps = tokens_this_step / max(1e-6, dt)

        ppl = float('nan')
        if (step + 1) % cfg.eval_every == 0 or step == 0:
            ppl = estimate_ppl(model, val_loader, cfg.device)

        mem = peak_memory_report()
        log['steps'].append(step+1)
        log['loss'].append(total_loss)
        log['ppl'].append(ppl)
        log['tps'].append(tps)
        log['peak_alloc_mb'].append(mem['alloc_mb'])
        log['peak_reserve_mb'].append(mem['reserve_mb'])
        log['lr'].append(sched.get_last_lr()[0])

        if (step+1) % max(1, cfg.eval_every//2) == 0 or step < 5:
            print(f"step {step+1}/{cfg.steps} | loss {total_loss:.3f} | ppl {ppl:.2f} | tps {tps:.1f} | allocMB {mem['alloc_mb']:.0f} | reserveMB {mem['reserve_mb']:.0f}")

    total_time = time.time() - start_time
    print(f"Done. Total time: {total_time/60:.2f} min. Peak alloc MB: {max(log['peak_alloc_mb']):.0f} | Peak reserve MB: {max(log['peak_reserve_mb']):.0f}")

    return log, model, (train_loader, val_loader), tokenizer
