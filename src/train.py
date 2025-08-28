import os
import math
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import set_seed, device_info, peak_mem_gb, reset_peak_mem


# ---------------------
# Toy Transformer blocks
# ---------------------

class ToyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        h = self.ln2(x)
        x = x + self.dropout(self.ff(h))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4, d_ff: int = 256, max_len: int = 256, task: str = 'cls', n_classes: int = 3):
        super().__init__()
        self.task = task  # 'cls' or 'lm'
        self.n_layers = n_layers
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([ToyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        if task == 'cls':
            self.head = nn.Linear(d_model, n_classes)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embed(input_ids) + self.pos_embed(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        out: Dict[str, torch.Tensor] = {}
        if self.task == 'cls':
            pooled = x.mean(dim=1)
            logits = self.head(pooled)
            out['logits'] = logits
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
                out['loss'] = loss
        else:
            logits = self.lm_head(x)
            out['logits'] = logits
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                out['loss'] = loss
        return out


# ---------------------
# LoRA (lightweight)
# ---------------------

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.weight = linear.weight
        self.bias = linear.bias
        for p in [self.weight, self.bias]:
            if p is not None:
                p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(self.out_features, r))
        self.B = nn.Parameter(torch.zeros(r, self.in_features))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        nn.init.zeros_(self.A)
        self.scaling = alpha / r

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = self.dropout(x) @ self.B.t() @ self.A.t() * self.scaling
        return base + lora


def apply_lora_to_toy(model: ToyTransformer, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
    for blk in model.blocks:
        lin1 = blk.ff[0]
        lin2 = blk.ff[3]
        blk.ff[0] = LoRALinear(lin1, r=r, alpha=alpha, dropout=dropout)
        blk.ff[3] = LoRALinear(lin2, r=r, alpha=alpha, dropout=dropout)
    if model.task == 'cls':
        model.head = LoRALinear(model.head, r=r, alpha=alpha, dropout=dropout)
    else:
        model.lm_head = LoRALinear(model.lm_head, r=r, alpha=alpha, dropout=dropout)
    return model


# ---------------------
# UCT minimal components
# ---------------------

@dataclass
class LayerPolicy:
    layer_id: int
    variant: str  # 'reversible' | 'projection' | 'store'
    bits: int = 3


class FewBitGrad:
    def __init__(self, bits: int = 3):
        assert 2 <= bits <= 8, "bits must be 2..8 for this demo"
        self.bits = bits
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: nn.Module):
        levels = 2 ** self.bits - 1
        scale_eps = 1e-8

        def hook_fn(grad):
            if grad is None:
                return None
            with torch.no_grad():
                gmax = grad.abs().amax()
                s = gmax / (levels / 2) + scale_eps
                q = torch.clamp((grad / s).round(), min=-(levels // 2), max=(levels // 2))
                deq = q * s
                return deq

        for p in model.parameters():
            if p.requires_grad:
                self._handles.append(p.register_hook(hook_fn))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class AnyPrecisionAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, ap_bits: int = 8):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        assert ap_bits in (2, 4, 8)
        self.ap_bits = ap_bits
        self._levels = 2 ** ap_bits - 1
        self._loss_window: List[float] = []
        self._divergence_hits = 0

    def state_init(self, p):
        st = self.state[p]
        if 'step' not in st:
            st['step'] = 0
            st['m_q'] = torch.zeros_like(p.data, dtype=torch.int8, device=p.device)
            st['v_q'] = torch.zeros_like(p.data, dtype=torch.int8, device=p.device)
            st['m_s'] = torch.ones(1, device=p.device) * 1e-3
            st['v_s'] = torch.ones(1, device=p.device) * 1e-3

    def _quantize(self, t: torch.Tensor, levels: int):
        gmax = t.abs().amax()
        s = gmax / (levels / 2 + 1e-8) + 1e-8
        q = torch.clamp((t / s).round(), min=-(levels // 2), max=(levels // 2)).to(torch.int8)
        return q, s

    def _dequantize(self, q: torch.Tensor, s: torch.Tensor):
        return q.float() * s

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        levels = self._levels
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state_init(p)
                st = self.state[p]
                st['step'] += 1
                grad = p.grad.data
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)
                m = self._dequantize(st['m_q'], st['m_s'])
                v = self._dequantize(st['v_q'], st['v_s'])
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                st['m_q'], st['m_s'] = self._quantize(m, levels)
                st['v_q'], st['v_s'] = self._quantize(v, levels)
                m_hat = m / (1 - beta1 ** st['step'])
                v_hat = v / (1 - beta2 ** st['step'])
                p.data.addcdiv_(m_hat, (v_hat.sqrt() + eps), value=-lr)
        return loss

    def update_loss(self, current_loss: float):
        self._loss_window.append(float(current_loss))
        if len(self._loss_window) > 10:
            self._loss_window.pop(0)
        if len(self._loss_window) >= 5:
            diffs = [self._loss_window[i + 1] - self._loss_window[i] for i in range(len(self._loss_window) - 1)]
            if all(d > 0 for d in diffs[-3:]):
                if self.ap_bits < 8:
                    self.ap_bits = 8
                    self._levels = 255
                    self._divergence_hits += 1
                    print(f"[AnyPrecisionAdamW] Upscaled optimizer state precision to 8 bits due to divergence.")


class UCTBlock(nn.Module):
    def __init__(self, base_block: ToyBlock, variant: str = 'store'):
        super().__init__()
        self.base = base_block
        assert variant in ('store', 'reversible', 'projection')
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == 'store':
            return self.base(x)
        elif self.variant == 'reversible':
            def fwd(inp):
                return self.base(inp)
            return torch.utils.checkpoint.checkpoint(fwd, x)
        else:
            y = self.base(x)
            proj = y.mean(dim=(1, 2))  # stored for monitoring, not used
            self._last_proj = proj.detach()
            return y


class UCTWrapper(nn.Module):
    def __init__(self, model: ToyTransformer, budget_bytes: int, rank: int = 1, bits: int = 3, planner: str = 'greedy', opt_quant: bool = True, controller: bool = True, profile: Optional[Dict] = None):
        super().__init__()
        assert isinstance(model, ToyTransformer), "UCTWrapper supports ToyTransformer in this demo"
        self.model = model
        self.rank = rank
        self.bits = bits
        self.budget_bytes = budget_bytes
        self.controller_on = controller
        self.opt_quant = opt_quant
        self.profile = profile or {"batch_size": 4, "seq_len": 128}
        self.policy: List[LayerPolicy] = []
        self.controller_events: List[Dict] = []
        self._fewbit = FewBitGrad(bits=self.bits)
        self._attach_fewbit()
        self._plan_layers(planner=planner)

    def _attach_fewbit(self):
        self._fewbit.attach(self.model)

    def _detach_fewbit(self):
        self._fewbit.detach()

    def _estimate_layer_activation_bytes(self, variant: str) -> int:
        B = int(self.profile.get("batch_size", 4))
        T = int(self.profile.get("seq_len", 128))
        D = int(self.model.d_model)
        bytes_fp16 = 2
        if variant == 'store':
            return B * T * D * bytes_fp16
        elif variant == 'projection':
            return B * self.rank * bytes_fp16
        else:
            return B * D * bytes_fp16

    def _estimate_optimizer_bytes(self) -> int:
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.opt_quant:
            per_param_bytes = (self.bits / 8.0) * 2
        else:
            per_param_bytes = 4.0 * 2
        return int(total_params * per_param_bytes)

    def estimate_peak_memory_bytes(self) -> int:
        param_bytes = sum(p.numel() for p in self.model.parameters()) * 2
        act_bytes = 0
        for pol in self.policy:
            act_bytes += self._estimate_layer_activation_bytes(pol.variant)
        opt_bytes = self._estimate_optimizer_bytes()
        return int(param_bytes + act_bytes + opt_bytes + 5 * 1024 ** 2)

    def _plan_layers(self, planner: Optional[str] = 'greedy'):
        L = self.model.n_layers
        provisional: List[LayerPolicy] = []
        for i in range(L):
            if planner is None:
                provisional.append(LayerPolicy(i, 'store', self.bits))
            else:
                if i < L // 3:
                    provisional.append(LayerPolicy(i, 'projection', self.bits))
                elif i >= 2 * L // 3:
                    provisional.append(LayerPolicy(i, 'reversible', self.bits))
                else:
                    provisional.append(LayerPolicy(i, 'store', self.bits))
        self.policy = provisional
        planned = self.estimate_peak_memory_bytes()
        while planned > self.budget_bytes:
            costs = [(i, self._estimate_layer_activation_bytes(pol.variant)) for i, pol in enumerate(self.policy)]
            idx_sorted = sorted(range(len(costs)), key=lambda k: costs[k][1], reverse=True)
            flipped = False
            for k in idx_sorted:
                if self.policy[k].variant != 'reversible':
                    self.policy[k].variant = 'reversible'
                    flipped = True
                    break
            if not flipped:
                break
            planned = self.estimate_peak_memory_bytes()
        new_blocks = nn.ModuleList()
        for i, blk in enumerate(self.model.blocks):
            new_blocks.append(UCTBlock(blk, variant=self.policy[i].variant))
        self.model.blocks = new_blocks

    def get_policy(self) -> List[Dict]:
        return [dict(layer_id=p.layer_id, variant=p.variant, bits=p.bits) for p in self.policy]

    def set_uniform_policy(self, variant: str):
        assert variant in ('store', 'reversible', 'projection')
        for p in self.policy:
            p.variant = variant
        new_blocks = nn.ModuleList([UCTBlock(blk.base if isinstance(blk, UCTBlock) else blk, variant=variant)
                                    for blk in self.model.blocks])
        self.model.blocks = new_blocks

    def maybe_controller_switch(self, last_peak_bytes: int):
        if not self.controller_on:
            return
        if last_peak_bytes <= self.budget_bytes:
            return
        costs = [(i, self._estimate_layer_activation_bytes(pol.variant)) for i, pol in enumerate(self.policy)]
        idx_sorted = sorted(range(len(costs)), key=lambda k: costs[k][1], reverse=True)
        for k in idx_sorted:
            if self.policy[k].variant != 'reversible':
                old = self.policy[k].variant
                self.policy[k].variant = 'reversible'
                blk = self.model.blocks[k]
                base_blk = blk.base if isinstance(blk, UCTBlock) else blk
                self.model.blocks[k] = UCTBlock(base_blk, variant='reversible')
                self.controller_events.append({
                    'layer_id': k, 'old': old, 'new': 'reversible',
                    'time': time.time(), 'peak_bytes': last_peak_bytes
                })
                print(f"[UCT Controller] Switched layer {k} from {old} to reversible due to peak {last_peak_bytes/1e6:.2f} MB exceeding budget {self.budget_bytes/1e6:.2f} MB")
                break

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.model(input_ids=input_ids, labels=labels)


# Public API for UCT

def uct_wrap(model: ToyTransformer, budget: int, rank: int = 1, bits: int = 3, planner: Optional[str] = 'greedy', opt_quant: bool = True, controller: bool = True, profile: Optional[Dict] = None) -> UCTWrapper:
    return UCTWrapper(model, budget_bytes=budget, rank=rank, bits=bits, planner=planner, opt_quant=opt_quant, controller=controller, profile=profile)


def uct_get_policy(wrapper: UCTWrapper) -> List[Dict]:
    return wrapper.get_policy()


def uct_estimate_peak_memory(wrapper: UCTWrapper) -> int:
    return wrapper.estimate_peak_memory_bytes()


def uct_set_uniform_policy(wrapper: UCTWrapper, variant: str):
    wrapper.set_uniform_policy(variant)


def uct_get_controller_events(wrapper: UCTWrapper) -> List[Dict]:
    return wrapper.controller_events


# ---------------------
# Training utilities
# ---------------------

def train_one_epoch(model: nn.Module, opt: torch.optim.Optimizer, loader, device: torch.device, controller: Optional[UCTWrapper] = None) -> Tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_tokens = 0
    reset_peak_mem()
    t0 = time.time()
    for batch in loader:
        x = batch['input_ids'].to(device)
        y = batch['labels'].to(device)
        out = model(x, labels=y)
        loss = out['loss']
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * x.size(0)
        total_tokens += int(x.numel())
        if 'logits' in out and out['logits'].dim() == 2:
            preds = out['logits'].argmax(dim=-1)
            total_correct += int((preds == y).sum().item())
            total_count += int(x.size(0))
        if controller is not None:
            last_peak = peak_mem_gb() * (1024 ** 3)
            controller.maybe_controller_switch(int(last_peak))
    dt = time.time() - t0
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = (total_correct / total_count) if total_count > 0 else float('nan')
    tps = int(total_tokens / max(dt, 1e-9))
    return avg_loss, acc, tps
