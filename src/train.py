# -*- coding: utf-8 -*-
"""
Training utilities for HEAT (Hierarchical Early-exit Autoregressive Transformer)
- Defines HEATWrapper that adds grouped exits and confidence heads to a HF Causal LM
- Training loop for multi-exit distillation + confidence supervision
- Utility compute_ppl and basic reproducibility helpers

Notes:
- Use small models (e.g., sshleifer/tiny-gpt2) for quick tests.
- Figures are saved by evaluate.py; this module focuses on model and training.
"""
from typing import List, Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Reproducibility & Device
# -------------------------

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------
# HEAT Wrapper (Training)
# -------------------------

class HEATWrapper(nn.Module):
    """
    Wraps a HuggingFace Causal LM (GPT-2-like) with:
      - Grouped exits (heads H_g)
      - Confidence estimators C_g (sigmoid) predicting stability of next-token argmax if exiting at group g
      - Multi-exit training with distillation to the final head
    Assumes base_model has attributes .transformer.h and config.n_embd/vocab_size like GPT-2 family.
    """
    def __init__(self, base_model: nn.Module, group_size: int = 2, betas: Optional[List[float]] = None):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.group_size = int(group_size)
        # Assume GPT-2-like block listing
        self.layers = self.model.transformer.h
        self.num_layers = len(self.layers)
        self.groups = [list(range(i, min(i + self.group_size, self.num_layers))) for i in range(0, self.num_layers, self.group_size)]
        d_model = self.model.config.n_embd
        vocab_size = self.model.config.vocab_size
        self.exit_heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in self.groups])
        self.conf_heads = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid()) for _ in self.groups])
        self.betas = betas or [1.0 for _ in self.groups]

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def forward_with_exits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                           labels: Optional[torch.Tensor] = None, temps: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        # Full-depth forward to collect per-layer hidden states
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True)
        final_logits = out.logits  # [B, T, V]
        hiddens = out.hidden_states  # tuple length L+1; index l+1 is output after layer l

        group_logits, group_confs_seq, group_confs_last = [], [], []
        for g_idx, group in enumerate(self.groups):
            last_layer_idx = group[-1]
            h_g = hiddens[last_layer_idx + 1]  # [B, T, D]
            logits_g = self.exit_heads[g_idx](h_g)  # [B, T, V]
            # Sequence-wide confidence for positions predicting next token (up to T-1)
            conf_seq = self.conf_heads[g_idx](h_g[:, :-1, :].contiguous())  # [B, T-1, 1]
            # Confidence at the last position (for decoding)
            conf_last = self.conf_heads[g_idx](h_g[:, -1, :])  # [B, 1]
            group_logits.append(logits_g)
            group_confs_seq.append(conf_seq)
            group_confs_last.append(conf_last)

        loss = None
        aux: Dict[str, torch.Tensor] = {}
        if labels is not None:
            shift_logits = final_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Final head CE loss
            loss_final = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            losses: List[torch.Tensor] = [loss_final]
            # Distillation: KL from final head distribution
            with torch.no_grad():
                teacher_p = F.softmax(shift_logits, dim=-1)
                del teacher_p  # not used directly but clarifies intent
            for g_idx, logits_g in enumerate(group_logits):
                lg = logits_g[:, :-1, :].contiguous()
                T = 1.0 if temps is None else float(temps[g_idx])
                student_logp = F.log_softmax(lg / T, dim=-1)
                teacher_temp = F.softmax(shift_logits / T, dim=-1)
                loss_g = F.kl_div(student_logp, teacher_temp, reduction='batchmean') * (T * T)
                losses.append(self.betas[g_idx] * loss_g)
            # Confidence BCE: 1 if argmax early==argmax final
            bce = 0.0
            with torch.no_grad():
                arg_f = shift_logits.argmax(dim=-1)
            for g_idx, logits_g in enumerate(group_logits):
                lg = logits_g[:, :-1, :].contiguous()
                arg_g = lg.argmax(dim=-1)
                y = (arg_g == arg_f).float().unsqueeze(-1)  # [B, T-1, 1]
                p = group_confs_seq[g_idx]
                bce += F.binary_cross_entropy(p, y)
            loss = sum(losses) + bce / len(self.groups)
            aux = {
                'loss_final': loss_final.detach(),
                'loss_kl_mean': torch.stack([l if torch.is_tensor(l) else torch.tensor(l) for l in losses[1:]]).mean().detach(),
                'loss_bce': (bce / len(self.groups)).detach()
            }

        return {
            'final_logits': final_logits,
            'group_logits': group_logits,
            'group_confs_seq': group_confs_seq,
            'group_confs_last': group_confs_last,
            'loss': loss,
            **aux
        }


# -------------------------
# Perplexity utility
# -------------------------

def compute_ppl(wrapper: HEATWrapper, tokenized_ds: List[List[int]], max_length: int = 256, device: str = 'cpu') -> float:
    model = wrapper.model
    model.eval()
    losses = []
    with torch.no_grad():
        for ids in tokenized_ds:
            ids_t = torch.tensor(ids[:max_length], dtype=torch.long, device=device).unsqueeze(0)
            out = model(input_ids=ids_t, labels=ids_t)
            losses.append(out.loss.item())
    return float(math.exp(sum(losses) / max(1, len(losses))))


# -------------------------
# Training loop
# -------------------------

def train_heat_heads(wrapper: HEATWrapper, tokenizer, train_texts: List[str], val_texts: List[str],
                     epochs: int = 1, batch_size: int = 4, max_len: int = 128, lr: float = 5e-5,
                     device: str = 'cpu') -> Tuple[List[float], List[float]]:
    optim = torch.optim.AdamW(wrapper.parameters(), lr=lr)
    train_losses: List[float] = []
    val_ppls: List[float] = []

    def batch_iter(texts: List[str], bs: int):
        for i in range(0, len(texts), bs):
            yield texts[i:i+bs]

    for ep in range(epochs):
        wrapper.train()
        ep_losses: List[float] = []
        for chunk in batch_iter(train_texts, batch_size):
            enc = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
            inp = enc.input_ids.to(device)
            out = wrapper.forward_with_exits(inp, labels=inp)
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optim.step(); optim.zero_grad()
            ep_losses.append(float(loss.item()))
        train_loss = float(sum(ep_losses) / max(1, len(ep_losses)))
        train_losses.append(train_loss)
        # quick validation perplexity of base model
        val_tok = [tokenizer(t, return_tensors='pt', truncation=True, max_length=max_len).input_ids[0].tolist() for t in val_texts]
        ppl = compute_ppl(wrapper, val_tok, max_length=max_len, device=device)
        val_ppls.append(ppl)
        ce = out.get('loss_final', torch.tensor(0.)).item()
        kl = out.get('loss_kl_mean', torch.tensor(0.)).item()
        bce = out.get('loss_bce', torch.tensor(0.)).item()
        print(f"Epoch {ep+1}/{epochs} | train_loss={train_loss:.4f} | val_ppl={ppl:.2f} | CE={ce:.4f} | KL={kl:.4f} | BCE={bce:.4f}")
    return train_losses, val_ppls
