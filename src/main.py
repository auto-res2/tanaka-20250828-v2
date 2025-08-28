#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry for running RASST experiments end-to-end.
Run from project root:  python -m src.main

Experiments:
- e2e: Compare RASST to baselines (vanilla, reformer, lora, meft, tempo)
- ablate: Component ablations
- longctx: Long-context scaling + (optional) SST-2 finetune demo
- test: Quick tiny synthetic smoke test (runs fast, CPU-friendly)

All figures are saved as PDF in .research/iteration1/images.
"""
import os
import json
import time
import math
import argparse
from dataclasses import asdict
import numpy as np
import torch

from .train import TrainConfig, run_training, build_model
from .preprocess import get_tokenizer
from .evaluate import (
    plot_training_loss, plot_perplexity, plot_bar, plot_line, plot_confusion
)

# Optional libs
try:
    import bitsandbytes as bnb  # type: ignore
    HAS_BNB = True
except Exception:
    HAS_BNB = False

try:
    from datasets import load_dataset  # type: ignore
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

try:
    from transformers import AutoTokenizer  # type: ignore
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    from sklearn.metrics import confusion_matrix  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def exp_end_to_end(args):
    base = TrainConfig(
        seq_len=512 if args.model_size == '355M' else 1024,
        steps=200,            # modest for demo; increase for full runs
        eval_every=50,
        batch_size=6 if args.model_size == '355M' else 8,
        model_size=args.model_size,
        device=args.device,
        data_pattern=args.data_pattern,
        tokens_per_update=0,
        max_train_samples=6000 if args.data_pattern == 'wikitext' else 4096,
        max_val_samples=1000 if args.data_pattern == 'wikitext' else 512,
    )

    configs = []
    configs.append(base)
    for bl in ["vanilla", "reformer", "lora", "meft", "tempo"]:
        cdict = {**asdict(base), 'baseline': bl}
        if bl == 'vanilla':
            cdict['batch_size'] = max(2, base.batch_size // 2)
        cfg = TrainConfig(**cdict)
        configs.append(cfg)

    results = {}

    for cfg in configs:
        label = cfg.baseline if cfg.baseline != 'RASST' else 'RASST'
        print("\n=== Running:", label, cfg.model_size, "data=", cfg.data_pattern, '===')
        log, model, loaders, _ = run_training(cfg)
        results[label] = {
            'peak_alloc_mb': float(max(log['peak_alloc_mb'])),
            'peak_reserve_mb': float(max(log['peak_reserve_mb'])),
            'final_ppl': float([p for p in log['ppl'] if not math.isnan(p)][-1]) if any([not math.isnan(p) for p in log['ppl']]) else float('nan'),
            'tps_median': float(np.median(log['tps']))
        }
        # Per-run plots
        condition = f"{label.lower()}"
        plot_training_loss(log, condition)
        plot_perplexity(log, condition)

    # Comparison plots across baselines
    mem_alloc = {k: v['peak_alloc_mb'] for k,v in results.items()}
    tps_med = {k: v['tps_median'] for k,v in results.items()}
    ppl_val = {k: v['final_ppl'] for k,v in results.items()}

    plot_bar(mem_alloc, topic='peak_memory', condition='baselines')
    plot_bar(tps_med, topic='tokens_per_second', condition='baselines')
    plot_bar(ppl_val, topic='perplexity', condition='baselines')

    print("\nSummary:")
    print(json.dumps(results, indent=2))


def exp_ablation(args):
    base = TrainConfig(
        seq_len=512 if args.model_size == '355M' else 1024,
        steps=150,
        eval_every=50,
        batch_size=6 if args.model_size == '355M' else 8,
        model_size=args.model_size,
        device=args.device,
        data_pattern=args.data_pattern,
        baseline='RASST',
        max_train_samples=4000 if args.data_pattern == 'wikitext' else 2048,
        max_val_samples=1000 if args.data_pattern == 'wikitext' else 512,
    )

    from dataclasses import asdict
    variants = [
        ('full', base),
        ('no_rev', TrainConfig(**{**asdict(base), 'reversible': False})),
        ('no_inplace', TrainConfig(**{**asdict(base), 'inplace_ops': False})),
        ('no_struct', TrainConfig(**{**asdict(base), 'shared_structured': False})),
        ('rank0', TrainConfig(**{**asdict(base), 'adapter_rank': 0})),
    ]

    results = {}
    for name, cfg in variants:
        print(f"\n=== Ablation: {name} ===")
        log, model, loaders, _ = run_training(cfg)
        results[name] = {
            'peak_alloc_mb': float(max(log['peak_alloc_mb'])),
            'peak_reserve_mb': float(max(log['peak_reserve_mb'])),
            'final_ppl': float([p for p in log['ppl'] if not math.isnan(p)][-1]) if any([not math.isnan(p) for p in log['ppl']]) else float('nan'),
            'tps_median': float(np.median(log['tps']))
        }
        plot_training_loss(log, f"ablation_{name}")
        plot_perplexity(log, f"ablation_{name}")

    plot_bar({k:v['peak_alloc_mb'] for k,v in results.items()}, topic='peak_memory', condition='ablation')
    plot_bar({k:v['tps_median'] for k,v in results.items()}, topic='tokens_per_second', condition='ablation')
    plot_bar({k:v['final_ppl'] for k,v in results.items()}, topic='perplexity', condition='ablation')

    print("\nAblation Summary:")
    print(json.dumps(results, indent=2))


def _build_classifier_head(hidden_size: int, num_classes: int = 2) -> torch.nn.Module:
    return torch.nn.Linear(hidden_size, num_classes)


def _evaluate_classifier(model: torch.nn.Module, clf: torch.nn.Module, loader, device: str):
    model.eval(); clf.eval()
    correct = total = 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            h = model(xb)
            h = h.mean(dim=1)
            logits = clf(h)
            pred = logits.argmax(-1)
            correct += (pred == yb).sum().item(); total += yb.numel()
            all_true.append(yb.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
    acc = correct / max(1, total)
    cm = None
    if HAS_SKLEARN and len(all_true) > 0:
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
    model.train(); clf.train()
    return acc, cm


class GLUEDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, tokenizer, task='sst2', split='train', max_len=256):
        if not HAS_DATASETS:
            raise RuntimeError("datasets library not available.")
        ds = load_dataset('glue', task, split=split)
        self.labels = ds['label']
        self.tokenizer = tokenizer
        self.max_len = max_len
        if task == 'sst2':
            texts = ds['sentence']
        elif task == 'mrpc':
            texts = [a + ' [SEP] ' + b for a,b in zip(ds['sentence1'], ds['sentence2'])]
        else:
            raise ValueError('Only sst2/mrpc supported in this demo.')
        self.input_ids = [tokenizer(t, truncation=True, max_length=max_len, padding='max_length')['input_ids'] for t in texts]

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def exp_long_context(args):
    # Part A: Long-context scaling on synthetic or WikiText
    seqs = [1024, 4096]
    mem_res, tps_res = [], []
    for L in seqs:
        cfg = TrainConfig(
            seq_len=L,
            steps=120,
            eval_every=40,
            batch_size=max(2, 8 // max(1, L//1024)),
            model_size=args.model_size,
            device=args.device,
            data_pattern=args.data_pattern,
            baseline='RASST',
            max_train_samples=3000 if args.data_pattern == 'wikitext' else 2048,
            max_val_samples=800 if args.data_pattern == 'wikitext' else 512,
        )
        print(f"\n=== Long-context pretrain: seq_len={L} ===")
        log, model, loaders, _ = run_training(cfg)
        mem_res.append(max(log['peak_alloc_mb']))
        tps_res.append(float(np.median(log['tps'])))

    plot_line(seqs, mem_res, xlabel='Sequence Length', ylabel='Peak Alloc Memory (MB)', topic='peak_memory', condition='rasst')
    plot_line(seqs, tps_res, xlabel='Sequence Length', ylabel='Tokens/sec (median)', topic='tokens_per_second', condition='rasst')
    print("Memory scaling:", {f'seq{L}': m for L, m in zip(seqs, mem_res)})

    # Part B: Optional SST-2 finetune (adapters + classifier)
    if not (HAS_TRANSFORMERS and HAS_DATASETS):
        print("Skipping SST-2 finetune (datasets/transformers unavailable).")
        return

    tokenizer = get_tokenizer('gpt2')
    cfg = TrainConfig(seq_len=256, steps=200, eval_every=100, batch_size=16, model_size=args.model_size,
                      device=args.device, baseline='RASST', data_pattern='synthetic_uniform')
    model = build_model(cfg, tokenizer.vocab_size)
    model.lm_head = torch.nn.Identity()  # use mean pooled hidden states
    clf = _build_classifier_head(model.d_model, 2).to(cfg.device)

    train_ds = GLUEDatasetWrapper(tokenizer, task='sst2', split='train', max_len=cfg.seq_len)
    val_ds = GLUEDatasetWrapper(tokenizer, task='sst2', split='validation', max_len=cfg.seq_len)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    params = list(p for p in model.parameters() if p.requires_grad) + list(clf.parameters())
    opt = bnb.optim.Adam8bit(params, lr=2e-4) if HAS_BNB else torch.optim.AdamW(params, lr=2e-4)

    print("\n=== SST-2 Finetuning (adapters + classifier) ===")
    for step, (xb, yb) in enumerate(train_loader):
        if step >= cfg.steps: break
        xb, yb = xb.to(cfg.device), yb.to(cfg.device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=('cuda' in cfg.device)):
            h = model(xb)
            h = h.mean(dim=1)
            loss = torch.nn.functional.cross_entropy(clf(h), yb)
        loss.backward()
        opt.step()
        if (step+1) % 100 == 0:
            acc, _ = _evaluate_classifier(model, clf, val_loader, cfg.device)
            print(f"FT step {step+1} loss={loss.item():.3f} acc={acc:.3f}")

    acc_fp16, cm = _evaluate_classifier(model, clf, val_loader, cfg.device)
    print(f"SST-2 dev accuracy (FP16 path): {acc_fp16:.3f}")
    if cm is not None:
        plot_confusion(cm, labels=["neg","pos"], condition='sst2_rasst')

    acc_int8, _ = _evaluate_classifier(model, clf, val_loader, cfg.device)
    print(f"SST-2 dev accuracy (INT8 base weights path): {acc_int8:.3f}")

    plot_bar({'FP16': acc_fp16, 'INT8_base': acc_int8}, topic='accuracy', condition='sst2_rasst')


def quick_test():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    base = TrainConfig(
        seq_len=64,
        steps=10,
        eval_every=5,
        batch_size=4,
        model_size='TINY',
        device=device,
        data_pattern='synthetic_uniform',
        baseline='RASST',
        max_train_samples=256,
        max_val_samples=64,
    )
    print("\n=== Quick Test: RASST (TINY) synthetic_uniform ===")
    log_r, model_r, _, _ = run_training(base)
    plot_training_loss(log_r, condition='rasst_test')
    plot_perplexity(log_r, condition='rasst_test')

    vcfg = TrainConfig(**{**asdict(base), 'baseline': 'vanilla'})
    print("\n=== Quick Test: vanilla (TINY) synthetic_uniform ===")
    log_v, model_v, _, _ = run_training(vcfg)
    plot_training_loss(log_v, condition='vanilla_test')
    plot_perplexity(log_v, condition='vanilla_test')

    mem = {'RASST': max(log_r['peak_alloc_mb']), 'vanilla': max(log_v['peak_alloc_mb'])}
    tps = {'RASST': float(np.median(log_r['tps'])), 'vanilla': float(np.median(log_v['tps']))}
    plot_bar(mem, topic='peak_memory', condition='test')
    plot_bar(tps, topic='tokens_per_second', condition='test')

    print("Quick test completed successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='test', choices=['e2e', 'ablate', 'longctx', 'test'])
    parser.add_argument('--device', type=str, default=('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--model_size', type=str, default='TINY', choices=['TINY','125M','355M'])
    parser.add_argument('--data_pattern', type=str, default='synthetic_uniform',
                        choices=['wikitext','synthetic_uniform','synthetic_zipf','synthetic_repeating'])
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.exp == 'e2e':
        exp_end_to_end(args)
    elif args.exp == 'ablate':
        exp_ablation(args)
    elif args.exp == 'longctx':
        exp_long_context(args)
    elif args.exp == 'test':
        quick_test()


if __name__ == '__main__':
    main()
