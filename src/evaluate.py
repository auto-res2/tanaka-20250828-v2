import os
import json
import shutil
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import confusion_matrix
    SK_AVAILABLE = True
except Exception:
    SK_AVAILABLE = False

from .preprocess import device_info
from .train import (
    ToyTransformer, apply_lora_to_toy,
    AnyPrecisionAdamW, uct_wrap, uct_get_policy, uct_estimate_peak_memory,
    uct_set_uniform_policy, train_one_epoch
)
from .preprocess import SyntheticSeqClassDataset, SyntheticLMDataset, reset_peak_mem, peak_mem_gb


IMAGES_DIR = os.path.join('.research', 'iteration1', 'images')
RESULTS_DIR = os.path.join('.research', 'iteration1')
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------
# Evaluation helpers
# ---------------------

def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            out = model(x, labels=y)
            loss = out['loss']
            total_loss += float(loss.item()) * x.size(0)
            if 'logits' in out and out['logits'].dim() == 2:
                preds = out['logits'].argmax(dim=-1)
                total_correct += int((preds == y).sum().item())
                total_count += int(x.size(0))
                ys.append(y.detach().cpu().numpy())
                ps.append(preds.detach().cpu().numpy())
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = (total_correct / total_count) if total_count > 0 else float('nan')
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    return avg_loss, acc, y_true, y_pred


# ---------------------
# Plotting (PDF only)
# ---------------------

def _savepdf(filename: str):
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'
    return os.path.join(IMAGES_DIR, filename)


def plot_loss_curve(history: Dict[str, List[float]], title: str, filename: str):
    plt.figure(figsize=(4, 3))
    for label, vals in history.items():
        plt.plot(vals, label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(_savepdf(filename), bbox_inches='tight', format='pdf')
    plt.close()


def plot_metric_bars(metrics: Dict[str, float], title: str, ylabel: str, filename: str):
    names = list(metrics.keys())
    vals = [metrics[k] for k in names]
    plt.figure(figsize=(4, 3))
    sns.barplot(x=names, y=vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(_savepdf(filename), bbox_inches='tight', format='pdf')
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str):
    if not SK_AVAILABLE or y_true.size == 0:
        print("[plot_confusion] Skipped (sklearn not available or empty data)")
        return
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(_savepdf(filename), bbox_inches='tight', format='pdf')
    plt.close()


def plot_scatter(x: List[float], y: List[float], labels: List[str], title: str, xlabel: str, ylabel: str, filename: str):
    plt.figure(figsize=(4, 3))
    for xi, yi, lab in zip(x, y, labels):
        plt.scatter([xi], [yi], label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(_savepdf(filename), bbox_inches='tight', format='pdf')
    plt.close()


# ---------------------
# Experiments (Toy)
# ---------------------

def run_experiment1_toy(seed: int = 0):
    print("\n===== Experiment 1 (Toy) — Budget-sweep vs Baselines =====")
    from torch.utils.data import DataLoader
    dev, devname = device_info()
    print(f"Device: {devname}")

    vocab_size = 128
    seq_len = 64
    n_classes = 3

    trainA = SyntheticSeqClassDataset(n=800, seq_len=seq_len, vocab_size=vocab_size, n_classes=n_classes, pattern='sum_mod_k', seed=seed)
    valA = SyntheticSeqClassDataset(n=200, seq_len=seq_len, vocab_size=vocab_size, n_classes=n_classes, pattern='sum_mod_k', seed=seed + 1)
    trainB = SyntheticSeqClassDataset(n=800, seq_len=seq_len, vocab_size=vocab_size, n_classes=n_classes, pattern='pattern_mix', seed=seed + 2)
    valB = SyntheticSeqClassDataset(n=200, seq_len=seq_len, vocab_size=vocab_size, n_classes=n_classes, pattern='pattern_mix', seed=seed + 3)

    def make_loader(ds, bs=32):
        from torch.utils.data import DataLoader
        return DataLoader(ds, batch_size=bs, shuffle=True)

    budgets_gb = [0.15, 0.20, 0.30]

    results = []
    for B in budgets_gb:
        print(f"\n-- Budget: {B:.2f} GB --")
        methods = []
        model_uct = ToyTransformer(vocab_size=vocab_size, d_model=128, n_layers=6, n_heads=4, d_ff=256, max_len=seq_len, task='cls', n_classes=n_classes).to(dev)
        wrapper = uct_wrap(model_uct, budget=int(B * (1024 ** 3)), rank=1, bits=3, planner='greedy', opt_quant=True, controller=True, profile={"batch_size": 32, "seq_len": seq_len})
        opt_uct = AnyPrecisionAdamW(wrapper.parameters(), lr=1e-3, ap_bits=8)
        methods.append(("UCT", wrapper, opt_uct))

        model_lora = ToyTransformer(vocab_size=vocab_size, d_model=128, n_layers=6, n_heads=4, d_ff=256, max_len=seq_len, task='cls', n_classes=n_classes).to(dev)
        model_lora = apply_lora_to_toy(model_lora, r=8, alpha=16, dropout=0.05)
        opt_lora = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_lora.parameters()), lr=1e-3)
        methods.append(("LoRA", model_lora, opt_lora))

        model_full = ToyTransformer(vocab_size=vocab_size, d_model=128, n_layers=6, n_heads=4, d_ff=256, max_len=seq_len, task='cls', n_classes=n_classes).to(dev)
        opt_full = torch.optim.AdamW(model_full.parameters(), lr=1e-3)
        methods.append(("FullFT", model_full, opt_full))

        for name, model, opt in methods:
            print(f"\nMethod: {name}")
            print("Policy (if UCT):", uct_get_policy(model) if hasattr(model, 'get_policy') else "N/A")
            hist_loss = {"trainA": [], "valA": [], "trainB": [], "valB": []}
            hist_acc = {"valA": [], "valB": []}
            loaders = [(make_loader(trainA), make_loader(valA), 'A'), (make_loader(trainB), make_loader(valB), 'B')]
            tokens_per_s_total: List[float] = []
            for ld_tr, ld_va, tag in loaders:
                for epoch in range(2):
                    train_loss, train_acc, tps = train_one_epoch(model, opt, ld_tr, dev, controller=model if hasattr(model, 'maybe_controller_switch') else None)
                    val_loss, val_acc, y_true, y_pred = evaluate(model, ld_va, dev)
                    tokens_per_s_total.append(tps)
                    hist_loss[f"train{tag}"].append(train_loss)
                    hist_loss[f"val{tag}"].append(val_loss)
                    hist_acc[f"val{tag}"].append(val_acc)
                    mem = peak_mem_gb()
                    print(f"[{name} | Pattern {tag} | Epoch {epoch + 1}] TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.3f}, PeakMem={mem:.3f} GB, Tokens/s={tps:.1f}")
                if y_true.size > 0:
                    plot_confusion(y_true, y_pred, title=f"{name} Confusion (Pattern {tag})", filename=f"confusion_matrix_{name.lower()}_pattern{tag}.pdf")
            plot_loss_curve({f"trainA": hist_loss['trainA'], f"valA": hist_loss['valA']}, title=f"Loss ({name}) Pattern A", filename=f"training_loss_{name.lower()}_patternA.pdf")
            plot_loss_curve({f"trainB": hist_loss['trainB'], f"valB": hist_loss['valB']}, title=f"Loss ({name}) Pattern B", filename=f"training_loss_{name.lower()}_patternB.pdf")
            res = {
                'method': name,
                'budget_gb': B,
                'mean_val_acc_A': float(np.mean(hist_acc['valA'])) if hist_acc['valA'] else float('nan'),
                'mean_val_acc_B': float(np.mean(hist_acc['valB'])) if hist_acc['valB'] else float('nan'),
                'peak_mem_gb': peak_mem_gb(),
                'tokens_per_s_avg': float(np.mean(tokens_per_s_total)) if tokens_per_s_total else float('nan')
            }
            results.append(res)
        xs = [r['peak_mem_gb'] for r in results if r['budget_gb'] == B]
        ys = [0.5 * (r['mean_val_acc_A'] + r['mean_val_acc_B']) for r in results if r['budget_gb'] == B]
        labs = [r['method'] for r in results if r['budget_gb'] == B]
        plot_scatter(xs, ys, labs, title=f"Memory vs Accuracy (B={B:.2f}GB)", xlabel="Peak Memory (GB)", ylabel="Avg Val Acc", filename=f"memory_vs_accuracy_B{int(B * 1000)}.pdf")

    tps_metrics: Dict[str, float] = {}
    for m in set([r['method'] for r in results]):
        vals = [r['tokens_per_s_avg'] for r in results if r['method'] == m]
        tps_metrics[m] = float(np.mean(vals)) if len(vals) else float('nan')
    plot_metric_bars(tps_metrics, title="Throughput (tokens/s)", ylabel="tokens/s", filename="throughput_tokens_per_s.pdf")

    out_json = os.path.join(RESULTS_DIR, "exp1_toy_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("Experiment 1 (Toy) complete. Results saved to:", out_json)


def run_experiment2_toy(seed: int = 0):
    print("\n===== Experiment 2 (Toy) — Planner Ablation & Policy Quality =====")
    from torch.utils.data import DataLoader
    dev, _ = device_info()

    vocab_size, seq_len, n_classes = 128, 64, 3
    train = SyntheticSeqClassDataset(n=800, seq_len=seq_len, vocab_size=vocab_size, n_classes=n_classes, pattern='sum_mod_k', seed=seed)
    val = SyntheticSeqClassDataset(n=200, seq_len=seq_len, vocab_size=vocab_size, n_classes=n_classes, pattern='sum_mod_k', seed=seed + 1)
    ld_tr = DataLoader(train, batch_size=32, shuffle=True)
    ld_va = DataLoader(val, batch_size=64)

    budget_gb = 0.20
    variants = [
        ("bap", dict(planner='greedy')),
        ("uniform_rev", dict(planner=None, uniform='reversible')),
        ("uniform_proj", dict(planner=None, uniform='projection')),
        ("uniform_store", dict(planner=None, uniform='store')),
    ]

    results = []
    for mode, cfg in variants:
        print(f"\nPolicy mode: {mode}")
        base = ToyTransformer(vocab_size=vocab_size, d_model=128, n_layers=6, n_heads=4, d_ff=256, max_len=seq_len, task='cls', n_classes=n_classes).to(dev)
        wrapper = uct_wrap(base, budget=int(budget_gb * (1024 ** 3)), rank=1, bits=3, planner=cfg.get('planner', 'greedy'), opt_quant=True, controller=False, profile={"batch_size": 32, "seq_len": seq_len})
        if cfg.get('planner') is None:
            uct_set_uniform_policy(wrapper, cfg['uniform'])
        policy = uct_get_policy(wrapper)
        planned_peak_gb = uct_estimate_peak_memory(wrapper) / (1024 ** 3)
        print("Planned policy:", policy)
        print(f"Planned peak memory (GB): {planned_peak_gb:.3f}")
        opt = AnyPrecisionAdamW(wrapper.parameters(), lr=1e-3, ap_bits=8)
        train_loss, train_acc, _ = train_one_epoch(wrapper, opt, ld_tr, dev, controller=None)
        reset_peak_mem()
        val_loss, val_acc, y_true, y_pred = evaluate(wrapper, ld_va, dev)
        actual_peak_gb = peak_mem_gb()
        mape = abs(planned_peak_gb - actual_peak_gb) / max(1e-6, actual_peak_gb)
        print(f"[Policy {mode}] TrainLoss={train_loss:.4f}, ValAcc={val_acc:.3f}, ActualPeak={actual_peak_gb:.3f} GB, MAPE={mape:.2%}")
        plot_confusion(y_true, y_pred, title=f"Policy {mode} Confusion", filename=f"confusion_matrix_{mode}.pdf")
        results.append({
            'mode': mode, 'planned_peak_gb': float(planned_peak_gb), 'actual_peak_gb': float(actual_peak_gb),
            'mape': float(mape), 'val_acc': float(val_acc)
        })

    for b in [2, 3, 4]:
        base = ToyTransformer(vocab_size=vocab_size, d_model=128, n_layers=6, n_heads=4, d_ff=256, max_len=seq_len, task='cls', n_classes=n_classes).to(dev)
        wrapper = uct_wrap(base, budget=int(budget_gb * (1024 ** 3)), rank=1, bits=b, planner='greedy', opt_quant=True, controller=False, profile={"batch_size": 32, "seq_len": seq_len})
        opt = AnyPrecisionAdamW(wrapper.parameters(), lr=1e-3, ap_bits=8)
        _ = train_one_epoch(wrapper, opt, ld_tr, dev)
        _, val_acc, _, _ = evaluate(wrapper, ld_va, dev)
        print(f"[Bits {b}] ValAcc={val_acc:.3f}")
        results.append({'mode': f'bits_{b}', 'val_acc': float(val_acc)})

    base = ToyTransformer(vocab_size=vocab_size, d_model=128, n_layers=6, n_heads=4, d_ff=256, max_len=seq_len, task='cls', n_classes=n_classes).to(dev)
    wrapper = uct_wrap(base, budget=int(budget_gb * (1024 ** 3)), rank=1, bits=3, planner='greedy', opt_quant=False, controller=False, profile={"batch_size": 32, "seq_len": seq_len})
    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)
    _ = train_one_epoch(wrapper, opt, ld_tr, dev)
    _, val_acc, _, _ = evaluate(wrapper, ld_va, dev)
    print(f"[OptQuant OFF] ValAcc={val_acc:.3f}")
    results.append({'mode': 'opt_quant_off', 'val_acc': float(val_acc)})

    mem_plan = {r['mode']: r['planned_peak_gb'] for r in results if 'planned_peak_gb' in r}
    mem_actual = {r['mode']: r['actual_peak_gb'] for r in results if 'actual_peak_gb' in r}
    if mem_plan:
        plot_metric_bars(mem_plan, title="Planned Peak Memory", ylabel="GB", filename="planned_peak_memory.pdf")
    if mem_actual:
        plot_metric_bars(mem_actual, title="Actual Peak Memory", ylabel="GB", filename="actual_peak_memory.pdf")

    out_json = os.path.join(RESULTS_DIR, "exp2_toy_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("Experiment 2 (Toy) complete. Results saved to:", out_json)


def run_experiment3_toy(seed: int = 0):
    print("\n===== Experiment 3 (Toy) — Runtime Controller Stress Test =====")
    from torch.utils.data import DataLoader
    dev, _ = device_info()

    vocab_size = 128
    n_classes = 3
    phases = [32, 64, 96, 128]
    budget_gb = 0.18

    results = []
    for controller_on in [True, False]:
        print(f"\nController ON={controller_on}")
        seq_len0 = phases[0]
        base = ToyTransformer(vocab_size=vocab_size, d_model=128, n_layers=6, n_heads=4, d_ff=256, max_len=phases[-1], task='cls', n_classes=n_classes).to(dev)
        wrapper = uct_wrap(base, budget=int(budget_gb * (1024 ** 3)), rank=1, bits=3, planner='greedy', opt_quant=True, controller=controller_on, profile={"batch_size": 32, "seq_len": seq_len0})
        opt = AnyPrecisionAdamW(wrapper.parameters(), lr=1e-3, ap_bits=8)
        total_tokens = 0
        total_oom = 0
        for L in phases:
            train = SyntheticSeqClassDataset(n=300, seq_len=L, vocab_size=vocab_size, n_classes=n_classes, pattern='sum_mod_k', seed=seed + L)
            ld_tr = DataLoader(train, batch_size=32, shuffle=True)
            try:
                tr_loss, tr_acc, tps = train_one_epoch(wrapper, opt, ld_tr, dev, controller=wrapper)
                total_tokens += int(L * len(train))
                print(f"[Phase L={L}] TrainLoss={tr_loss:.4f}, PeakMem={peak_mem_gb():.3f} GB")
            except torch.cuda.OutOfMemoryError:
                total_oom += 1
                print(f"OOM at sequence length {L}")
                if not controller_on:
                    break
        thpt = total_tokens
        results.append({'controller_on': controller_on, 'ooms': total_oom, 'tokens_processed': thpt, 'peak_gb': float(peak_mem_gb())})

    tps = {('ON' if r['controller_on'] else 'OFF'): r['tokens_processed'] for r in results}
    plot_metric_bars(tps, title="Throughput Controller ON vs OFF", ylabel="tokens processed", filename="controller_throughput.pdf")

    out_json = os.path.join(RESULTS_DIR, "exp3_toy_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("Experiment 3 (Toy) complete. Results saved to:", out_json)


# ---------------------
# Quick test (default)
# ---------------------

def quick_test():
    print("\n===== QUICK TEST START =====")
    from torch.utils.data import DataLoader
    dev, devname = device_info()
    print(f"Using device: {devname}")

    seed = 123
    vocab_size, seq_len, n_classes = 64, 32, 3
    train = SyntheticSeqClassDataset(n=200, seq_len=seq_len, vocab_size=vocab_size, n_classes=n_classes, pattern='threshold', seed=seed)
    val = SyntheticSeqClassDataset(n=80, seq_len=seq_len, vocab_size=vocab_size, n_classes=n_classes, pattern='threshold', seed=seed + 1)
    ld_tr = DataLoader(train, batch_size=16, shuffle=True)
    ld_va = DataLoader(val, batch_size=32)

    budget_gb = 0.10
    base_uct = ToyTransformer(vocab_size=vocab_size, d_model=64, n_layers=4, n_heads=4, d_ff=128, max_len=seq_len, task='cls', n_classes=n_classes).to(dev)
    uct = uct_wrap(base_uct, budget=int(budget_gb * (1024 ** 3)), rank=1, bits=3, planner='greedy', opt_quant=True, controller=True, profile={"batch_size": 16, "seq_len": seq_len})
    opt_uct = AnyPrecisionAdamW(uct.parameters(), lr=1e-3, ap_bits=8)

    base_lora = ToyTransformer(vocab_size=vocab_size, d_model=64, n_layers=4, n_heads=4, d_ff=128, max_len=seq_len, task='cls', n_classes=n_classes).to(dev)
    lora = apply_lora_to_toy(base_lora, r=4, alpha=8, dropout=0.05)
    opt_lora = torch.optim.AdamW(filter(lambda p: p.requires_grad, lora.parameters()), lr=1e-3)

    for epoch in range(2):
        tl, ta, tps = train_one_epoch(uct, opt_uct, ld_tr, dev, controller=uct)
        vl, va, y_true_uct, y_pred_uct = evaluate(uct, ld_va, dev)
        print(f"[UCT][Epoch {epoch + 1}] TrainLoss={tl:.4f}, ValLoss={vl:.4f}, ValAcc={va:.3f}, PeakMem={peak_mem_gb():.3f} GB, TPS={tps:.1f}")
    for epoch in range(2):
        tl, ta, tps = train_one_epoch(lora, opt_lora, ld_tr, dev, controller=None)
        vl, va, y_true_lora, y_pred_lora = evaluate(lora, ld_va, dev)
        print(f"[LoRA][Epoch {epoch + 1}] TrainLoss={tl:.4f}, ValLoss={vl:.4f}, ValAcc={va:.3f}, PeakMem={peak_mem_gb():.3f} GB, TPS={tps:.1f}")

    plot_loss_curve({'UCT': [vl]}, title='UCT Validation Loss', filename='training_loss_uct.pdf')
    plot_loss_curve({'LoRA': [vl]}, title='LoRA Validation Loss', filename='training_loss_lora.pdf')
    plot_confusion(y_true_uct, y_pred_uct, title='UCT Confusion', filename='confusion_matrix_uct.pdf')
    plot_confusion(y_true_lora, y_pred_lora, title='LoRA Confusion', filename='confusion_matrix_lora.pdf')

    print("===== QUICK TEST END =====\n")
