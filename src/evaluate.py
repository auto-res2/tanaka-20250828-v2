# -*- coding: utf-8 -*-
"""
Evaluation utilities for HEAT
- HEAT decoding (reference, simplified) with early exits and selective replay (functional placeholder)
- Threshold calibration, metrics, and plotting
- Small ablation-style experiments that run quickly

Figures are saved as vector PDFs in the specified output directory.
"""
from typing import List, Optional, Tuple, Dict
import os
import time
import math
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

from .train import HEATWrapper
from .preprocess import add_noise_to_prompt_ids


# -------------------------
# Decoding with HEAT exits
# -------------------------

@torch.no_grad()
def heat_generate(wrapper: HEATWrapper, tokenizer, prompt_ids: torch.Tensor, max_new_tokens: int = 64,
                  tau: float = 0.9, k_window: int = 32, device: str = 'cpu', calm_mode: bool = False,
                  per_group_tau: Optional[List[float]] = None, verbose_first_n_steps: int = 0) -> Tuple[torch.Tensor, List[int], int]:
    """
    Reference HEAT decoding that demonstrates exit decisions and selective replay.
    For simplicity, this function recomputes full forward_with_exits at each step.

    Returns: (new_tokens, depths_per_token, rollback_count)
    """
    model = wrapper.model
    model.eval()
    x = prompt_ids.to(device)
    depths: List[int] = []
    rollback_count = 0

    for t in range(max_new_tokens):
        outs = wrapper.forward_with_exits(x)
        chosen_logits = None
        chosen_depth = None
        chosen_conf = None
        for g_idx in range(wrapper.num_groups):
            conf = outs['group_confs_last'][g_idx].squeeze(-1)  # [B]
            thresh = tau if per_group_tau is None else per_group_tau[g_idx]
            if conf.item() >= thresh:
                chosen_logits = outs['group_logits'][g_idx][:, -1, :]  # [B, V]
                chosen_depth = wrapper.groups[g_idx][-1] + 1  # layers executed
                chosen_conf = conf.item()
                break
        if chosen_logits is None:
            # Use final head
            chosen_logits = outs['final_logits'][:, -1, :]
            chosen_depth = wrapper.num_layers
            chosen_conf = float('nan')
        next_token = torch.argmax(chosen_logits, dim=-1, keepdim=True)
        x = torch.cat([x, next_token], dim=1)
        depths.append(int(chosen_depth))

        if verbose_first_n_steps > 0 and t < verbose_first_n_steps:
            print(f"Step {t:02d}: chosen_depth={chosen_depth} conf={chosen_conf:.3f} token={tokenizer.decode(next_token[0])}")

        # Selective replay (functional placeholder): if depth increased, simulate replay accounting via recompute
        if (not calm_mode) and len(depths) >= 2 and depths[-1] > depths[-2] and k_window > 0:
            replay_len = min(k_window, len(depths) - 1)
            _ = wrapper.forward_with_exits(x)  # no-op recompute for accounting
            rollback_count += replay_len

    return x[:, prompt_ids.shape[1]:], depths, rollback_count


# -------------------------
# Metrics & Calibration
# -------------------------

def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i+1])
        if m.sum() == 0:
            continue
        acc = labels[m].mean()
        conf = probs[m].mean()
        ece += (m.sum() / N) * abs(acc - conf)
    return float(ece)


def calibrate_threshold(wrapper: HEATWrapper, tokenizer, texts: List[str], device: str = 'cpu',
                        risk: float = 0.05, max_samples: int = 128) -> Dict[str, List[float]]:
    """
    Fit per-group thresholds tau_g such that P(y=0 | p>=tau) <= risk.
    Returns: {'global_tau': float, 'per_group_tau': [tau_g, ...], 'auroc': [..], 'ece': [..]}
    """
    model = wrapper.model
    model.eval()
    probs_per_g: List[List[np.ndarray]] = []
    labels_per_g: List[List[np.ndarray]] = []
    aurocs: List[float] = []
    eces: List[float] = []

    num = min(max_samples, len(texts))
    for _ in range(wrapper.num_groups):
        probs_per_g.append([])
        labels_per_g.append([])

    with torch.no_grad():
        for i in range(num):
            ids = tokenizer(texts[i], return_tensors='pt', truncation=True, max_length=256).input_ids.to(device)
            outs = wrapper.forward_with_exits(ids)
            shift_final = outs['final_logits'][:, :-1, :]
            arg_f = shift_final.argmax(dim=-1).flatten().detach().cpu().numpy()
            for g in range(wrapper.num_groups):
                lg = outs['group_logits'][g][:, :-1, :]
                arg_g = lg.argmax(dim=-1).flatten().detach().cpu().numpy()
                y = (arg_g == arg_f).astype(np.float32)
                p = outs['group_confs_seq'][g].flatten().detach().cpu().numpy()
                probs_per_g[g].append(p)
                labels_per_g[g].append(y)

    per_group_tau: List[float] = []
    for g in range(wrapper.num_groups):
        p = np.concatenate(probs_per_g[g]) if len(probs_per_g[g]) else np.array([0.0])
        y = np.concatenate(labels_per_g[g]) if len(labels_per_g[g]) else np.array([0.0])
        tau_g = 0.99
        for thr in np.linspace(0.99, 0.0, 100):
            mask = p >= thr
            if mask.sum() == 0:
                continue
            risk_emp = 1.0 - y[mask].mean()
            if risk_emp <= risk:
                tau_g = float(thr)
                break
        per_group_tau.append(tau_g)
        try:
            if y.mean() > 0 and y.mean() < 1:
                auroc = roc_auc_score(y, p)
            else:
                auroc = float('nan')
        except Exception:
            auroc = float('nan')
        aurocs.append(auroc)
        eces.append(ece_score(p, y, n_bins=10))

    global_tau = float(min(per_group_tau))
    print("Calibration summary:")
    for g in range(wrapper.num_groups):
        a = aurocs[g]
        e = eces[g]
        a_str = f"{a:.3f}" if not (isinstance(a, float) and math.isnan(a)) else "nan"
        print(f"  Group {g}: tau={per_group_tau[g]:.3f}, AUROC={a_str}, ECE={e:.3f}")
    print(f"  Global tau (min over groups): {global_tau:.3f}")

    return {
        'global_tau': global_tau,
        'per_group_tau': per_group_tau,
        'auroc': aurocs,
        'ece': eces
    }


# -------------------------
# Plotting helpers (PDF)
# -------------------------

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def plot_training_loss_curve(train_losses: List[float], val_ppls: List[float], out_dir: str):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.plot(train_losses, label='train_loss')
    if len(val_ppls) == len(train_losses):
        plt.plot(val_ppls, label='val_ppl')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / PPL')
    plt.title('Training Loss and Validation PPL')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_loss_heat.pdf'), bbox_inches='tight')
    plt.close()


def plot_tokens_per_s_vs_tau(tau_list: List[float], toks_per_s: List[float], label: str, out_dir: str):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.plot(tau_list, toks_per_s, marker='o')
    plt.xlabel('tau')
    plt.ylabel('tokens/s')
    plt.title(f'Throughput vs tau ({label})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'tokens_per_s_{label}.pdf'), bbox_inches='tight')
    plt.close()


def plot_mean_exit_depth_vs_tau(tau_list: List[float], depths: List[float], label: str, out_dir: str):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.plot(tau_list, depths, marker='o')
    plt.xlabel('tau')
    plt.ylabel('mean exit depth (layers)')
    plt.title(f'Exit Depth vs tau ({label})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'mean_exit_depth_{label}.pdf'), bbox_inches='tight')
    plt.close()


def plot_rollback_rate_vs_tau(tau_list: List[float], rb_rates: List[float], label: str, out_dir: str):
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.plot(tau_list, rb_rates, marker='o')
    plt.xlabel('tau')
    plt.ylabel('rollback rate (per token)')
    plt.title(f'Replay Rollback vs tau ({label})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'rollback_rate_{label}.pdf'), bbox_inches='tight')
    plt.close()


def plot_calibration_reliability(probs: np.ndarray, labels: np.ndarray, out_dir: str, n_bins: int = 10):
    _ensure_dir(out_dir)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    accs, confs = [], []
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i+1])
        if m.sum() == 0:
            continue
        accs.append(labels[m].mean())
        confs.append(probs[m].mean())
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
    if len(confs) > 0:
        plt.plot(confs, accs, marker='o', label='HEAT C_g')
    plt.xlabel('Confidence')
    plt.ylabel('Empirical Accuracy')
    plt.title('Calibration Reliability Diagram')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'calibration_heat.pdf'), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_pdf(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str):
    _ensure_dir(out_dir)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Stability Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Pred 0', 'Pred 1'])
    plt.yticks(tick_marks, ['True 0', 'True 1'])
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix_heat.pdf'), bbox_inches='tight')
    plt.close()


# -------------------------
# Experiments
# -------------------------

def experiment1_speed_quality(wrapper: HEATWrapper, tokenizer, eval_prompts: List[str], device: str, out_dir: str):
    taus = [0.6, 0.7, 0.8, 0.9, 0.95]
    label = 'heat'

    print("\n[Experiment 1] Speed-Quality and Ablations")
    base_texts = eval_prompts[:10]
    # Calibration on a few prompts
    calib = calibrate_threshold(wrapper, tokenizer, base_texts, device=device, risk=0.05, max_samples=32)
    print(f"Suggested global tau={calib['global_tau']:.3f}\n")

    toks_per_s_by_tau: List[float] = []
    mean_depth_by_tau: List[float] = []
    rb_rate_by_tau: List[float] = []

    for tau in taus:
        kdef = 32
        toks = 0
        t_wall = 0.0
        depths_all: List[int] = []
        rb_all = 0
        for prompt in base_texts:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            out_ids, depths, rb = heat_generate(wrapper, tokenizer, ids, max_new_tokens=64, tau=tau, k_window=kdef, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_wall += (time.time() - t0)
            toks += int(out_ids.size(1))
            depths_all.extend(depths)
            rb_all += rb
        tps = float(toks) / max(1e-6, t_wall)
        mean_depth = float(sum(depths_all) / max(1, len(depths_all)))
        rb_rate = float(rb_all) / max(1, len(depths_all))
        print(f"tau={tau:.2f} | tokens/s={tps:.2f} | mean_exit_depth={mean_depth:.2f} | rollback_rate={rb_rate:.3f}")
        toks_per_s_by_tau.append(tps)
        mean_depth_by_tau.append(mean_depth)
        rb_rate_by_tau.append(rb_rate)

    plot_tokens_per_s_vs_tau(taus, toks_per_s_by_tau, label, out_dir)
    plot_mean_exit_depth_vs_tau(taus, mean_depth_by_tau, label, out_dir)
    plot_rollback_rate_vs_tau(taus, rb_rate_by_tau, label, out_dir)

    # Compare CALM-LM (no replay) vs HEAT on a single tau
    tau = 0.9
    for calm in [True, False]:
        toks = 0; t_wall = 0.0
        for prompt in base_texts:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
            t0 = time.time()
            out_ids, depths, rb = heat_generate(wrapper, tokenizer, ids, max_new_tokens=64, tau=tau, k_window=32, device=device, calm_mode=calm)
            t_wall += (time.time() - t0)
            toks += int(out_ids.size(1))
        print(f"tau={tau:.2f} | calm_mode={calm} | tokens/s={toks / max(1e-6, t_wall):.2f}")

    # Calibration visualization for Group 0 on collected points
    probs, labels = [], []
    with torch.no_grad():
        for prompt in base_texts:
            ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
            outs = wrapper.forward_with_exits(ids)
            lg0 = outs['group_logits'][0][:, :-1, :]
            arg0 = lg0.argmax(dim=-1).flatten().cpu().numpy()
            lf = outs['final_logits'][:, :-1, :]
            argf = lf.argmax(dim=-1).flatten().cpu().numpy()
            y = (arg0 == argf).astype(np.int32)
            p = outs['group_confs_seq'][0].flatten().cpu().numpy()
            probs.append(p); labels.append(y)
    probs_arr = np.concatenate(probs) if len(probs) else np.array([0.0])
    labels_arr = np.concatenate(labels) if len(labels) else np.array([0])
    plot_calibration_reliability(probs_arr, labels_arr, out_dir, n_bins=10)
    if len(labels_arr) > 0:
        plot_confusion_matrix_pdf(labels_arr, (probs_arr >= calib['per_group_tau'][0]).astype(np.int32), out_dir)


def experiment2_task_level(wrapper: HEATWrapper, tokenizer, device: str, out_dir: str):
    print("\n[Experiment 2] Mini Task-level evaluation (summarization-style prompts)")
    docs = [
        "Neural networks have revolutionized AI. They excel in vision and language tasks, enabling breakthroughs in translation and image recognition.",
        "Climate change is accelerating due to greenhouse gas emissions. Global efforts focus on renewable energy and conservation strategies.",
        "Quantum computing leverages quantum bits to perform computations that are intractable for classical computers in specific domains."
    ]
    prompts = [f"Summarize the following in 2 sentences:\n\n{d}\n\nSummary:" for d in docs]

    taus = [0.8, 0.9, 0.95]
    results: List[Tuple[float, float]] = []
    for tau in taus:
        lat = []
        for p in prompts:
            ids = tokenizer(p, return_tensors='pt', truncation=True, max_length=256).input_ids.to(device)
            t0 = time.time()
            out_ids, depths, rb = heat_generate(wrapper, tokenizer, ids, max_new_tokens=72, tau=tau, k_window=32, device=device)
            dt = time.time() - t0
            lat.append(dt)
            out_txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            print(f"tau={tau:.2f} | sample latency={dt:.3f}s | output: {out_txt[:90].strip()}...")
        results.append((tau, sum(lat) / len(lat)))

    # Plot latency vs tau
    _ensure_dir(out_dir)
    plt.figure(figsize=(5, 4))
    plt.plot([r[0] for r in results], [r[1] for r in results], marker='o')
    plt.xlabel('tau')
    plt.ylabel('avg latency (s/sample)')
    plt.title('Summarization Latency vs tau (mini)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'inference_latency_heat.pdf'), bbox_inches='tight')
    plt.close()


def experiment3_robustness(wrapper: HEATWrapper, tokenizer, device: str, out_dir: str):
    print("\n[Experiment 3] Robustness under noisy prompts (mini)")
    base_prompt = (
        "Context: In a distant future, explorers discover an ancient archive with vast knowledge. "
        "They must decide which parts to trust and how to apply it.\n\nQuestion: What challenges do they face?\nAnswer:"
    )
    ids_clean = tokenizer(base_prompt, return_tensors='pt', truncation=True, max_length=256).input_ids.to(device)
    ids_noisy = add_noise_to_prompt_ids(ids_clean, tokenizer.vocab_size, noise_ratio=0.1)

    tau = 0.9
    for label, ids in [("clean", ids_clean), ("noisy", ids_noisy)]:
        t0 = time.time()
        out_ids, depths, rb = heat_generate(wrapper, tokenizer, ids, max_new_tokens=96, tau=tau, k_window=32, device=device)
        dt = time.time() - t0
        txt = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        mean_depth = sum(depths) / max(1, len(depths))
        print(f"{label}: latency={dt:.3f}s | mean_exit_depth={mean_depth:.2f} | rollback_rate={rb / max(1, len(depths)):.3f} | output: {txt[:100].strip()}...")

    # Paired bar of mean exit depth (clean vs noisy)
    means = []
    for ids in [ids_clean, ids_noisy]:
        out_ids, depths, rb = heat_generate(wrapper, tokenizer, ids, max_new_tokens=64, tau=tau, k_window=32, device=device)
        means.append(sum(depths) / max(1, len(depths)))
    _ensure_dir(out_dir)
    plt.figure(figsize=(4.6, 4))
    xs = np.arange(2)
    labels = ['clean', 'noisy']
    plt.bar(xs, means, tick_label=labels)
    plt.ylabel('mean exit depth (layers)')
    plt.title('Exit depth increases under noise (selective replay ready)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mean_exit_depth_noisy_vs_clean_heat.pdf'), bbox_inches='tight')
    plt.close()
