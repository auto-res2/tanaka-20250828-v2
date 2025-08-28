# -*- coding: utf-8 -*-
"""
Main entry point for HEAT experiments.
Run from project root:
  python -m src.main --config config/config.yaml

This script orchestrates:
- Data preparation
- HEAT head training
- Calibration + speed/quality experiments
- Mini task and robustness demos
- Saves all figures as PDF in .research/iteration1/images
- Saves HEAT wrapper weights in models/
"""
import os
import argparse
import yaml

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .preprocess import set_seed, get_device, prepare_text_corpus
from .train import HEATWrapper, train_heat_heads
from .evaluate import (
    experiment1_speed_quality,
    experiment2_task_level,
    experiment3_robustness,
    plot_training_loss_curve,
    heat_generate,
)


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dirs():
    os.makedirs('.research/iteration1/images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {
        'seed': 42,
        'device': 'auto',
        'model_name': 'sshleifer/tiny-gpt2',
        'group_size': 2,
        'training': {
            'epochs': 1,
            'batch_size': 4,
            'lr': 5e-5,
            'max_len': 128,
            'n_train_samples': 32,
            'n_val_samples': 16
        },
        'eval': {
            'max_new_tokens': 64
        }
    }

    ensure_dirs()
    set_seed(cfg.get('seed', 42))
    device = get_device() if cfg.get('device', 'auto') == 'auto' else torch.device(cfg['device'])
    print(f"Device: {device}")

    # Load tokenizer and model
    model_name = cfg.get('model_name', 'sshleifer/tiny-gpt2')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Build HEAT wrapper
    group_size = int(cfg.get('group_size', 2))
    wrapper = HEATWrapper(base, group_size=group_size).to(device)

    # Prepare tiny corpora
    n_train = int(cfg.get('training', {}).get('n_train_samples', 32))
    n_val = int(cfg.get('training', {}).get('n_val_samples', 16))
    max_len = int(cfg.get('training', {}).get('max_len', 128))
    train_texts = prepare_text_corpus(tokenizer, n_samples=n_train, max_len=max_len)
    val_texts = prepare_text_corpus(tokenizer, n_samples=n_val, max_len=max_len)

    # Train HEAT heads
    print("\n[Training] Start")
    epochs = int(cfg.get('training', {}).get('epochs', 1))
    batch_size = int(cfg.get('training', {}).get('batch_size', 4))
    lr = float(cfg.get('training', {}).get('lr', 5e-5))
    train_losses, val_ppls = train_heat_heads(wrapper, tokenizer, train_texts, val_texts,
                                              epochs=epochs, batch_size=batch_size, max_len=max_len, lr=lr,
                                              device=str(device))
    # Save training curve
    plot_training_loss_curve(train_losses, val_ppls, out_dir='.research/iteration1/images')

    # Save HEAT wrapper weights (exit and conf heads included)
    save_path = os.path.join('models', 'heat_wrapper.pt')
    torch.save(wrapper.state_dict(), save_path)
    print(f"[Training] Saved HEAT wrapper weights to {save_path}")

    # Experiments
    print("\n[Evaluation] Experiment 1: Speed/Quality")
    experiment1_speed_quality(wrapper, tokenizer, val_texts, device=str(device), out_dir='.research/iteration1/images')

    print("\n[Evaluation] Experiment 2: Mini Task-level")
    experiment2_task_level(wrapper, tokenizer, device=str(device), out_dir='.research/iteration1/images')

    print("\n[Evaluation] Experiment 3: Robustness")
    experiment3_robustness(wrapper, tokenizer, device=str(device), out_dir='.research/iteration1/images')

    # Quick verbose generation demo
    print("\n[Demo] Verbose HEAT decoding (first 5 steps)")
    demo_prompt = "Explain HEAT early-exit in one sentence:"
    ids = tokenizer(demo_prompt, return_tensors='pt').input_ids.to(device)
    out_ids, depths, rb = heat_generate(wrapper, tokenizer, ids, max_new_tokens=24, tau=0.9, k_window=16, device=str(device), verbose_first_n_steps=5)
    print(f"Generated: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")
    print(f"Mean exit depth: {sum(depths)/len(depths):.2f} | Rollback rate: {rb/len(depths):.3f}")

    # List saved figures
    print("\nSaved PDF figures in .research/iteration1/images:")
    for f in [
        'training_loss_heat.pdf',
        'tokens_per_s_heat.pdf',
        'mean_exit_depth_heat.pdf',
        'rollback_rate_heat.pdf',
        'calibration_heat.pdf',
        'confusion_matrix_heat.pdf',
        'inference_latency_heat.pdf',
        'mean_exit_depth_noisy_vs_clean_heat.pdf'
    ]:
        p = os.path.join('.research/iteration1/images', f)
        print(f"  - {p} (exists: {os.path.exists(p)})")


if __name__ == '__main__':
    main()
