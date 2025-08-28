import os
import sys
import yaml

from .evaluate import quick_test, run_experiment1_toy, run_experiment2_toy, run_experiment3_toy


CONFIG_PATH = os.path.join('config', 'uct_config.yaml')


def load_config(path: str):
    if not os.path.exists(path):
        return {
            'run': {
                'quick_test': True,
                'experiment1_toy': False,
                'experiment2_toy': False,
                'experiment3_toy': False
            },
            'seed': 0
        }
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config(CONFIG_PATH)
    run_flags = cfg.get('run', {})
    seed = cfg.get('seed', 0)

    print("Unified Compressed Training (UCT) — Experiments")
    print(f"Loaded config from {CONFIG_PATH} (exists={os.path.exists(CONFIG_PATH)})")

    if run_flags.get('quick_test', True):
        quick_test()

    if run_flags.get('experiment1_toy', False):
        run_experiment1_toy(seed=seed)

    if run_flags.get('experiment2_toy', False):
        run_experiment2_toy(seed=seed)

    if run_flags.get('experiment3_toy', False):
        run_experiment3_toy(seed=seed)


if __name__ == '__main__':
    main()
